"""
CSV処理ツール
巨大なCSVファイルを効率的に処理し、企業IDとセグメント名を含むカラム名で出力するツール

主な機能:
- 巨大なCSVファイルの効率的な処理
- 設定ファイルからの設定読み込み
- 並列処理による高速化
- エラーハンドリングとログ機能
- 企業IDとセグメント名を含むカラム名生成（デフォルト）
- L5/L6プロダクトレベル設定対応

カラム名生成:
- 連結系: median_[FACTSET_ENTITY_ID], std_[FACTSET_ENTITY_ID]
- セグメント系: median_[FACTSET_ENTITY_ID]_[セグメント名], std_[FACTSET_ENTITY_ID]_[セグメント名]

使用方法:
1. 設定ファイル (gppm_config.yml) で出力設定を設定
2. CSVProcessor を初期化（ConfigManagerを使用）
3. process_csv_to_pkl() でCSV処理を実行

例:
    from gppm.cli.csv_to_pkl import CSVProcessor
    from gppm.core.config_manager import ConfigManager
    
    # ConfigManagerを使用して設定を読み込み
    config_manager = ConfigManager()
    processor = CSVProcessor(config_manager=config_manager)
    
    # CSV処理の実行（出力パスは設定ファイルから自動取得）
    result = processor.process_csv_to_pkl(
        csv_path="input.csv",
        out_path="output.pkl"  # 設定ファイルの出力設定を使用する場合は任意
    )
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import sys
from dataclasses import dataclass
import time
from tqdm import tqdm
import argparse

from gppm.core.config_manager import ConfigManager, get_logger
import re


@dataclass
class ProcessingConfig:
    """処理設定を管理するデータクラス"""
    chunk_size: int = 1000
    max_workers: int = 4
    memory_limit_mb: int = 1024
    dtype: str = 'float64'
    engine: str = 'c'
    comment_char: str = '#'
    skip_blank_lines: bool = True
    # 設定ファイルパス
    config_file_path: Optional[str] = None
    # カラム名生成設定
    product_level: str = 'L6'
    
    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ProcessingConfig':
        """ConfigManagerから設定を読み込んでProcessingConfigを作成"""
        config = config_manager.get_config()
        
        return cls(
            chunk_size=1000,  # デフォルト値
            max_workers=config.data_processing.parallel_workers,
            memory_limit_mb=config.data_processing.memory_limit_mb,
            dtype=config.data_processing.data_type,
            engine='c',  # デフォルト値
            comment_char='#',  # デフォルト値
            skip_blank_lines=True,  # デフォルト値
            config_file_path=None,  # デフォルト値
            product_level=config.product_level.level
        )


class ColumnNameGenerator:
    """カラム名生成クラス"""
    
    def __init__(self, product_level: str = 'L6'):
        """
        初期化
        
        Args:
            product_level: 製品レベル（L6, L5）
        """
        self.product_level = product_level
    
    def extract_entity_info(self, column_name: str) -> Tuple[Optional[str], Optional[str]]:
        """
        カラム名から企業IDとセグメント名を抽出
        
        Args:
            column_name: カラム名
            
        Returns:
            (企業ID, セグメント名)のタプル
        """
        # FACTSET_ENTITY_ID形式のパターン（連結系）
        entity_pattern = r'^([A-Z0-9]+)$'
        entity_match = re.match(entity_pattern, column_name)
        if entity_match:
            return entity_match.group(1), None
        
        # FACTSET_ENTITY_ID_SEGMENT_NAME形式のパターン（セグメント系）
        segment_pattern = r'^([A-Z0-9]+)_(.+)$'
        segment_match = re.match(segment_pattern, column_name)
        if segment_match:
            return segment_match.group(1), segment_match.group(2)
        
        # 製品名を含むパターン（L5/L6レベル対応）
        # PRODUCT_L5_NAME や PRODUCT_L6_NAME の形式
        product_pattern = r'^([A-Z0-9]+)_([A-Z0-9]+)_(.+)$'
        product_match = re.match(product_pattern, column_name)
        if product_match:
            return product_match.group(1), f"{product_match.group(2)}_{product_match.group(3)}"
        
        return None, None
    
    def generate_column_name(self, original_name: str, stat_type: str, index: int) -> str:
        """
        新しいカラム名を生成
        
        Args:
            original_name: 元のカラム名
            stat_type: 統計タイプ（median, std）
            index: インデックス
            
        Returns:
            生成されたカラム名
        """
        entity_id, segment_name = self.extract_entity_info(original_name)
        
        if entity_id is None:
            # 企業IDが抽出できない場合は従来の形式
            return f"{stat_type}_{index}"
        
        if segment_name is None:
            # 連結系の場合
            return f"{stat_type}_{entity_id}"
        else:
            # セグメント系の場合
            # 製品レベルに応じてセグメント名を調整
            if self.product_level == "L5" and "L6" in segment_name:
                # L5レベルに集約する場合、L6の詳細を除去
                segment_name = segment_name.replace("_L6_", "_L5_")
            elif self.product_level == "L6" and "L5" in segment_name:
                # L6レベルを使用する場合、L5をL6に変更
                segment_name = segment_name.replace("_L5_", "_L6_")
            
            return f"{stat_type}_{entity_id}_{segment_name}"
    
    def get_product_level_suffix(self) -> str:
        """
        製品レベルに応じたサフィックスを取得
        
        Returns:
            製品レベルサフィックス
        """
        return f"_{self.product_level}" if self.product_level in ["L5", "L6"] else ""
    
    def is_consolidated_data(self, column_name: str) -> bool:
        """
        連結データかどうかを判定
        
        Args:
            column_name: カラム名
            
        Returns:
            連結データの場合True
        """
        entity_id, segment_name = self.extract_entity_info(column_name)
        return entity_id is not None and segment_name is None
    
    def is_segment_data(self, column_name: str) -> bool:
        """
        セグメントデータかどうかを判定
        
        Args:
            column_name: カラム名
            
        Returns:
            セグメントデータの場合True
        """
        entity_id, segment_name = self.extract_entity_info(column_name)
        return entity_id is not None and segment_name is not None


class CSVProcessor:
    """CSV処理と変分推論結果統合を管理するクラス"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None, config_manager: Optional[ConfigManager] = None):
        """
        初期化
        
        Args:
            config: 処理設定。Noneの場合はConfigManagerから読み込み
            config_manager: 設定管理オブジェクト。Noneの場合は新規作成
        """
        if config is None:
            if config_manager is None:
                config_manager = ConfigManager()
            self.config = ProcessingConfig.from_config_manager(config_manager)
            self.config_manager = config_manager
        else:
            self.config = config
            self.config_manager = config_manager or ConfigManager()
        
        self.logger = get_logger(__name__)
        
        # カラム名生成器を初期化
        self.column_name_generator = ColumnNameGenerator(
            product_level=self.config.product_level
        )
    
    
    def _validate_inputs(self, csv_path: Union[str, Path], out_path: Union[str, Path]) -> None:
        """
        入力パラメータの検証
        
        Args:
            csv_path: 入力CSVファイルのパス
            out_path: 出力PKLファイルのパス
            
        Raises:
            FileNotFoundError: CSVファイルが存在しない場合
            ValueError: パスが無効な場合
        """
        csv_path = Path(csv_path)
        out_path = Path(out_path)
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSVファイルが見つかりません: {csv_path}")
        
        if not csv_path.suffix.lower() == '.csv':
            raise ValueError(f"CSVファイルではありません: {csv_path}")
        
        # 出力ディレクトリを作成
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"入力ファイル: {csv_path}")
        self.logger.info(f"出力ファイル: {out_path}")
    
    def _get_column_info(self, csv_path: Path) -> Tuple[List[str], int]:
        """
        CSVファイルの列情報を取得
        
        Args:
            csv_path: CSVファイルのパス
            
        Returns:
            Tuple[列名リスト, 行数]
        """
        try:
            # 最初の数行を読み込んで列情報を取得
            sample_df = pd.read_csv(
                csv_path,
                comment=self.config.comment_char,
                skip_blank_lines=self.config.skip_blank_lines,
                nrows=10,
                engine=self.config.engine
            )
            
            columns = sample_df.columns.tolist()
            
            # 行数を取得（効率的に）
            with open(csv_path, 'r') as f:
                total_rows = sum(1 for line in f if not line.strip().startswith(self.config.comment_char))
            
            self.logger.info(f"列数: {len(columns)}, 推定行数: {total_rows}")
            return columns, total_rows
            
        except Exception as e:
            self.logger.error(f"列情報の取得に失敗: {e}")
            raise
    
    def _process_chunk(self, chunk_data: Tuple[List[str], Path, ProcessingConfig, str]) -> pd.Series:
        """
        チャンクデータを処理して統計値を計算
        
        Args:
            chunk_data: (列名リスト, CSVパス, 設定, 製品レベル)のタプル
            
        Returns:
            処理結果のSeries
        """
        columns, csv_path, config, product_level = chunk_data
        
        # カラム名生成器をローカルで作成
        column_name_generator = ColumnNameGenerator(
            product_level=product_level
        )
        
        try:
            # チャンクを読み込み
            chunk_df = pd.read_csv(
                csv_path,
                usecols=columns,
                comment=config.comment_char,
                skip_blank_lines=config.skip_blank_lines,
                engine=config.engine,
                dtype=config.dtype
            )
            
            if chunk_df.empty:
                return pd.Series(dtype=config.dtype)
            
            # 統計値を計算
            median_series = chunk_df.median(axis=1)
            std_series = chunk_df.std(axis=1)
            
            # 新しいカラム名を生成
            median_names = []
            std_names = []
            
            for i, (_, row) in enumerate(chunk_df.iterrows()):
                # 最初の非nullカラム名を取得（企業ID/セグメント名の推定に使用）
                first_valid_col = None
                for col in columns:
                    if col in chunk_df.columns and not pd.isna(chunk_df.loc[i, col]):
                        first_valid_col = col
                        break
                
                if first_valid_col is None:
                    first_valid_col = columns[0] if columns else f"col_{i}"
                
                median_name = column_name_generator.generate_column_name(first_valid_col, "median", i)
                std_name = column_name_generator.generate_column_name(first_valid_col, "std", i)
                
                median_names.append(median_name)
                std_names.append(std_name)
            
            # 結果を結合
            result = pd.concat([median_series, std_series], axis=0)
            result.index = median_names + std_names
            
            return result
            
        except Exception as e:
            self.logger.error(f"チャンク処理エラー: {e}")
            return pd.Series(dtype=config.dtype)
    
    def _split_columns(self, columns: List[str], num_chunks: int) -> List[List[str]]:
        """
        列を指定数に分割
        
        Args:
            columns: 列名リスト
            num_chunks: 分割数
            
        Returns:
            分割された列名リストのリスト
        """
        if num_chunks <= 0:
            raise ValueError("分割数は1以上である必要があります")
        
        chunk_size = max(1, len(columns) // num_chunks)
        return [columns[i:i + chunk_size] for i in range(0, len(columns), chunk_size)]
    
    def process_csv_to_pkl(
        self, 
        csv_path: Union[str, Path], 
        out_path: Union[str, Path],
        num_chunks: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.Series:
        """
        CSVファイルをPKLファイルに変換
        
        Args:
            csv_path: 入力CSVファイルのパス
            out_path: 出力PKLファイルのパス
            num_chunks: 処理チャンク数（Noneの場合は自動決定）
            show_progress: 進捗表示の有無
            
        Returns:
            処理結果のSeries
        """
        start_time = time.time()
        
        # 入力検証
        self._validate_inputs(csv_path, out_path)
        
        # 列情報取得
        columns, total_rows = self._get_column_info(Path(csv_path))
        
        # チャンク数決定
        if num_chunks is None:
            num_chunks = min(self.config.max_workers * 2, len(columns))
        
        # 列を分割
        column_chunks = self._split_columns(columns, num_chunks)
        self.logger.info(f"処理チャンク数: {len(column_chunks)}")
        
        # 並列処理
        results = []
        with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
            # タスクを投入
            future_to_chunk = {
                executor.submit(
                    self._process_chunk, 
                    (chunk, Path(csv_path), self.config, self.config.product_level)
                ): i for i, chunk in enumerate(column_chunks)
            }
            
            # 進捗表示付きで結果を収集
            if show_progress:
                futures = tqdm(
                    as_completed(future_to_chunk), 
                    total=len(future_to_chunk),
                    desc="CSV処理中"
                )
            else:
                futures = as_completed(future_to_chunk)
            
            for future in futures:
                try:
                    result = future.result()
                    if not result.empty:
                        results.append(result)
                except Exception as e:
                    chunk_idx = future_to_chunk[future]
                    self.logger.error(f"チャンク {chunk_idx} の処理に失敗: {e}")
        
        # 結果を結合
        if not results:
            self.logger.warning("処理結果が空です")
            final_result = pd.Series(dtype=self.config.dtype)
        else:
            final_result = pd.concat(results, axis=0)
        
        # PKLファイルに保存
        final_result.to_pickle(out_path)
        
        # 処理時間と結果情報をログ出力
        processing_time = time.time() - start_time
        self.logger.info(f"処理完了: {processing_time:.2f}秒")
        self.logger.info(f"結果サイズ: {len(final_result)} 要素")
        self.logger.info(f"出力ファイル: {out_path}")
        
        # カラム名生成設定の情報をログ出力
        self.logger.info(f"カラム名生成設定:")
        self.logger.info(f"- 製品レベル: {self.config.product_level}")
        
        # 生成されたカラム名の例を表示
        if len(final_result) > 0:
            sample_names = list(final_result.index[:5])  # 最初の5個のカラム名
            self.logger.info(f"生成されたカラム名の例: {sample_names}")
        
        return final_result
    


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="CSV処理ツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 設定ファイルのCSVファイルパスを使用
  python csv_to_pkl.py
  
  # コマンドライン引数でCSVファイルパスを指定
  python csv_to_pkl.py --csv-file /path/to/input.csv
  
  # 出力パスも指定
  python csv_to_pkl.py --csv-file /path/to/input.csv --output /path/to/output.pkl
  
  # L5プロダクトレベルで処理
  python csv_to_pkl.py --csv-file /path/to/input.csv --product-level L5
        """
    )
    
    parser.add_argument(
        '--csv-file', '-i',
        type=str,
        help='入力CSVファイルのパス（設定ファイルで指定されていない場合に必要）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='出力PKLファイルのパス（設定ファイルの出力設定を使用する場合は省略可能）'
    )
    
    
    parser.add_argument(
        '--chunks', '-c',
        type=int,
        help='処理チャンク数（自動決定する場合は省略）'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='進捗表示を無効にする'
    )
    
    
    parser.add_argument(
        '--product-level',
        choices=['L6', 'L5'],
        help='製品レベル（L6: 最詳細、L5: 詳細サブ業界）（設定ファイルの値を上書き）'
    )
    
    return parser.parse_args()


def main():
    """メイン実行関数"""
    # コマンドライン引数を解析
    args = parse_arguments()
    
    # ConfigManagerを使用して設定を読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # プロセッサー初期化（ConfigManagerから設定を自動読み込み）
    processor = CSVProcessor(config_manager=config_manager)
    
    # コマンドライン引数で設定を上書き
    if args.product_level:
        processor.config.product_level = args.product_level
        processor.column_name_generator.product_level = args.product_level
    
    # CSVファイルパスの決定（優先順位: コマンドライン引数 > 設定ファイル）
    csv_path = args.csv_file or config.csv_processing.input_file
    if csv_path is None:
        print("エラー: CSVファイルパスが指定されていません。")
        print("設定ファイル（gppm_config.yml）のcsv_processing.input_fileにパスを設定するか、")
        print("コマンドライン引数 --csv-file でパスを指定してください。")
        sys.exit(1)
    
    # 出力パスの決定（優先順位: コマンドライン引数 > 設定ファイル）
    if args.output:
        out_path = Path(args.output)
    else:
        output_dir = Path(config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / config.output.files.csv_processed
    
    
    try:
        # CSV処理の実行
        result = processor.process_csv_to_pkl(
            csv_path=csv_path,
            out_path=out_path,
            num_chunks=args.chunks,
            show_progress=not args.no_progress
        )
        
        # 結果の表示
        print("\n=== 処理結果 ===")
        print(f"CSV処理結果: {len(result)} 要素")
        print(f"出力ファイル: {out_path}")
        print(f"カラム名生成設定:")
        print(f"- 製品レベル: {processor.config.product_level}")
        
        # 生成されたカラム名の例を表示
        if len(result) > 0:
            sample_names = list(result.index[:3])  # 最初の3個のカラム名
            print(f"生成されたカラム名の例: {sample_names}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
