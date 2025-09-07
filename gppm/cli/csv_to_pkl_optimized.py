"""
最適化されたCSVからPKL生成コード
巨大なCSVファイルからPKLファイルを効率的に生成するための最適化実装

主な機能:
- 巨大なCSVファイルの効率的な処理
- ベイズ結果の読み込みと統合
- 設定ファイルからの設定読み込み
- 並列処理による高速化
- エラーハンドリングとログ機能

使用方法:
1. 設定ファイル (gppm_config.yml) でベイズ結果のパスを設定
2. CSVToPKLProcessor を初期化
3. process_with_bayesian_integration() で統合処理を実行

例:
    from gppm.cli.csv_to_pkl_optimized import CSVToPKLProcessor, ProcessingConfig
    
    config = ProcessingConfig(config_file_path="gppm_config.yml")
    processor = CSVToPKLProcessor(config)
    results = processor.process_with_bayesian_integration(
        csv_path="input.csv",
        out_path="output.pkl"
    )
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any
import logging
import sys
from dataclasses import dataclass
import time
from tqdm import tqdm
import yaml
import pickle


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
    # ベイズ関連設定
    bayesian_result_path: Optional[str] = None
    bayesian_output_dir: Optional[str] = None
    config_file_path: Optional[str] = None


@dataclass
class BayesianConfig:
    """ベイズ推論設定を管理するデータクラス"""
    result_path: str
    output_dir: str
    model: Dict[str, Any]
    variational: Dict[str, Any]


class CSVToPKLProcessor:
    """CSVからPKLへの変換処理を管理するクラス"""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        初期化
        
        Args:
            config: 処理設定。Noneの場合はデフォルト設定を使用
        """
        self.config = config or ProcessingConfig()
        self.logger = self._setup_logger()
        self.bayesian_config = self._load_bayesian_config()
        
    def _setup_logger(self) -> logging.Logger:
        """ログ設定"""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_bayesian_config(self) -> Optional[BayesianConfig]:
        """
        設定ファイルからベイズ設定を読み込み
        
        Returns:
            ベイズ設定オブジェクト。設定ファイルが存在しない場合はNone
        """
        config_path = self.config.config_file_path or "gppm_config.yml"
        
        try:
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f)
                
                bayesian_data = config_data.get('bayesian', {})
                if bayesian_data:
                    return BayesianConfig(
                        result_path=bayesian_data.get('result_path', ''),
                        output_dir=bayesian_data.get('output_dir', ''),
                        model=bayesian_data.get('model', {}),
                        variational=bayesian_data.get('variational', {})
                    )
            else:
                self.logger.warning(f"設定ファイルが見つかりません: {config_path}")
                
        except Exception as e:
            self.logger.error(f"ベイズ設定の読み込みに失敗: {e}")
        
        return None
    
    def load_bayesian_result(self, result_path: Optional[str] = None) -> Optional[pd.Series]:
        """
        ベイズ結果を読み込み
        
        Args:
            result_path: ベイズ結果ファイルのパス。Noneの場合は設定から取得
            
        Returns:
            ベイズ結果のSeries。読み込みに失敗した場合はNone
        """
        if result_path is None:
            if self.bayesian_config is None:
                self.logger.error("ベイズ設定が読み込まれていません")
                return None
            result_path = self.bayesian_config.result_path
        
        try:
            result_path = Path(result_path)
            if not result_path.exists():
                self.logger.error(f"ベイズ結果ファイルが見つかりません: {result_path}")
                return None
            
            self.logger.info(f"ベイズ結果を読み込み中: {result_path}")
            with open(result_path, 'rb') as f:
                result = pickle.load(f)
            
            if isinstance(result, pd.Series):
                self.logger.info(f"ベイズ結果読み込み完了: {len(result)} 要素")
                return result
            else:
                self.logger.error(f"ベイズ結果の形式が不正です: {type(result)}")
                return None
                
        except Exception as e:
            self.logger.error(f"ベイズ結果の読み込みに失敗: {e}")
            return None
    
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
    
    def _process_chunk(self, chunk_data: Tuple[List[str], Path, ProcessingConfig]) -> pd.Series:
        """
        チャンクデータを処理して統計値を計算
        
        Args:
            chunk_data: (列名リスト, CSVパス, 設定)のタプル
            
        Returns:
            処理結果のSeries
        """
        columns, csv_path, config = chunk_data
        
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
            
            # 結果を結合
            result = pd.concat([median_series, std_series], axis=0)
            result.index = [f"median_{i}" for i in range(len(median_series))] + \
                          [f"std_{i}" for i in range(len(std_series))]
            
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
                    (chunk, Path(csv_path), self.config)
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
        
        return final_result
    
    def process_with_bayesian_integration(
        self,
        csv_path: Union[str, Path],
        out_path: Union[str, Path],
        bayesian_result_path: Optional[str] = None,
        num_chunks: Optional[int] = None,
        show_progress: bool = True
    ) -> Dict[str, pd.Series]:
        """
        CSV処理とベイズ結果の統合処理
        
        Args:
            csv_path: 入力CSVファイルのパス
            out_path: 出力PKLファイルのパス
            bayesian_result_path: ベイズ結果ファイルのパス
            num_chunks: 処理チャンク数
            show_progress: 進捗表示の有無
            
        Returns:
            統合結果の辞書
        """
        results = {}
        
        # ベイズ結果の読み込み
        bayesian_result = self.load_bayesian_result(bayesian_result_path)
        if bayesian_result is not None:
            results['bayesian'] = bayesian_result
            self.logger.info("ベイズ結果を統合処理に含めます")
        
        # CSV処理実行
        csv_result = self.process_csv_to_pkl(csv_path, out_path, num_chunks, show_progress)
        results['csv_processed'] = csv_result
        
        # 統合結果の保存
        if bayesian_result is not None and not csv_result.empty:
            # ベイズ結果とCSV結果を統合
            integrated_result = pd.concat([bayesian_result, csv_result], axis=0)
            integrated_path = Path(out_path).with_suffix('.integrated.pkl')
            integrated_result.to_pickle(integrated_path)
            results['integrated'] = integrated_result
            self.logger.info(f"統合結果を保存: {integrated_path}")
        
        return results


def main():
    """メイン実行関数"""
    # 設定
    config = ProcessingConfig(
        chunk_size=1000,
        max_workers=4,
        memory_limit_mb=2048,
        config_file_path="gppm_config.yml"
    )
    
    # プロセッサー初期化
    processor = CSVToPKLProcessor(config)
    
    # ファイルパス（例）
    csv_path = '/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202505/item_roic_rbics/result/Item_ROIC_Item_Beta-20250627033115.csv'
    out_path = '/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202505/stan_result_15_usd2_optimized.pkl'
    
    try:
        # 統合処理の実行
        results = processor.process_with_bayesian_integration(csv_path, out_path)
        
        # 結果の表示
        for key, result in results.items():
            if isinstance(result, pd.Series):
                print(f"{key}: {len(result)} 要素")
            else:
                print(f"{key}: {result}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
