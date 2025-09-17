"""
CSV処理ツール（改善版）
CmdStanPyの出力CSVファイルを効率的に処理し、Stanパラメータ名から意味のあるカラム名で出力するツール

主な改善点:
- 関数の分割と責任の明確化
- エラーハンドリングの強化
- メモリ効率の最適化
- テスト可能性の向上
- 設定検証の強化
- 型安全性の向上

使用方法:
1. 設定ファイル (gppm_config.yml) で出力設定を設定
2. CSVProcessor を初期化（ConfigManagerを使用）
3. process_csv_to_pkl() でCSV処理を実行

例:
    from gppm.cli.csv_to_pkl_improved import CSVProcessor
    from gppm.core.config_manager import ConfigManager
    
    # ConfigManagerを使用して設定を読み込み
    config_manager = ConfigManager()
    processor = CSVProcessor(config_manager=config_manager)
    
    # CmdStanPyの出力CSVファイルを処理
    result = processor.process_csv_to_pkl(
        csv_path="global_ppm_roic_model-20250910175117.csv",
        out_path="output.pkl"
    )
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict, Any, Protocol
import sys
from dataclasses import dataclass, field
import time
from tqdm import tqdm
import argparse
import pickle
import logging
from contextlib import contextmanager
import gc

from gppm.core.config_manager import ConfigManager, get_logger
import re


class DataLoader(Protocol):
    """データ読み込みのインターフェース"""
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """データセットファイルを読み込む"""
        ...


class PickleDataLoader:
    """Pickleファイル用のデータローダー"""
    
    def load_dataset(self, file_path: str) -> Dict[str, Any]:
        """Pickleファイルからデータセットを読み込む"""
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except (FileNotFoundError, pickle.PickleError, EOFError) as e:
            logging.warning(f"データセットファイルの読み込みに失敗: {file_path}, エラー: {e}")
            return {}


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
    config_file_path: Optional[str] = None
    product_level: str = 'L6'
    dataset_file: Optional[str] = None
    
    def __post_init__(self):
        """設定値の検証"""
        if self.chunk_size <= 0:
            raise ValueError("chunk_sizeは1以上である必要があります")
        if self.max_workers <= 0:
            raise ValueError("max_workersは1以上である必要があります")
        if self.memory_limit_mb <= 0:
            raise ValueError("memory_limit_mbは1以上である必要があります")
        if self.dtype not in ['float32', 'float64']:
            raise ValueError("dtypeは'float32'または'float64'である必要があります")
        if self.product_level not in ['L5', 'L6']:
            raise ValueError("product_levelは'L5'または'L6'である必要があります")
    
    @classmethod
    def from_config_manager(cls, config_manager: ConfigManager) -> 'ProcessingConfig':
        """ConfigManagerから設定を読み込んでProcessingConfigを作成"""
        config = config_manager.get_config()
        
        return cls(
            chunk_size=1000,
            max_workers=config.data_processing.parallel_workers,
            memory_limit_mb=config.data_processing.memory_limit_mb,
            dtype=config.data_processing.data_type,
            engine='c',
            comment_char='#',
            skip_blank_lines=True,
            config_file_path=None,
            product_level=config.product_level.level,
            dataset_file=str(config.output.directory / config.output.files.dataset)
        )


class ColumnNameGenerator:
    """カラム名生成クラス（改善版）"""
    
    def __init__(
        self, 
        product_level: str = 'L6', 
        dataset_file: Optional[str] = None,
        data_loader: Optional[DataLoader] = None
    ):
        """
        初期化
        
        Args:
            product_level: 製品レベル（L6, L5）
            dataset_file: データセットファイルパス（名前マッピング用）
            data_loader: データローダー（テスト用に注入可能）
        """
        self.product_level = product_level
        self.dataset_file = dataset_file
        self.data_loader = data_loader or PickleDataLoader()
        
        # 名前マッピング情報を読み込み
        self.product_names: List[str] = []
        self.entity_info: Dict[int, str] = {}
        self.segment_info: Dict[int, str] = {}
        self._load_name_mappings()
    
    def _load_name_mappings(self) -> None:
        """データセットファイルから名前マッピング情報を読み込み"""
        if not self.dataset_file or not Path(self.dataset_file).exists():
            return
        
        try:
            dataset = self.data_loader.load_dataset(self.dataset_file)
            
            # 製品名の読み込み
            if 'product_names' in dataset:
                self.product_names = dataset['product_names']
            
            # エンティティ情報の読み込み
            if 'entity_info' in dataset:
                entity_df = dataset['entity_info']
                if isinstance(entity_df, pd.DataFrame):
                    self.entity_info = dict(zip(
                        range(len(entity_df)), 
                        entity_df['FACTSET_ENTITY_ID'].tolist()
                    ))
            
            # セグメント情報の読み込み
            if 'pivot_tables' in dataset:
                pivot_tables = dataset['pivot_tables']
                if 'Y_segment' in pivot_tables:
                    segment_df = pivot_tables['Y_segment']
                    if hasattr(segment_df, 'index'):
                        self.segment_info = dict(zip(
                            range(len(segment_df.index)), 
                            segment_df.index.tolist()
                        ))
            
        except Exception as e:
            logging.warning(f"名前マッピングの読み込みに失敗: {e}")
    
    def parse_stan_parameter(self, column_name: str) -> Dict[str, Any]:
        """
        Stanパラメータ名を解析
        
        Args:
            column_name: Stanパラメータ名（例: Item_ROIC.1.1, segment_private.2.3）
            
        Returns:
            解析結果の辞書
        """
        result = {
            'parameter_type': None,
            'entity_id': None,
            'segment_name': None,
            'product_id': None,
            'time_id': None,
            'is_consolidated': False,
            'is_segment': False,
            'is_product': False
        }
        
        # lp__（対数事後確率）の処理
        if column_name == 'lp__':
            result['parameter_type'] = 'log_posterior'
            return result
        
        # パラメータ名とインデックスを分離
        if '.' in column_name:
            param_name, indices = column_name.split('.', 1)
            indices = indices.split('.')
        else:
            param_name = column_name
            indices = []
        
        if not param_name:
            return result
        
        # パラメータタイプの判定
        parameter_mappings = {
            'Item_ROIC': ('product_roic', True, False, False),
            'segment_private': ('segment_private_effect', False, True, False),
            'consol_private': ('consol_private_effect', False, False, True),
            's_t': ('product_roic_std', True, False, False),
            'seg_sigma': ('segment_observation_error', False, True, False),
            'consol_sigma': ('consol_observation_error', False, False, True),
            's_segment_private': ('segment_private_std', False, True, False),
            's_consol_private': ('consol_private_std', False, False, True),
        }
        
        if param_name in parameter_mappings:
            param_type, is_product, is_segment, is_consolidated = parameter_mappings[param_name]
            result['parameter_type'] = param_type
            result['is_product'] = is_product
            result['is_segment'] = is_segment
            result['is_consolidated'] = is_consolidated
            
            # インデックスの処理
            if param_name == 'Item_ROIC' and len(indices) >= 2:
                result['product_id'] = int(indices[0])
                result['time_id'] = int(indices[1])
            elif param_name in ['segment_private', 'consol_private'] and len(indices) >= 2:
                result['entity_id'] = f"{'SEG' if is_segment else 'CONSOL'}_{indices[0]}"
                result['time_id'] = int(indices[1])
            elif param_name in ['s_t', 'seg_sigma', 'consol_sigma', 's_segment_private', 's_consol_private'] and len(indices) >= 1:
                if param_name == 's_t':
                    result['product_id'] = int(indices[0])
                else:
                    result['entity_id'] = f"{'SEG' if is_segment else 'CONSOL'}_{indices[0]}"
        
        elif param_name in ['nu_consol_roic', 'nu_seg_roic']:
            result['parameter_type'] = 'student_t_df'
            result['is_consolidated'] = 'consol' in param_name
            result['is_segment'] = 'seg' in param_name
        
        return result
    
    def generate_column_name(self, original_name: str, stat_type: str, index: int) -> str:
        """
        新しいカラム名を生成
        
        Args:
            original_name: 元のカラム名（Stanパラメータ名）
            stat_type: 統計タイプ（median, std）
            index: インデックス
            
        Returns:
            生成されたカラム名
        """
        parsed = self.parse_stan_parameter(original_name)
        
        if parsed['parameter_type'] is None:
            return f"{stat_type}_{original_name}"
        
        # パラメータタイプに応じてカラム名を生成
        name_generators = {
            'log_posterior': lambda: f"{stat_type}_log_posterior",
            'product_roic': lambda: self._generate_product_roic_name(parsed, stat_type),
            'product_roic_std': lambda: self._generate_product_roic_std_name(parsed, stat_type),
            'segment_private_effect': lambda: self._generate_segment_private_name(parsed, stat_type),
            'consol_private_effect': lambda: self._generate_consol_private_name(parsed, stat_type),
            'segment_observation_error': lambda: self._generate_segment_error_name(parsed, stat_type),
            'consol_observation_error': lambda: self._generate_consol_error_name(parsed, stat_type),
            'segment_private_std': lambda: self._generate_segment_std_name(parsed, stat_type),
            'consol_private_std': lambda: self._generate_consol_std_name(parsed, stat_type),
            'student_t_df': lambda: self._generate_student_t_name(parsed, stat_type),
        }
        
        generator = name_generators.get(parsed['parameter_type'])
        if generator:
            return generator()
        
        return f"{stat_type}_{index}"
    
    def _generate_product_roic_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """製品ROIC名を生成"""
        product_id = parsed.get('product_id', 'unknown')
        time_id = parsed.get('time_id', 'unknown')
        product_name = self._get_product_name(product_id)
        return f"{stat_type}_roic_{product_name}_t{time_id}"
    
    def _generate_product_roic_std_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """製品ROIC標準偏差名を生成"""
        product_id = parsed.get('product_id', 'unknown')
        product_name = self._get_product_name(product_id)
        return f"{stat_type}_roic_{product_name}_std"
    
    def _generate_segment_private_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """セグメントプライベート効果名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        time_id = parsed.get('time_id', 'unknown')
        segment_name = self._get_segment_name(entity_id)
        return f"{stat_type}_segment_{segment_name}_private_effect_t{time_id}"
    
    def _generate_consol_private_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """連結プライベート効果名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        time_id = parsed.get('time_id', 'unknown')
        company_name = self._get_company_name(entity_id)
        return f"{stat_type}_consol_{company_name}_private_effect_t{time_id}"
    
    def _generate_segment_error_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """セグメント観測誤差名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        segment_name = self._get_segment_name(entity_id)
        return f"{stat_type}_segment_{segment_name}_observation_error"
    
    def _generate_consol_error_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """連結観測誤差名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        company_name = self._get_company_name(entity_id)
        return f"{stat_type}_consol_{company_name}_observation_error"
    
    def _generate_segment_std_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """セグメントプライベート標準偏差名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        segment_name = self._get_segment_name(entity_id)
        return f"{stat_type}_segment_{segment_name}_private_std"
    
    def _generate_consol_std_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """連結プライベート標準偏差名を生成"""
        entity_id = parsed.get('entity_id', 'unknown')
        company_name = self._get_company_name(entity_id)
        return f"{stat_type}_consol_{company_name}_private_std"
    
    def _generate_student_t_name(self, parsed: Dict[str, Any], stat_type: str) -> str:
        """Student's t分布自由度名を生成"""
        if parsed['is_consolidated']:
            return f"{stat_type}_consol_student_t_df"
        else:
            return f"{stat_type}_segment_student_t_df"
    
    def _get_product_name(self, product_id: Any) -> str:
        """製品IDから製品名を取得"""
        try:
            product_id = int(product_id) - 1  # Stanのインデックスは1ベース
            if 0 <= product_id < len(self.product_names):
                return self.product_names[product_id]
        except (ValueError, IndexError):
            pass
        return f"product_{product_id}"
    
    def _get_company_name(self, entity_id: Any) -> str:
        """エンティティIDから企業名を取得"""
        try:
            if isinstance(entity_id, str) and entity_id.startswith("CONSOL_"):
                consol_id = int(entity_id.replace("CONSOL_", "")) - 1
                if 0 <= consol_id < len(self.entity_info):
                    return self.entity_info[consol_id]
        except (ValueError, IndexError):
            pass
        return f"company_{entity_id}"
    
    def _get_segment_name(self, entity_id: Any) -> str:
        """エンティティIDからセグメント名を取得"""
        try:
            if isinstance(entity_id, str) and entity_id.startswith("SEG_"):
                segment_id = int(entity_id.replace("SEG_", "")) - 1
                if 0 <= segment_id < len(self.segment_info):
                    return self.segment_info[segment_id]
        except (ValueError, IndexError):
            pass
        return f"segment_{entity_id}"


class CSVFileValidator:
    """CSVファイルの検証クラス"""
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> Path:
        """ファイルパスの検証"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"ファイルが見つかりません: {path}")
        
        if not path.suffix.lower() == '.csv':
            raise ValueError(f"CSVファイルではありません: {path}")
        
        if not path.is_file():
            raise ValueError(f"ファイルではありません: {path}")
        
        return path
    
    @staticmethod
    def validate_output_path(output_path: Union[str, Path]) -> Path:
        """出力パスの検証とディレクトリ作成"""
        path = Path(output_path)
        
        # 出力ディレクトリを作成
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return path


class CSVProcessor:
    """CSV処理と変分推論結果統合を管理するクラス（改善版）"""
    
    def __init__(
        self, 
        config: Optional[ProcessingConfig] = None, 
        config_manager: Optional[ConfigManager] = None,
        data_loader: Optional[DataLoader] = None
    ):
        """
        初期化
        
        Args:
            config: 処理設定。Noneの場合はConfigManagerから読み込み
            config_manager: 設定管理オブジェクト。Noneの場合は新規作成
            data_loader: データローダー（テスト用に注入可能）
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
        self.data_loader = data_loader or PickleDataLoader()
        
        # カラム名生成器を初期化
        self.column_name_generator = ColumnNameGenerator(
            product_level=self.config.product_level,
            dataset_file=self.config.dataset_file,
            data_loader=self.data_loader
        )
    
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
            with open(csv_path, 'r', encoding='utf-8') as f:
                total_rows = sum(
                    1 for line in f 
                    if not line.strip().startswith(self.config.comment_char)
                )
            
            self.logger.info(f"列数: {len(columns)}, 推定行数: {total_rows}")
            return columns, total_rows
            
        except Exception as e:
            self.logger.error(f"列情報の取得に失敗: {e}")
            raise
    
    def _process_chunk(self, chunk_data: Tuple[List[str], Path, ProcessingConfig, str, str]) -> pd.Series:
        """
        チャンクデータを処理して統計値を計算
        
        Args:
            chunk_data: (列名リスト, CSVパス, 設定, 製品レベル, データセットファイルパス)のタプル
            
        Returns:
            処理結果のSeries
        """
        columns, csv_path, config, product_level, dataset_file = chunk_data
        
        # カラム名生成器をローカルで作成
        column_name_generator = ColumnNameGenerator(
            product_level=product_level,
            dataset_file=dataset_file,
            data_loader=self.data_loader
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
            median_series = chunk_df.median(axis=0)
            std_series = chunk_df.std(axis=0)
            
            # 新しいカラム名を生成
            median_names = []
            std_names = []
            
            for col_name in chunk_df.columns:
                median_name = column_name_generator.generate_column_name(col_name, "median", 0)
                std_name = column_name_generator.generate_column_name(col_name, "std", 0)
                
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
    
    @contextmanager
    def _memory_monitor(self):
        """メモリ使用量の監視"""
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            yield
        finally:
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            self.logger.info(f"メモリ使用量: {memory_used:.2f} MB")
            
            if memory_used > self.config.memory_limit_mb:
                self.logger.warning(f"メモリ制限を超過: {memory_used:.2f} MB > {self.config.memory_limit_mb} MB")
                gc.collect()
    
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
        csv_path = CSVFileValidator.validate_file_path(csv_path)
        out_path = CSVFileValidator.validate_output_path(out_path)
        
        self.logger.info(f"入力ファイル: {csv_path}")
        self.logger.info(f"出力ファイル: {out_path}")
        
        with self._memory_monitor():
            # 列情報取得
            columns, total_rows = self._get_column_info(csv_path)
            
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
                        (chunk, csv_path, self.config, self.config.product_level, self.config.dataset_file)
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
            sample_names = list(final_result.index[:5])
            self.logger.info(f"生成されたカラム名の例: {sample_names}")
        
        return final_result


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(
        description="CSV処理ツール（改善版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 設定ファイルのCSVファイルパスを使用
  python csv_to_pkl_improved.py
  
  # コマンドライン引数でCSVファイルパスを指定
  python csv_to_pkl_improved.py --csv-file /path/to/input.csv
  
  # 出力パスも指定
  python csv_to_pkl_improved.py --csv-file /path/to/input.csv --output /path/to/output.pkl
  
  # L5プロダクトレベルで処理
  python csv_to_pkl_improved.py --csv-file /path/to/input.csv --product-level L5
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
    
    # プロセッサー初期化
    processor = CSVProcessor(config_manager=config_manager)
    
    # コマンドライン引数で設定を上書き
    if args.product_level:
        processor.config.product_level = args.product_level
        processor.column_name_generator.product_level = args.product_level
    
    # CSVファイルパスの決定
    csv_path = args.csv_file or config.csv_processing.input_file
    if csv_path is None:
        print("エラー: CSVファイルパスが指定されていません。")
        print("設定ファイル（gppm_config.yml）のcsv_processing.input_fileにパスを設定するか、")
        print("コマンドライン引数 --csv-file でパスを指定してください。")
        sys.exit(1)
    
    # 出力パスの決定
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
            sample_names = list(result.index[:3])
            print(f"生成されたカラム名の例: {sample_names}")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
