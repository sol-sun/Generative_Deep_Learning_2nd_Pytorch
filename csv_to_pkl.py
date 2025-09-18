"""
CSV処理ツール（DataFrame辞書出力版）
CmdStanPyの出力CSVファイルを効率的に処理し、StanパラメータタイプごとにDataFrameの辞書として出力するツール

主な機能:
- StanパラメータタイプごとのDataFrame構築
- 時系列データの整理（製品ROIC、プライベート効果など）
- 統計値（median, std）の分離
- 分析しやすい構造での出力

出力形式:
{
    'Item_ROIC': {
        'median': DataFrame(index=製品名, columns=時点),
        'std': DataFrame(index=製品名, columns=時点)
    },
    'segment_private_effect': {
        'median': DataFrame(index=セグメント名, columns=時点),
        'std': DataFrame(index=セグメント名, columns=時点)
    },
    'consol_private_effect': {
        'median': DataFrame(index=企業名, columns=時点),
        'std': DataFrame(index=企業名, columns=時点)
    },
    'observation_errors': {
        'segment': DataFrame(index=セグメント名, columns=['median', 'std']),
        'consol': DataFrame(index=企業名, columns=['median', 'std'])
    },
    'other_parameters': {
        'student_t_df': Series,
        'log_posterior': Series
    },
    'actual_roic': {
        'Y_segment': DataFrame(index=セグメント名, columns=時点),
        'Y_consol': DataFrame(index=企業名, columns=時点)
    },
    'predicted_roic': {
        'Y_segment': {
            'median': DataFrame(index=セグメント名, columns=時点),
            'std': DataFrame(index=セグメント名, columns=時点)
        },
        'Y_consol': {
            'median': DataFrame(index=企業名, columns=時点),
            'std': DataFrame(index=企業名, columns=時点)
        }
    }
}

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
    
    # CmdStanPyの出力CSVファイルを処理
    result = processor.process_csv_to_pkl(
        csv_path="global_ppm_roic_model-20250910175117.csv",
        out_path="output.pkl"
    )
    
    # 結果の使用例
    item_roic_median = result['Item_ROIC']['median']
    segment_effects = result['segment_private_effect']['median']
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
        self.time_mapping: Dict[int, str] = {}  # 時点ID -> 月次文字列のマッピング
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
                
                # 時点マッピング情報の読み込み
                self._load_time_mapping(pivot_tables)
            
        except Exception as e:
            logging.warning(f"名前マッピングの読み込みに失敗: {e}")
            # 時点マッピングは空のままにする
            self.time_mapping = {}
    
    def _load_time_mapping(self, pivot_tables: Dict[str, pd.DataFrame]) -> None:
        """時点マッピング情報を読み込み"""
        try:
            # 複数のソースから時点情報を取得を試行
            time_columns = None
            
            # 1. Y_segmentから時点情報を取得
            if 'Y_segment' in pivot_tables:
                segment_df = pivot_tables['Y_segment']
                if hasattr(segment_df, 'columns') and len(segment_df.columns) > 0:
                    time_columns = segment_df.columns.tolist()
                    logging.info(f"Y_segmentから時点情報を取得: {len(time_columns)}時点")
            
            # 2. Y_consolから時点情報を取得（Y_segmentで取得できない場合）
            if time_columns is None and 'Y_consol' in pivot_tables:
                consol_df = pivot_tables['Y_consol']
                if hasattr(consol_df, 'columns') and len(consol_df.columns) > 0:
                    time_columns = consol_df.columns.tolist()
                    logging.info(f"Y_consolから時点情報を取得: {len(time_columns)}時点")
            
            # 3. X2_segmentから時点情報を取得（MultiIndexの場合）
            if time_columns is None and 'X2_segment' in pivot_tables:
                x2_segment_df = pivot_tables['X2_segment']
                if hasattr(x2_segment_df, 'index') and isinstance(x2_segment_df.index, pd.MultiIndex):
                    # MultiIndexの2番目のレベル（時点）を取得
                    time_levels = x2_segment_df.index.get_level_values(1).unique().tolist()
                    if time_levels:
                        time_columns = time_levels
                        logging.info(f"X2_segmentから時点情報を取得: {len(time_columns)}時点")
            
            # 時点情報が見つかった場合、マッピングを作成
            if time_columns:
                self.time_mapping = self._create_time_mapping(time_columns)
                logging.info(f"時点マッピングを作成しました: {self.time_mapping}")
            else:
                # 時点マッピングは空のままにする
                self.time_mapping = {}
                logging.warning("時点情報が見つかりませんでした。時点マッピングは空です。")
                logging.warning("利用可能なピボットテーブル: " + ", ".join(pivot_tables.keys()))
                
        except Exception as e:
            logging.warning(f"時点マッピングの読み込みに失敗: {e}")
            # 時点マッピングは空のままにする
            self.time_mapping = {}
            logging.warning("時点マッピングは空です。")
    
    def _create_time_mapping(self, time_columns: List) -> Dict[int, str]:
        """時点IDから月次文字列へのマッピングを作成"""
        time_mapping = {}
        
        for i, time_value in enumerate(time_columns):
            try:
                # DatetimeIndexの場合（pandas.Timestamp）
                if hasattr(time_value, 'strftime'):
                    time_mapping[i] = time_value.strftime('%Y-%m')
                else:
                    # その他の場合は元の値をそのまま使用
                    time_mapping[i] = str(time_value)
            except Exception as e:
                logging.warning(f"時点値の変換に失敗: {time_value}, エラー: {e}")
                time_mapping[i] = str(time_value)
        
        return time_mapping
    
    def get_time_string(self, time_id: int) -> str:
        """時点IDから月次文字列を取得"""
        if not self.time_mapping:
            # 時点マッピングが空の場合は、t{time_id}形式を返す
            logging.warning(f"時点マッピングが空です。time_id={time_id}をt{time_id}として返します。")
            return f"t{time_id}"
        
        # Stanパラメータの時点IDは1ベース、時点マッピングは0ベースなので調整
        adjusted_time_id = time_id - 1
        
        result = self.time_mapping.get(adjusted_time_id, f"t{time_id}")
        if result == f"t{time_id}":
            logging.warning(f"時点マッピングにtime_id={time_id}(adjusted={adjusted_time_id})が見つかりません。利用可能なキー: {list(self.time_mapping.keys())}")
        return result
    
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


class DataFrameBuilder:
    """DataFrame構築クラス"""
    
    def __init__(self, column_name_generator: ColumnNameGenerator):
        """
        初期化
        
        Args:
            column_name_generator: カラム名生成器
        """
        self.column_name_generator = column_name_generator
        
        # データを格納する辞書
        self.data_storage = {
            'Item_ROIC': {'median': {}, 'std': {}},
            'segment_private_effect': {'median': {}, 'std': {}},
            'consol_private_effect': {'median': {}, 'std': {}},
            'observation_errors': {
                'segment': {'median': {}, 'std': {}},
                'consol': {'median': {}, 'std': {}}
            },
            'other_parameters': {
                'student_t_df': {'median': {}, 'std': {}},
                'log_posterior': {'median': {}, 'std': {}}
            },
            'actual_roic': {
                'Y_segment': None,
                'Y_consol': None
            },
            'predicted_roic': {
                'Y_segment': {'median': {}, 'std': {}},
                'Y_consol': {'median': {}, 'std': {}}
            }
        }
        
        # データセットファイルから実績ROICデータを読み込み
        self._load_actual_roic_data()
    
    def _load_actual_roic_data(self) -> None:
        """データセットファイルから実績ROICデータを読み込み"""
        try:
            if not self.column_name_generator.dataset_file or not Path(self.column_name_generator.dataset_file).exists():
                logging.warning("データセットファイルが存在しません。実績ROICデータは読み込まれません。")
                return
            
            # データセットファイルを読み込み
            with open(self.column_name_generator.dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            
            # ピボットテーブルから実績ROICデータを取得
            if 'pivot_tables' in dataset:
                pivot_tables = dataset['pivot_tables']
                
                # Y_segmentの実績ROICデータ
                if 'Y_segment' in pivot_tables:
                    self.data_storage['actual_roic']['Y_segment'] = pivot_tables['Y_segment'].copy()
                    logging.info(f"Y_segmentの実績ROICデータを読み込みました: {pivot_tables['Y_segment'].shape}")
                
                # Y_consolの実績ROICデータ
                if 'Y_consol' in pivot_tables:
                    self.data_storage['actual_roic']['Y_consol'] = pivot_tables['Y_consol'].copy()
                    logging.info(f"Y_consolの実績ROICデータを読み込みました: {pivot_tables['Y_consol'].shape}")
            
        except Exception as e:
            logging.warning(f"実績ROICデータの読み込みに失敗: {e}")
    
    def add_parameter_data(self, original_name: str, median_value: float, std_value: float) -> None:
        """
        パラメータデータを追加
        
        Args:
            original_name: 元のStanパラメータ名
            median_value: 中央値
            std_value: 標準偏差
        """
        parsed = self.column_name_generator.parse_stan_parameter(original_name)
        
        if parsed['parameter_type'] is None:
            return
        
        # パラメータタイプに応じてデータを格納
        if parsed['parameter_type'] == 'product_roic':
            self._add_product_roic_data(parsed, median_value, std_value)
        elif parsed['parameter_type'] == 'segment_private_effect':
            self._add_segment_private_data(parsed, median_value, std_value)
        elif parsed['parameter_type'] == 'consol_private_effect':
            self._add_consol_private_data(parsed, median_value, std_value)
        elif parsed['parameter_type'] in ['segment_observation_error', 'consol_observation_error']:
            self._add_observation_error_data(parsed, median_value, std_value)
        elif parsed['parameter_type'] == 'student_t_df':
            self._add_student_t_data(parsed, median_value, std_value)
        elif parsed['parameter_type'] == 'log_posterior':
            self._add_log_posterior_data(median_value, std_value)
    
    def _add_product_roic_data(self, parsed: Dict[str, Any], median_value: float, std_value: float) -> None:
        """製品ROICデータを追加"""
        product_id = parsed.get('product_id')
        time_id = parsed.get('time_id')
        
        if product_id is not None and time_id is not None:
            product_name = self.column_name_generator._get_product_name(product_id)
            time_key = self.column_name_generator.get_time_string(time_id)
            
            if product_name not in self.data_storage['Item_ROIC']['median']:
                self.data_storage['Item_ROIC']['median'][product_name] = {}
                self.data_storage['Item_ROIC']['std'][product_name] = {}
            
            self.data_storage['Item_ROIC']['median'][product_name][time_key] = median_value
            self.data_storage['Item_ROIC']['std'][product_name][time_key] = std_value
    
    def _add_segment_private_data(self, parsed: Dict[str, Any], median_value: float, std_value: float) -> None:
        """セグメントプライベート効果データを追加"""
        entity_id = parsed.get('entity_id')
        time_id = parsed.get('time_id')
        
        if entity_id is not None and time_id is not None:
            segment_name = self.column_name_generator._get_segment_name(entity_id)
            time_key = self.column_name_generator.get_time_string(time_id)
            
            if segment_name not in self.data_storage['segment_private_effect']['median']:
                self.data_storage['segment_private_effect']['median'][segment_name] = {}
                self.data_storage['segment_private_effect']['std'][segment_name] = {}
            
            self.data_storage['segment_private_effect']['median'][segment_name][time_key] = median_value
            self.data_storage['segment_private_effect']['std'][segment_name][time_key] = std_value
    
    def _add_consol_private_data(self, parsed: Dict[str, Any], median_value: float, std_value: float) -> None:
        """連結プライベート効果データを追加"""
        entity_id = parsed.get('entity_id')
        time_id = parsed.get('time_id')
        
        if entity_id is not None and time_id is not None:
            company_name = self.column_name_generator._get_company_name(entity_id)
            time_key = self.column_name_generator.get_time_string(time_id)
            
            if company_name not in self.data_storage['consol_private_effect']['median']:
                self.data_storage['consol_private_effect']['median'][company_name] = {}
                self.data_storage['consol_private_effect']['std'][company_name] = {}
            
            self.data_storage['consol_private_effect']['median'][company_name][time_key] = median_value
            self.data_storage['consol_private_effect']['std'][company_name][time_key] = std_value
    
    def _add_observation_error_data(self, parsed: Dict[str, Any], median_value: float, std_value: float) -> None:
        """観測誤差データを追加"""
        entity_id = parsed.get('entity_id')
        
        if entity_id is not None:
            if parsed['parameter_type'] == 'segment_observation_error':
                segment_name = self.column_name_generator._get_segment_name(entity_id)
                self.data_storage['observation_errors']['segment']['median'][segment_name] = median_value
                self.data_storage['observation_errors']['segment']['std'][segment_name] = std_value
            elif parsed['parameter_type'] == 'consol_observation_error':
                company_name = self.column_name_generator._get_company_name(entity_id)
                self.data_storage['observation_errors']['consol']['median'][company_name] = median_value
                self.data_storage['observation_errors']['consol']['std'][company_name] = std_value
    
    def _add_student_t_data(self, parsed: Dict[str, Any], median_value: float, std_value: float) -> None:
        """Student's t分布自由度データを追加"""
        key = 'consol' if parsed['is_consolidated'] else 'segment'
        self.data_storage['other_parameters']['student_t_df']['median'][key] = median_value
        self.data_storage['other_parameters']['student_t_df']['std'][key] = std_value
    
    def _add_log_posterior_data(self, median_value: float, std_value: float) -> None:
        """対数事後確率データを追加"""
        self.data_storage['other_parameters']['log_posterior']['median']['log_posterior'] = median_value
        self.data_storage['other_parameters']['log_posterior']['std']['log_posterior'] = std_value
    
    def build_dataframes(self) -> Dict[str, Any]:
        """
        構築されたデータからDataFrame辞書を作成
        
        Returns:
            パラメータタイプごとのDataFrame辞書
        """
        result = {}
        
        # Item_ROICのDataFrame構築
        if self.data_storage['Item_ROIC']['median']:
            result['Item_ROIC'] = {
                'median': self._build_time_series_dataframe(self.data_storage['Item_ROIC']['median']),
                'std': self._build_time_series_dataframe(self.data_storage['Item_ROIC']['std'])
            }
        
        # segment_private_effectのDataFrame構築
        if self.data_storage['segment_private_effect']['median']:
            result['segment_private_effect'] = {
                'median': self._build_time_series_dataframe(self.data_storage['segment_private_effect']['median']),
                'std': self._build_time_series_dataframe(self.data_storage['segment_private_effect']['std'])
            }
        
        # consol_private_effectのDataFrame構築
        if self.data_storage['consol_private_effect']['median']:
            result['consol_private_effect'] = {
                'median': self._build_time_series_dataframe(self.data_storage['consol_private_effect']['median']),
                'std': self._build_time_series_dataframe(self.data_storage['consol_private_effect']['std'])
            }
        
        # observation_errorsのDataFrame構築
        observation_errors = {}
        if self.data_storage['observation_errors']['segment']['median']:
            observation_errors['segment'] = self._build_observation_error_dataframe(
                self.data_storage['observation_errors']['segment']
            )
        if self.data_storage['observation_errors']['consol']['median']:
            observation_errors['consol'] = self._build_observation_error_dataframe(
                self.data_storage['observation_errors']['consol']
            )
        
        if observation_errors:
            result['observation_errors'] = observation_errors
        
        # other_parametersのSeries構築
        other_params = {}
        if self.data_storage['other_parameters']['student_t_df']['median']:
            other_params['student_t_df'] = pd.Series(self.data_storage['other_parameters']['student_t_df']['median'])
        if self.data_storage['other_parameters']['log_posterior']['median']:
            other_params['log_posterior'] = pd.Series(self.data_storage['other_parameters']['log_posterior']['median'])
        
        if other_params:
            result['other_parameters'] = other_params
        
        # 実績ROICデータの追加
        if self.data_storage['actual_roic']['Y_segment'] is not None:
            result['actual_roic'] = {
                'Y_segment': self.data_storage['actual_roic']['Y_segment']
            }
        if self.data_storage['actual_roic']['Y_consol'] is not None:
            if 'actual_roic' not in result:
                result['actual_roic'] = {}
            result['actual_roic']['Y_consol'] = self.data_storage['actual_roic']['Y_consol']
        
        # 推測ROICデータの計算と追加
        predicted_roic = self._calculate_predicted_roic()
        if predicted_roic:
            result['predicted_roic'] = predicted_roic
        
        return result
    
    def _build_time_series_dataframe(self, data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        時系列DataFrameを構築
        
        Args:
            data: {entity_name: {time_key: value}} の形式
            
        Returns:
            時系列DataFrame (index=企業名/セグメント名, columns=時点)
        """
        if not data:
            return pd.DataFrame()
        
        # 全ての時点を収集
        all_times = set()
        for entity_data in data.values():
            all_times.update(entity_data.keys())
        
        # 時点でソート（月次形式を考慮）
        def time_sort_key(time_str):
            if time_str.startswith('t'):
                # t1, t2形式の場合は数値でソート
                try:
                    return int(time_str[1:])
                except ValueError:
                    return 0
            elif '-' in time_str and len(time_str) == 7:
                # 2023-01形式の場合は年月でソート
                try:
                    year, month = time_str.split('-')
                    return int(year) * 12 + int(month)
                except ValueError:
                    return 0
            else:
                return 0
        
        sorted_times = sorted(all_times, key=time_sort_key)
        
        # DataFrameを構築（entity_nameをindex、time_keyをcolumnsに）
        df_data = {}
        for entity_name, entity_data in data.items():
            df_data[entity_name] = {}
            for time_key in sorted_times:
                df_data[entity_name][time_key] = entity_data.get(time_key, np.nan)
        
        return pd.DataFrame(df_data).T  # 転置してentity_nameをindexに
    
    def _build_observation_error_dataframe(self, data: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        観測誤差DataFrameを構築
        
        Args:
            data: {'median': {entity_name: value}, 'std': {entity_name: value}} の形式
            
        Returns:
            観測誤差DataFrame (index=企業名/セグメント名, columns=['median', 'std'])
        """
        if not data['median']:
            return pd.DataFrame()
        
        df_data = {}
        for entity_name in data['median'].keys():
            df_data[entity_name] = {
                'median': data['median'].get(entity_name, np.nan),
                'std': data['std'].get(entity_name, np.nan)
            }
        
        return pd.DataFrame(df_data).T  # 転置してentity_nameをindexに
    
    def _calculate_predicted_roic(self) -> Dict[str, Any]:
        """
        推測ROICを計算（Stanモデルの式に基づく）
        
        Returns:
            推測ROICデータの辞書
        """
        try:
            # データセットファイルからシェアデータを読み込み
            if not self.column_name_generator.dataset_file or not Path(self.column_name_generator.dataset_file).exists():
                logging.warning("データセットファイルが存在しません。推測ROICは計算されません。")
                return {}
            
            with open(self.column_name_generator.dataset_file, 'rb') as f:
                dataset = pickle.load(f)
            
            if 'pivot_tables' not in dataset:
                logging.warning("ピボットテーブルが見つかりません。推測ROICは計算されません。")
                return {}
            
            pivot_tables = dataset['pivot_tables']
            result = {}
            
            # セグメント推測ROICの計算
            if ('X2_segment' in pivot_tables and 
                self.data_storage['Item_ROIC']['median'] and 
                self.data_storage['segment_private_effect']['median']):
                
                segment_predicted = self._calculate_segment_predicted_roic(pivot_tables)
                if segment_predicted:
                    result['Y_segment'] = segment_predicted
            
            # 連結推測ROICの計算
            if ('X2_consol' in pivot_tables and 
                self.data_storage['Item_ROIC']['median'] and 
                self.data_storage['consol_private_effect']['median']):
                
                consol_predicted = self._calculate_consol_predicted_roic(pivot_tables)
                if consol_predicted:
                    result['Y_consol'] = consol_predicted
            
            return result
            
        except Exception as e:
            logging.warning(f"推測ROICの計算に失敗: {e}")
            return {}
    
    def _calculate_segment_predicted_roic(self, pivot_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """セグメント推測ROICを計算"""
        try:
            # 必要なデータを取得
            X2_segment = pivot_tables['X2_segment']  # セグメント×製品シェア
            item_roic_median = self._build_time_series_dataframe(self.data_storage['Item_ROIC']['median'])
            segment_private_median = self._build_time_series_dataframe(self.data_storage['segment_private_effect']['median'])
            
            if item_roic_std := self._build_time_series_dataframe(self.data_storage['Item_ROIC']['std']):
                item_roic_std = item_roic_std
            else:
                item_roic_std = None
            
            if segment_private_std := self._build_time_series_dataframe(self.data_storage['segment_private_effect']['std']):
                segment_private_std = segment_private_std
            else:
                segment_private_std = None
            
            # 推測ROICを計算: mu = Share * Item_ROIC + segment_private_eff
            predicted_median = self._calculate_weighted_roic(X2_segment, item_roic_median, segment_private_median)
            predicted_std = None
            
            if item_roic_std is not None and segment_private_std is not None:
                predicted_std = self._calculate_weighted_roic(X2_segment, item_roic_std, segment_private_std)
            
            result = {'median': predicted_median}
            if predicted_std is not None:
                result['std'] = predicted_std
            
            return result
            
        except Exception as e:
            logging.warning(f"セグメント推測ROICの計算に失敗: {e}")
            return {}
    
    def _calculate_consol_predicted_roic(self, pivot_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """連結推測ROICを計算"""
        try:
            # 必要なデータを取得
            X2_consol = pivot_tables['X2_consol']  # 連結×製品シェア
            item_roic_median = self._build_time_series_dataframe(self.data_storage['Item_ROIC']['median'])
            consol_private_median = self._build_time_series_dataframe(self.data_storage['consol_private_effect']['median'])
            
            if item_roic_std := self._build_time_series_dataframe(self.data_storage['Item_ROIC']['std']):
                item_roic_std = item_roic_std
            else:
                item_roic_std = None
            
            if consol_private_std := self._build_time_series_dataframe(self.data_storage['consol_private_effect']['std']):
                consol_private_std = consol_private_std
            else:
                consol_private_std = None
            
            # 推測ROICを計算: mu_consol = Share_consol * Item_ROIC + consol_private_eff
            predicted_median = self._calculate_weighted_roic(X2_consol, item_roic_median, consol_private_median)
            predicted_std = None
            
            if item_roic_std is not None and consol_private_std is not None:
                predicted_std = self._calculate_weighted_roic(X2_consol, item_roic_std, consol_private_std)
            
            result = {'median': predicted_median}
            if predicted_std is not None:
                result['std'] = predicted_std
            
            return result
            
        except Exception as e:
            logging.warning(f"連結推測ROICの計算に失敗: {e}")
            return {}
    
    def _calculate_weighted_roic(self, share_data: pd.DataFrame, item_roic: pd.DataFrame, private_effect: pd.DataFrame) -> pd.DataFrame:
        """
        加重ROICを計算: Share * Item_ROIC + private_effect
        
        Args:
            share_data: シェアデータ（MultiIndex: entity, time）
            item_roic: 製品ROICデータ（index: product, columns: time）
            private_effect: プライベート効果データ（index: entity, columns: time）
            
        Returns:
            推測ROICデータ（index: entity, columns: time）
        """
        try:
            # 時点ごとに計算
            result_data = {}
            
            for time_col in item_roic.columns:
                if time_col in share_data.index.get_level_values(1):
                    # 該当時点のシェアデータを取得
                    time_share = share_data.xs(time_col, level=1)
                    
                    # 該当時点の製品ROICを取得
                    time_item_roic = item_roic[time_col]
                    
                    # 該当時点のプライベート効果を取得
                    time_private = private_effect[time_col] if time_col in private_effect.columns else pd.Series(0, index=private_effect.index)
                    
                    # 加重ROICを計算: Share * Item_ROIC + private_effect
                    weighted_roic = time_share.dot(time_item_roic) + time_private
                    
                    result_data[time_col] = weighted_roic
            
            return pd.DataFrame(result_data)
            
        except Exception as e:
            logging.warning(f"加重ROICの計算に失敗: {e}")
            return pd.DataFrame()


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
    """CSV処理と変分推論結果統合を管理するクラス（DataFrame辞書出力版）"""
    
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
    
    def _process_chunk(self, chunk_data: Tuple[List[str], Path, ProcessingConfig, str, str]) -> Dict[str, Tuple[float, float]]:
        """
        チャンクデータを処理して統計値を計算
        
        Args:
            chunk_data: (列名リスト, CSVパス, 設定, 製品レベル, データセットファイルパス)のタプル
            
        Returns:
            処理結果の辞書 {original_name: (median_value, std_value)}
        """
        columns, csv_path, config, product_level, dataset_file = chunk_data
        
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
                return {}
            
            # 統計値を計算
            median_series = chunk_df.median(axis=0)
            std_series = chunk_df.std(axis=0)
            
            # 結果を辞書形式で返す
            result = {}
            for col_name in chunk_df.columns:
                result[col_name] = (median_series[col_name], std_series[col_name])
            
            return result
            
        except Exception as e:
            self.logger.error(f"チャンク処理エラー: {e}")
            return {}
    
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
    ) -> Dict[str, Any]:
        """
        CSVファイルをPKLファイルに変換（DataFrame辞書出力）
        
        Args:
            csv_path: 入力CSVファイルのパス
            out_path: 出力PKLファイルのパス
            num_chunks: 処理チャンク数（Noneの場合は自動決定）
            show_progress: 進捗表示の有無
            
        Returns:
            パラメータタイプごとのDataFrame辞書
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
            
            # DataFrameBuilderを初期化
            dataframe_builder = DataFrameBuilder(self.column_name_generator)
            
            # 並列処理
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
                        chunk_result = future.result()
                        # DataFrameBuilderにデータを追加
                        for original_name, (median_value, std_value) in chunk_result.items():
                            dataframe_builder.add_parameter_data(original_name, median_value, std_value)
                    except Exception as e:
                        chunk_idx = future_to_chunk[future]
                        self.logger.error(f"チャンク {chunk_idx} の処理に失敗: {e}")
            
            # DataFrame辞書を構築
            final_result = dataframe_builder.build_dataframes()
            
            # PKLファイルに保存
            with open(out_path, 'wb') as f:
                pickle.dump(final_result, f)
        
        # 処理時間と結果情報をログ出力
        processing_time = time.time() - start_time
        self.logger.info(f"処理完了: {processing_time:.2f}秒")
        self.logger.info(f"出力ファイル: {out_path}")
        
        # 結果の構造をログ出力
        self.logger.info(f"結果の構造:")
        for key, value in final_result.items():
            if isinstance(value, dict):
                self.logger.info(f"- {key}: {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        self.logger.info(f"  - {sub_key}: {sub_value.shape}")
                    else:
                        self.logger.info(f"  - {sub_key}: {type(sub_value)}")
            else:
                self.logger.info(f"- {key}: {type(value)}")
        
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
        print(f"出力ファイル: {out_path}")
        print(f"カラム名生成設定:")
        print(f"- 製品レベル: {processor.config.product_level}")
        
        # 結果の構造を表示
        print("\n=== 結果の構造 ===")
        for key, value in result.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    if hasattr(sub_value, 'shape'):
                        print(f"  - {sub_key}: {sub_value.shape}")
                    else:
                        print(f"  - {sub_key}: {type(sub_value)}")
            else:
                print(f"{key}: {type(value)}")
        
        # データの例を表示
        if 'Item_ROIC' in result and 'median' in result['Item_ROIC']:
            print("\n=== Item_ROIC (median) の例 ===")
            print(result['Item_ROIC']['median'].head())
        
        if 'segment_private_effect' in result and 'median' in result['segment_private_effect']:
            print("\n=== segment_private_effect (median) の例 ===")
            print(result['segment_private_effect']['median'].head())
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
