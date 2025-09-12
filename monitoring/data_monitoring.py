#!/usr/bin/env python
"""
データモニタリング用ユーティリティ

データ処理パイプラインでの企業数変化、国別分布、その他の統計情報を
with文を使ったコンテキストマネージャーでモニタリングする。
"""

import time
import json
import logging
from typing import Dict, Any, Optional, Union, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .monitoring_config import MonitoringConfig
from contextlib import contextmanager
from dataclasses import dataclass, asdict
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import queue


@dataclass
class MonitoringStats:
    """モニタリング統計情報"""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    start_count: Optional[int] = None
    end_count: Optional[int] = None
    count_change: Optional[int] = None
    change_reason: str = ""
    country_distribution: Optional[Dict[str, int]] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    memory_usage_mb: Optional[float] = None
    performance_impact_ms: Optional[float] = None


class DataMonitoringContext:
    """
    データモニタリング用コンテキストマネージャー
    
    with文を使用してデータ処理の各段階をモニタリングし、
    企業数の変化、国別分布、処理時間などを記録する。
    """
    
    def __init__(self, 
                 name: str, 
                 logger: logging.Logger,
                 log_file_path: Optional[str] = None,
                 enable_performance_monitoring: bool = True,
                 enable_memory_monitoring: bool = False):
        """
        Args:
            name: モニタリング対象の処理名
            logger: ログ出力用logger
            log_file_path: 詳細ログの出力先ファイルパス
            enable_performance_monitoring: パフォーマンスモニタリングの有効化
            enable_memory_monitoring: メモリモニタリングの有効化
        """
        self.name = name
        self.logger = logger
        self.enable_performance_monitoring = enable_performance_monitoring
        self.enable_memory_monitoring = enable_memory_monitoring
        self.log_file_path = log_file_path
        
        self.stats = MonitoringStats(
            name=name,
            start_time=time.time(),
            metadata={}
        )
        
        self._start_count = None
        self._start_data = None
        self._log_queue = queue.Queue()
        self._executor = ThreadPoolExecutor(max_workers=1) if log_file_path else None
        
    def __enter__(self):
        """コンテキスト開始"""
        self.logger.info(f"=== {self.name} 処理開始 ===")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了"""
        self.stats.end_time = time.time()
        self.stats.duration = self.stats.end_time - self.stats.start_time
        
        # パフォーマンス影響の測定
        if self.enable_performance_monitoring:
            self.stats.performance_impact_ms = self.stats.duration * 1000
        
        if exc_type is not None:
            self.stats.error = f"{exc_type.__name__}: {exc_val}"
            self.logger.error(f"=== {self.name} 処理エラー ===")
            self.logger.error(f"エラー: {self.stats.error}")
        else:
            self.logger.info(f"=== {self.name} 処理完了 ===")
            
        # 統計情報の出力
        self._log_summary()
        
        # 詳細ログファイルへの出力（非同期）
        if self.log_file_path and self._executor:
            self._executor.submit(self._save_detailed_log)
            self._executor.shutdown(wait=False)
            
        return False  # 例外を再発生させない
    
    def set_initial_data(self, data: Union[pd.DataFrame, Dict[str, Any]], 
                        entity_id_col: Optional[str] = None,
                        track_columns: Optional[List[str]] = None):
        """
        初期データを設定（メモリ効率を考慮）
        
        Args:
            data: 初期データ（DataFrameまたは辞書）
            entity_id_col: エンティティIDの列名（Noneの場合はエラー）
            track_columns: 追跡対象の列名リスト（分布分析用）
        """
        if isinstance(data, pd.DataFrame):
            self._start_count = len(data)
            
            # エンティティID列の存在確認
            if entity_id_col is None:
                raise ValueError("entity_id_col must be specified for DataFrame monitoring")
            
            if entity_id_col not in data.columns:
                raise ValueError(f"Entity ID column '{entity_id_col}' not found in DataFrame columns: {list(data.columns)}")
            
            # メモリ効率のため、必要最小限のデータのみコピー
            if self.enable_memory_monitoring:
                copy_columns = [entity_id_col]
                if track_columns:
                    copy_columns.extend([col for col in track_columns if col in data.columns])
                self._start_data = data[copy_columns].copy() if copy_columns else None
            else:
                self._start_data = None
            
            # 指定された列の分布を計算
            if track_columns:
                for col in track_columns:
                    if col in data.columns:
                        self.stats.metadata[f"{col}_initial_distribution"] = data[col].value_counts().to_dict()
                
        elif isinstance(data, dict):
            # 辞書形式の場合、各キーの要素数を記録（データはコピーしない）
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    self.stats.metadata[f"{key}_initial_count"] = len(value)
                elif isinstance(value, (list, tuple)):
                    self.stats.metadata[f"{key}_initial_count"] = len(value)
                    
        self.stats.start_count = self._start_count
        self.logger.info(f"初期データ設定: {self._start_count} 件")
        
    def log_count_change(self, 
                        after_data: Union[pd.DataFrame, int],
                        reason: str = "",
                        entity_id_col: Optional[str] = None,
                        track_columns: Optional[List[str]] = None):
        """
        カウント変化を記録
        
        Args:
            after_data: 処理後のデータ（DataFrameまたは数値）
            reason: 変化の理由
            entity_id_col: エンティティIDの列名（DataFrameの場合は必須）
            track_columns: 追跡対象の列名リスト（分布分析用）
        """
        if isinstance(after_data, pd.DataFrame):
            after_count = len(after_data)
            
            # エンティティID列の存在確認
            if entity_id_col is None:
                raise ValueError("entity_id_col must be specified for DataFrame monitoring")
            
            if entity_id_col not in after_data.columns:
                raise ValueError(f"Entity ID column '{entity_id_col}' not found in DataFrame columns: {list(after_data.columns)}")
            
            self._log_column_distributions(after_data, track_columns, "処理後")
        else:
            after_count = after_data
            
        self.stats.end_count = after_count
        self.stats.count_change = after_count - (self.stats.start_count or 0)
        self.stats.change_reason = reason
        
        change_pct = (self.stats.count_change / (self.stats.start_count or 1)) * 100
        
        self.logger.info(f"カウント変化: {self.stats.start_count} → {after_count} "
                        f"({self.stats.count_change:+d}, {change_pct:+.1f}%)")
        if reason:
            self.logger.info(f"変化理由: {reason}")
            
    def log_column_distribution(self, 
                               data: pd.DataFrame,
                               column_name: str,
                               label: str = "",
                               top_n: int = 10):
        """
        指定された列の分布を記録
        
        Args:
            data: データフレーム
            column_name: 列名
            label: ラベル（処理前、処理後等）
            top_n: 表示する上位N件
        """
        if column_name not in data.columns:
            self.logger.warning(f"列 '{column_name}' が見つかりません")
            return
            
        column_dist = data[column_name].value_counts()
        self.logger.info(f"{label}{column_name}分布:")
        for value, count in column_dist.head(top_n).items():
            self.logger.info(f"  {value}: {count} 件")
            
        if len(column_dist) > top_n:
            self.logger.info(f"  ... 他 {len(column_dist) - top_n} 件")
            
    def _log_column_distributions(self, data: pd.DataFrame, track_columns: Optional[List[str]], label: str):
        """内部用：複数列の分布ログ出力"""
        if track_columns:
            for col in track_columns:
                if col in data.columns:
                    self.log_column_distribution(data, col, label)
            
    def _log_summary(self):
        """処理サマリーのログ出力"""
        self.logger.info(f"処理時間: {self.stats.duration:.2f}秒")
        
        if self.enable_performance_monitoring and self.stats.performance_impact_ms:
            self.logger.info(f"パフォーマンス影響: {self.stats.performance_impact_ms:.1f}ms")
        
        if self.stats.start_count is not None and self.stats.end_count is not None:
            change_pct = (self.stats.count_change / self.stats.start_count) * 100
            self.logger.info(f"データ数変化: {self.stats.start_count} → {self.stats.end_count} "
                           f"({self.stats.count_change:+d}, {change_pct:+.1f}%)")
                           
        if self.stats.metadata:
            self.logger.info("メタデータ:")
            for key, value in self.stats.metadata.items():
                self.logger.info(f"  {key}: {value}")
                
    def _save_detailed_log(self):
        """詳細ログのファイル保存（エラーハンドリング強化）"""
        try:
            log_data = asdict(self.stats)
            log_data['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.stats.start_time))
            
            log_file = Path(self.log_file_path)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ファイルサイズチェック
            if log_file.exists() and log_file.stat().st_size > 100 * 1024 * 1024:  # 100MB
                # ログローテーション
                backup_file = log_file.with_suffix('.bak')
                if backup_file.exists():
                    backup_file.unlink()
                log_file.rename(backup_file)
            
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, ensure_ascii=False, indent=2) + '\n')
                
        except PermissionError as e:
            self.logger.error(f"ログファイルへの書き込み権限がありません: {e}")
        except OSError as e:
            self.logger.error(f"ログファイルの保存中にOSエラーが発生: {e}")
        except json.JSONEncodeError as e:
            self.logger.error(f"ログデータのJSONエンコードに失敗: {e}")
        except Exception as e:
            self.logger.error(f"詳細ログの保存に予期しないエラーが発生: {e}")
            self.logger.debug(f"エラー詳細: {type(e).__name__}: {e}", exc_info=True)


@contextmanager
def monitor_data_processing(name: str,
                          logger: logging.Logger,
                          log_file_path: Optional[str] = None,
                          enable_performance_monitoring: bool = True,
                          enable_memory_monitoring: bool = False):
    """
    データ処理モニタリング用のコンテキストマネージャー
    
    Args:
        name: 処理名
        logger: ログ出力用logger
        log_file_path: 詳細ログの出力先ファイルパス
        enable_performance_monitoring: パフォーマンスモニタリングの有効化
        enable_memory_monitoring: メモリモニタリングの有効化
        
    Yields:
        DataMonitoringContext: モニタリングコンテキスト
    """
    context = DataMonitoringContext(
        name=name,
        logger=logger,
        log_file_path=log_file_path,
        enable_performance_monitoring=enable_performance_monitoring,
        enable_memory_monitoring=enable_memory_monitoring
    )
    
    with context:
        yield context


@contextmanager
def monitor(name: str, 
           config: Optional['MonitoringConfig'] = None,
           logger: Optional[logging.Logger] = None):
    """
    シンプルなモニタリング用コンテキストマネージャー
    
    最も簡単な使用方法：
    ```python
    with monitor("データ処理") as m:
        m.set_initial_data(data, entity_id_col="entity_id")
        # 処理の実行
        processed_data = process_data(data)
        m.log_count_change(processed_data, "処理完了", entity_id_col="entity_id")
    ```
    
    Args:
        name: 処理名
        config: モニタリング設定（Noneの場合はデフォルト設定）
        logger: ロガー（Noneの場合は自動取得）
        
    Yields:
        DataMonitoringContext: モニタリングコンテキスト
    """
    # 設定の取得
    if config is None:
        from .monitoring_config import MonitoringConfig
        config = MonitoringConfig()
    
    # ロガーの取得
    if logger is None:
        # 標準ロガーを使用
        logger = logging.getLogger("data_monitoring")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    # コンテキストマネージャーを作成
    context = DataMonitoringContext(
        name=name,
        logger=logger,
        log_file_path=config.log_file_path,
        enable_performance_monitoring=config.enable_performance_monitoring,
        enable_memory_monitoring=config.enable_memory_monitoring
    )
    
    with context:
        yield context


