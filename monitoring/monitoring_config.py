#!/usr/bin/env python
"""
モニタリング設定管理モジュール

データモニタリングの設定を管理し、動的な設定変更を可能にする。
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import yaml
from pathlib import Path


class MonitoringLevel(Enum):
    """モニタリングレベル"""
    MINIMAL = "minimal"      # 最小限のログ
    STANDARD = "standard"    # 標準的なログ
    DETAILED = "detailed"    # 詳細なログ
    DEBUG = "debug"         # デバッグ用の全ログ


@dataclass
class MonitoringConfig:
    """モニタリング設定"""
    # 基本設定
    level: MonitoringLevel = MonitoringLevel.STANDARD
    enable_performance_monitoring: bool = True
    enable_memory_monitoring: bool = False
    
    # ログ設定
    log_file_path: Optional[str] = None
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_level: str = "INFO"
    
    # パフォーマンス設定
    max_log_file_size_mb: int = 100
    async_logging: bool = True
    batch_log_size: int = 100
    
    # デフォルト設定
    default_track_columns: List[str] = field(default_factory=list)
    enable_entity_tracking: bool = False
    
    # 分布モニタリング設定
    distribution_monitoring: Dict[str, Any] = field(default_factory=lambda: {
        "top_values": 10,
        "track_changes": True
    })
    
    # カスタムメトリクス
    custom_metrics: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'MonitoringConfig':
        """YAMLファイルから設定を読み込み"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # レベルをEnumに変換
        if 'level' in config_data:
            config_data['level'] = MonitoringLevel(config_data['level'])
        
        return cls(**config_data)
    
    def to_yaml(self, config_path: str) -> None:
        """設定をYAMLファイルに保存"""
        config_data = {
            'level': self.level.value,
            'enable_performance_monitoring': self.enable_performance_monitoring,
            'enable_memory_monitoring': self.enable_memory_monitoring,
            'log_file_path': self.log_file_path,
            'log_format': self.log_format,
            'log_level': self.log_level,
            'max_log_file_size_mb': self.max_log_file_size_mb,
            'async_logging': self.async_logging,
            'batch_log_size': self.batch_log_size,
            'default_track_columns': self.default_track_columns,
            'enable_entity_tracking': self.enable_entity_tracking,
            'distribution_monitoring': self.distribution_monitoring,
            'custom_metrics': self.custom_metrics
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def should_track_distributions(self) -> bool:
        """分布モニタリングが必要かどうかを判定"""
        return len(self.default_track_columns) > 0
    
    def should_track_entities(self) -> bool:
        """エンティティ追跡が必要かどうかを判定"""
        return self.enable_entity_tracking
    
    def get_track_columns(self) -> List[str]:
        """追跡対象の列を取得"""
        return self.default_track_columns
    
    def is_debug_level(self) -> bool:
        """デバッグレベルかどうかを判定"""
        return self.level == MonitoringLevel.DEBUG
    
    def is_detailed_level(self) -> bool:
        """詳細レベル以上かどうかを判定"""
        return self.level in [MonitoringLevel.DETAILED, MonitoringLevel.DEBUG]


# デフォルト設定
DEFAULT_CONFIG = MonitoringConfig()

# 設定インスタンス（グローバル）
_global_config: Optional[MonitoringConfig] = None


def get_global_config() -> MonitoringConfig:
    """グローバル設定を取得"""
    global _global_config
    if _global_config is None:
        _global_config = DEFAULT_CONFIG
    return _global_config


def set_global_config(config: MonitoringConfig) -> None:
    """グローバル設定を設定"""
    global _global_config
    _global_config = config


def load_config_from_file(config_path: str) -> MonitoringConfig:
    """ファイルから設定を読み込み、グローバル設定として設定"""
    config = MonitoringConfig.from_yaml(config_path)
    set_global_config(config)
    return config


