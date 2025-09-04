"""
データプロバイダー共通バリデーション機能
=====================================

データプロバイダー間で共通して使用されるバリデーション機能を提供します。

主な機能:
- データフレームバリデーション
- 共通フィールドバリデーション
- データ品質チェック

使用例:
    from data_providers.utils.data_validation import validate_dataframe, check_data_quality
    
    # データフレームバリデーション
    is_valid = validate_dataframe(df, required_columns=['id', 'name'])
    
    # データ品質チェック
    quality_report = check_data_quality(df)
"""

from typing import List, Dict, Optional, Any
import pandas as pd
from gppm.utils.config_manager import get_logger

logger = get_logger(__name__)


def validate_dataframe(df: pd.DataFrame, required_columns: Optional[List[str]] = None) -> bool:
    """
    データフレームの基本バリデーション。
    
    Args:
        df: 検証対象のDataFrame
        required_columns: 必須列のリスト
        
    Returns:
        バリデーション結果（True: 有効、False: 無効）
    """
    if df is None or df.empty:
        logger.warning("データフレームが空またはNoneです")
        return False
    
    if required_columns:
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            logger.warning(f"必須列が不足しています: {missing_columns}")
            return False
    
    return True


def check_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    データ品質チェック。
    
    Args:
        df: チェック対象のDataFrame
        
    Returns:
        品質レポート辞書
    """
    if df.empty:
        return {"status": "empty", "issues": ["データフレームが空"]}
    
    report = {
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "null_counts": df.isnull().sum().to_dict(),
        "duplicate_rows": df.duplicated().sum(),
        "status": "ok",
        "issues": []
    }
    
    # データ品質問題の検出
    if report["duplicate_rows"] > 0:
        report["issues"].append(f"重複行: {report['duplicate_rows']}件")
    
    # 欠損値の多い列を検出
    high_null_columns = []
    for col, null_count in report["null_counts"].items():
        null_ratio = null_count / report["total_rows"]
        if null_ratio > 0.5:  # 50%以上欠損
            high_null_columns.append(f"{col} ({null_ratio:.1%})")
    
    if high_null_columns:
        report["issues"].append(f"高欠損率列: {', '.join(high_null_columns)}")
    
    # ステータス判定
    if report["issues"]:
        report["status"] = "warning"
    
    return report