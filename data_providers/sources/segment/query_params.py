"""
Segment プロバイダー用クエリパラメーター
====================================

セグメントデータ取得用のクエリパラメーター定義。

使用例:
    from data_providers.classification.segment.query_params import SegmentQueryParams
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class SegmentQueryParams(BaseModel):
    """セグメントクエリパラメータ"""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
    
    # フィルタ
    fsym_ids: Optional[List[str]] = Field(
        default=None,
        description="対象のFSYM IDリスト"
    )
    min_sales_ratio: Optional[float] = Field(
        default=None,
        description="最小売上比率"
    )
