"""
Segment プロバイダー用型定義
==========================

セグメントデータの基本的な型定義。

使用例:
    from data_providers.classification.segment.types import SegmentRecord
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SegmentRecord(BaseModel):
    """セグメントデータレコード"""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )
    
    # 基本情報
    fsym_id: Optional[str] = Field(
        default=None,
        description="FactSet Symbol ID"
    )
    segment_name: Optional[str] = Field(
        default=None,
        description="セグメント名"
    )
    sales_amount: Optional[float] = Field(
        default=None,
        description="売上高"
    )
    sales_ratio: Optional[float] = Field(
        default=None,
        description="売上比率"
    )
