"""
Revere プロバイダー用型定義
=========================

FactSet REVEREデータの基本的な型定義。

使用例:
    from data_providers.data_sources.revere.types import RevereRecord
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RevereRecord(BaseModel):
    """REVEREデータレコード"""
    
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
    revenue_share: Optional[float] = Field(
        default=None,
        description="売上シェア"
    )
    fiscal_year: Optional[int] = Field(
        default=None,
        description="会計年度"
    )
