"""
Revere プロバイダー用クエリパラメーター
===================================

REVEREデータ取得用のクエリパラメーター定義。

使用例:
    from data_providers.data_sources.revere.query_params import RevereQueryParams
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict


class RevereQueryParams(BaseModel):
    """REVEREクエリパラメータ"""
    
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
    fiscal_year: Optional[int] = Field(
        default=None,
        description="対象会計年度"
    )
