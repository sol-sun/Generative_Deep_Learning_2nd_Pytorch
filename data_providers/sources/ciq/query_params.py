"""
CIQ プロバイダー用クエリパラメーター
=================================

CIQ固有のクエリパラメーター定義。
複雑なCIQ検索条件の型安全な定義とバリデーション・変換ロジック。

使用例:
    from data_providers.sources.ciq.query_params import CIQQueryParams
"""

from __future__ import annotations

from typing import List, Optional, Union, Dict, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    field_serializer,
)
from pydantic.types import conint

from wolf_period import WolfPeriod, WolfPeriodRange
from gppm.utils.country_code_manager import (
    convert_to_alpha3,
    Alpha2Code,
    Alpha3Code
)


class CIQQueryParams(BaseModel):
    """CIQ クエリパラメータ。

    目的:
    - 取得対象（企業/地域）を安全に指定
    - データベースアクセスの安全性を確保
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        extra="forbid",
    )
    
    # 企業フィルタ
    company_id: Optional[conint(ge=1, le=9999999999)] = Field(
        default=None,
        description="特定のCompany IDでフィルタ (10桁)",
        examples=[1234567890]
    )
    company_ids: Optional[List[conint(ge=1, le=9999999999)]] = Field(
        default=None,
        description="特定のCompany IDリストでフィルタ (10桁)",
        examples=[[1234567890, 9876543210]]
    )
    
    # 上場フィルタ
    listed_only: bool = Field(
        default=True,
        description="上場企業のみを取得（UN_LISTED_FLG = 0）"
    )
    
    # 地域フィルタ（パフォーマンス最適化）
    country_codes: Optional[List[Union[Alpha2Code, Alpha3Code]]] = Field(
        default=None,
        description="国コードリスト（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-3に変換）",
        examples=[["US", "USA", "JP", "JPN"]]
    )
    
    # WolfPeriodベースの期間フィルタ（将来実装用）
    period_range: Optional[WolfPeriodRange] = Field(
        default=None,
        description="期間範囲（WolfPeriodRange）（将来実装予定）",
        examples=["2020M1:2023M12"]
    )
    period_start: Optional[WolfPeriod] = Field(
        default=None,
        description="開始期間（WolfPeriod）（将来実装予定）",
        examples=["2020M1"]
    )
    period_end: Optional[WolfPeriod] = Field(
        default=None,
        description="終了期間（WolfPeriod）（将来実装予定）",
        examples=["2023M12"]
    )
    
    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    
    @field_validator("country_codes", mode="before")
    @classmethod
    def _normalize_country_codes_list(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """国コードリストをalpha-3に正規化。"""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("country_codesはリストである必要があります")
        
        normalized_codes = []
        for code in v:
            if isinstance(code, str):
                normalized = convert_to_alpha3(code.strip())
                if normalized:
                    normalized_codes.append(normalized)
        
        return normalized_codes if normalized_codes else None
    
    @field_serializer("period_range", "period_start", "period_end")
    def _serialize_periods(self, period: Optional[Union[WolfPeriod, WolfPeriodRange]]) -> Optional[Dict[str, Any]]:
        """WolfPeriod/WolfPeriodRangeをシリアライズ。"""
        return period.model_dump() if period else None

