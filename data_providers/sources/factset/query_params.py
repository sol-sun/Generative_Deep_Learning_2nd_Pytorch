"""
FactSet プロバイダー用クエリパラメーター
====================================

FactSet固有のクエリパラメーター定義。
複雑なFactSet検索条件の型安全な定義とバリデーション・変換ロジック。

使用例:
    from data_providers.sources.factset.query_params import FactSetQueryParams
    from wolf_period import WolfPeriod, WolfPeriodRange

    # クエリパラメータの作成
    params = FactSetQueryParams(
        country_codes=["US", "JP"],
        active_only=True,
        period_range=WolfPeriodRange.from_periods(
            WolfPeriod.from_month(2023, 1),
            WolfPeriod.from_month(2023, 12)
        ),
        batch_size=1000
    )
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
from pydantic.types import conint, constr

from wolf_period import WolfPeriod, WolfPeriodRange
from gppm.utils.country_code_manager import convert_to_alpha2, Alpha2Code, Alpha3Code


class FactSetQueryParams(BaseModel):
    """FactSet クエリパラメータ（WolfPeriod対応）。

    目的:
    - 取得対象（企業/証券/地域）と期間を安全に指定
    - データベースアクセスの安全性と性能を確保
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
    entity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="特定のEntity IDでフィルタ",
        examples=["001C7F-E"]
    )
    fsym_ids: Optional[List[constr(min_length=1, max_length=20)]] = Field(
        default=None,
        description="特定のFSYM IDリストでフィルタ",
        examples=[["000C7F-E", "001C7F-E"]]
    )
    
    # 証券フィルタ
    active_only: bool = Field(
        default=True,
        description="アクティブな証券のみを取得"
    )
    primary_equity_only: bool = Field(
        default=False,
        description="主力証券のみを取得"
    )
    
    # WolfPeriodベースの期間フィルタ
    period_range: Optional[WolfPeriodRange] = Field(
        default=None,
        description="期間範囲（WolfPeriodRange）",
        examples=["2020M1:2023M12"]
    )
    period_start: Optional[WolfPeriod] = Field(
        default=None,
        description="開始期間（WolfPeriod）",
        examples=["2020M1"]
    )
    period_end: Optional[WolfPeriod] = Field(
        default=None,
        description="終了期間（WolfPeriod）",
        examples=["2023M12"]
    )
    
    # データ品質フィルタ
    exclude_zero_sales: bool = Field(
        default=True,
        description="売上高ゼロの企業を除外"
    )
    max_fterm: Optional[conint(ge=190001, le=299912)] = Field(
        default=202412,
        description="最大会計期間（YYYYMM形式）",
        examples=[202412]
    )
    
    @field_validator('period_range', mode='before')
    @classmethod
    def validate_period_range(cls, v):
        """WolfPeriodRangeの適切な処理を確保します。"""
        if v is None:
            return v
        # 既にWolfPeriodRangeインスタンスの場合はそのまま返す
        if isinstance(v, WolfPeriodRange):
            return v
        # その他の変換処理は既存のWolfPeriodRangeの機能に委ねる
        return v

    # 国コードフィルタ（パフォーマンス最適化）
    country_codes: Optional[List[Union[Alpha2Code, Alpha3Code]]] = Field(
        default=None,
        description="国コードリスト（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-2に変換）",
        examples=[["US", "USA", "JP", "JPN"]]
    )
    
    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    
    @field_validator("country_codes", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """国コードリストをalpha-2に正規化。"""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("country_codesはリストである必要があります")
        
        normalized_codes = []
        for code in v:
            if isinstance(code, str):
                normalized = convert_to_alpha2(code.strip())
                if normalized:
                    normalized_codes.append(normalized)
        
        return normalized_codes if normalized_codes else None
    
    @field_serializer("period_range", "period_start", "period_end")
    def _serialize_periods(self, period: Optional[Union[WolfPeriod, WolfPeriodRange]]) -> Optional[Dict[str, Any]]:
        """WolfPeriod/WolfPeriodRangeをシリアライズ。"""
        return period.model_dump() if period else None
