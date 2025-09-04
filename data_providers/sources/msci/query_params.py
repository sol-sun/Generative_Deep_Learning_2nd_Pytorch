"""
MSCI プロバイダー用クエリパラメーター
==================================

MSCI固有のクエリパラメーター定義。
複雑なMSCI検索条件の型安全な定義とバリデーション・変換ロジック。

使用例:
    from data_providers.data_sources.msci.query_params import MSCIQueryParams
"""

from __future__ import annotations

from datetime import date
from typing import List, Optional, Union, Dict, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    field_serializer,
    model_validator,
)
from pydantic.types import conint, constr

from wolf_period import WolfPeriod, WolfPeriodRange
from gppm.utils.country_code_manager import (
    convert_to_alpha3,
    Alpha2Code,
    Alpha3Code
)


class MSCIQueryParams(BaseModel):
    """MSCIクエリパラメータ（WolfPeriod対応）。

    目的:
    - 取得対象（インデックス/証券/期間）と計算条件を安全に指定
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
    
    # インデックスフィルタ
    index_name: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="インデックス名（'WORLD', 'ACWI'）",
        examples=["WORLD", "ACWI"]
    )
    country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="国コード（ローカルインデックス用）",
        examples=["US", "USA", "JP", "JPN"]
    )
    dividend_flag: bool = Field(
        default=False,
        description="配当込みインデックスフラグ"
    )
    
    # 証券フィルタ
    security_ids: Optional[List[constr(min_length=1, max_length=50)]] = Field(
        default=None,
        description="証券識別子リスト",
        examples=[["US0378331005", "B0YQ5F0"]]
    )
    identifier_type: constr(min_length=1, max_length=10) = Field(
        default="isin",
        description="識別子タイプ",
        examples=["isin", "sedol", "cusip"]
    )
    
    # 統一された期間指定（WolfPeriodベース）
    period: Optional[Union[WolfPeriod, WolfPeriodRange, List[WolfPeriod]]] = Field(
        default=None,
        description=(
            "期間指定（統一パラメータ）。以下の形式をサポート：\n"
            "- WolfPeriod: 単一期間（構成銘柄取得等）\n"
            "- WolfPeriodRange: 期間範囲（時系列データ取得等）\n"
            "- List[WolfPeriod]: 複数期間（ベータ計算等）"
        ),
        examples=[
            "WolfPeriod.from_month(2023, 12, freq=Frequency.M)",
            "WolfPeriodRange(start_period, end_period)",
            "[WolfPeriod.from_month(2023, 12, freq=Frequency.M), WolfPeriod.from_month(2023, 11, freq=Frequency.M)]"
        ]
    )
    
    # ベータ計算パラメータ
    lookback_periods: conint(ge=1, le=1200) = Field(
        default=60,
        description="遡及期間数",
        examples=[60, 120, 240]
    )
    frequency: constr(min_length=1, max_length=10) = Field(
        default="monthly",
        description="計算頻度",
        examples=["monthly", "weekly", "daily"]
    )
    levered: bool = Field(
        default=True,
        description="レバードベータフラグ"
    )
    
    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    
    @field_validator("index_name", mode="before")
    @classmethod
    def _normalize_index_name(cls, v: Optional[str]) -> Optional[str]:
        """インデックス名を正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()
    
    @field_validator("identifier_type", "frequency", mode="before")
    @classmethod
    def _normalize_string_params(cls, v: str) -> str:
        """文字列パラメータを正規化（小文字化）。"""
        if not isinstance(v, str):
            raise ValueError("文字列パラメータは文字列である必要があります")
        return v.strip().lower()
    
    @field_validator("country_code", mode="before")
    @classmethod
    def _normalize_country_code(cls, v: Optional[str]) -> Optional[str]:
        """国コードをISO 3166-1 alpha-3に正規化。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return convert_to_alpha3(v.strip())
    
    @model_validator(mode="after")
    def _validate_query_consistency(self) -> "MSCIQueryParams":
        """クエリパラメータの整合性を検証。"""
        # インデックス指定の整合性
        if self.index_name is None and self.country_code is None:
            raise ValueError("index_name または country_code のいずれかを指定してください")
        
        # 頻度の妥当性
        if self.frequency not in ["monthly", "weekly", "daily"]:
            raise ValueError("frequency は 'monthly', 'weekly', 'daily' のいずれかである必要があります")
        
        return self
    
    @field_serializer("period")
    def _serialize_dates(self, value: Optional[Union[WolfPeriod, WolfPeriodRange, List[WolfPeriod]]]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """期間関連フィールドをシリアライズ。"""
        if value is None:
            return None
        if isinstance(value, WolfPeriod):
            return value.model_dump()
        if isinstance(value, WolfPeriodRange):
            return value.model_dump()
        if isinstance(value, list) and all(isinstance(p, WolfPeriod) for p in value):
            return [p.model_dump() for p in value]
        return value
