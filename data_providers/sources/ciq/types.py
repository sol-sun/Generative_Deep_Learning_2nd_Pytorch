"""
CIQ プロバイダー用型定義
======================

CIQ (S&P Capital IQ) 固有のデータレコード型定義。
元々のCIQプロバイダーにあった型定義を分離。

使用例:
    from data_providers.sources.ciq.types import (
        CIQIdentityRecord,
        CIQFinancialRecord,
    )
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Union, Dict, Any
from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    field_validator,
    computed_field,
    field_serializer,
)
from pydantic.types import conint, constr

from wolf_period import WolfPeriod

from gppm.utils.country_code_manager import convert_to_alpha3, Alpha2Code, Alpha3Code


class CIQIdentityRecord(BaseModel):
    """CIQ 企業識別子レコード（読み取り専用）。

    概要:
    - 企業の基本情報（Company ID、名前、各種識別コード）を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - 企業の基本情報（名前、各種識別コード）を統合管理
    - 上場・非上場の自動判別機能
    - データの取得時期を明確に記録
    - 国コードはalpha-3形式で統一管理
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        # パフォーマンス最適化
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 企業基本情報
    company_id: Optional[conint(ge=1, le=9999999999)] = Field(
        default=None,
        description="CIQ Company ID (10桁)",
        #examples=[]
    )
    company_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="企業名",
        examples=["Apple Inc.", "Microsoft Corporation"]
    )
    
    # 証券識別子（パターンマッチング最適化）
    cusip: Optional[constr(pattern=r"^[0-9A-Z]{9}$")] = Field(
        default=None,
        description="CUSIP識別子",
        examples=["037833100"]
    )
    isin: Optional[constr(pattern=r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")] = Field(
        default=None,
        description="ISIN識別子",
        examples=["US0378331005"]
    )
    sedol: Optional[constr(pattern=r"^[0-9A-Z]{7}$")] = Field(
        default=None,
        description="SEDOL識別子",
        examples=["B0YQ5F0"]
    )
    ticker: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="ティッカーシンボル",
        examples=["AAPL", "MSFT", "7203"]
    )
    
    # 地理情報（ISO国コード制約）
    headquarters_country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="本社所在地国コード（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-3に変換）",
        examples=["US", "USA", "JP", "JPN"]
    )
    exchange_country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="上場国コード（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-3に変換）",
        examples=["US", "USA", "JP", "JPN"]
    )
    
    # 上場状況
    un_listed_flg: Optional[conint(ge=0, le=1)] = Field(
        default=None,
        description="非上場フラグ（0=上場、1=非上場）",
        examples=[0, 1]
    )
    
    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )
    
    @field_validator("cusip", "isin", "sedol", mode="before")
    @classmethod
    def _normalize_security_codes(cls, v: Optional[str]) -> Optional[str]:
        """証券識別子を正規化（大文字化・英数字チェック）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("証券識別子は文字列である必要があります")
        normalized = v.strip().upper()
        if not normalized.isalnum():
            raise ValueError("証券識別子は英数字のみ")
        return normalized
    
    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, v: Optional[str]) -> Optional[str]:
        """ティッカーを正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        # CIQのティッカーは取引所サフィックスを含まない前提
        return v.strip().upper()
    
    @field_validator("headquarters_country_code", "exchange_country_code", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[str]) -> Optional[str]:
        """国コードをISO 3166-1 alpha-3に正規化。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return convert_to_alpha3(v.strip())
    
    @computed_field
    @property
    def is_listed(self) -> Optional[bool]:
        """上場フラグ（遅延評価）。"""
        if self.un_listed_flg is None:
            return None
        return self.un_listed_flg == 0
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class CIQFinancialRecord(BaseModel):
    """CIQ 財務データレコード（将来拡張用）。

    概要:
    - 財務データ取得機能に備えたプレースホルダーです。
    - 期間は `WolfPeriod` で表現します。

    特徴:
    - 将来の財務データ取得機能に対応
    - 期間管理による時系列データの整理
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 基本情報
    company_id: conint(ge=1, le=9999999999) = Field(
        description="CIQ Company ID (10桁)",
        examples=[1234567890]
    )
    period: WolfPeriod = Field(
        description="データ期間（WolfPeriod）",
        examples=["2023Q4", "2023-12"]
    )
    currency: Optional[constr(pattern=r"^[A-Z]{3}$")] = Field(
        default=None,
        description="通貨コード",
        examples=["USD", "JPY", "EUR"]
    )
    
    # 将来拡張用フィールド（現在は使用しない）
    revenue: Optional[float] = Field(
        default=None,
        description="売上高（将来実装予定）"
    )
    operating_income: Optional[float] = Field(
        default=None,
        description="営業利益（将来実装予定）"
    )
    net_income: Optional[float] = Field(
        default=None,
        description="純利益（将来実装予定）"
    )
    total_assets: Optional[float] = Field(
        default=None,
        description="総資産（将来実装予定）"
    )
    
    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        """通貨コードを正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.period)
    
    @field_serializer("period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()

