"""
MSCI プロバイダー固有の型定義
=============================

MSCI(モルガン・スタンレー・キャピタル・インターナショナル)固有のデータレコード型定義

使用例:
    from data_providers.data_sources.msci.types import (
        MSCIIndexRecord,
        MSCISecurityRecord,
        IndexType,
        DividendType
    )
"""

from __future__ import annotations

from datetime import datetime, timezone, date
from typing import List, Optional, Union, Dict, Any
from enum import Enum
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    computed_field,
    field_serializer,
)
from pydantic.types import constr, confloat

from wolf_period import WolfPeriod

from gppm.utils.country_code_manager import (
    convert_to_alpha3,
    Alpha2Code,
    Alpha3Code
)


class IndexType(str, Enum):
    """MSCIインデックスタイプ"""
    GLOBAL = "global"
    LOCAL = "local"


class DividendType(str, Enum):
    """配当タイプ"""
    GROSS = "gross"  # 配当なし
    NET = "net"      # 配当込み


class MSCIIndexRecord(BaseModel):
    """MSCIインデックス情報レコード（読み取り専用）。

    概要:
    - インデックスの基本情報（コード、名称、タイプ、配当種別）を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - グローバル・ローカルインデックスの自動判別機能
    - 配当込み/なしの明確な区別
    - データの取得時期を明確に記録
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # インデックス基本情報
    index_code: constr(min_length=1, max_length=10) = Field(
        description="MSCIインデックスコード",
        examples=["@DP", "@DT", "@AP", "@AT", "JPN", "USA"]
    )
    index_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="インデックス名称",
        examples=["MSCI World Index", "MSCI ACWI", "MSCI Japan"]
    )
    index_type: IndexType = Field(
        description="インデックスタイプ（グローバル/ローカル）",
        examples=[IndexType.GLOBAL, IndexType.LOCAL]
    )
    dividend_type: DividendType = Field(
        description="配当タイプ（配当なし/配当込み）",
        examples=[DividendType.GROSS, DividendType.NET]
    )
    
    # 地理情報（ローカルインデックスの場合）
    country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="国コード（ローカルインデックスの場合のみ）",
        examples=["US", "USA", "JP", "JPN"]
    )
    
    # インデックス値情報
    index_value: Optional[confloat(ge=0.0)] = Field(
        default=None,
        description="インデックス値",
        examples=[1234.56, 5678.90]
    )
    usd_rate: Optional[confloat(ge=0.0)] = Field(
        default=None,
        description="USD為替レート",
        examples=[1.0, 110.25, 0.85]
    )
    
    # 期間情報
    data_date: date = Field(
        description="データ日付",
        examples=[date(2023, 12, 31)]
    )
    
    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )
    
    @field_validator("index_code", mode="before")
    @classmethod
    def _normalize_index_code(cls, v: str) -> str:
        """インデックスコードを正規化（大文字化）。"""
        if not isinstance(v, str):
            raise ValueError("インデックスコードは文字列である必要があります")
        return v.strip().upper()
    
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
    def _validate_index_consistency(self) -> "MSCIIndexRecord":
        """インデックスタイプと国コードの整合性を検証。"""
        if self.index_type == IndexType.LOCAL and self.country_code is None:
            raise ValueError("ローカルインデックスには国コードが必要です")
        if self.index_type == IndexType.GLOBAL and self.country_code is not None:
            raise ValueError("グローバルインデックスには国コードを設定できません")
        return self
    
    @computed_field
    @property
    def is_global_index(self) -> bool:
        """グローバルインデックスかどうか（遅延評価）。"""
        return self.index_type == IndexType.GLOBAL
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.retrieved_period)
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class MSCISecurityRecord(BaseModel):
    """MSCI証券情報レコード（読み取り専用）。

    概要:
    - 証券の基本情報（識別子、価格、時価総額、セクター情報）を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - 複数の証券識別子（ISIN/SEDOL/CUSIP）の統合管理
    - セクター・業界情報の階層化
    - 企業コード情報との連携
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 証券識別子
    msci_security_code: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="MSCI証券コード（数値コード）",
        examples=["230501"]
    )
    msci_issuer_code: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="MSCI発行体コード（数値コード）",
        examples=["230501"]
    )
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
    
    # 地理・セクター情報
    country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="国コード",
        examples=["US", "USA", "JP", "JPN"]
    )
    sector: Optional[int] = Field(
        default=None,
        description="セクター（2桁の数値コード）",
        examples=[10, 15, 20],
        ge=10,
        le=99
    )
    industry_group: Optional[int] = Field(
        default=None,
        description="業界グループ（4桁の数値コード）",
        examples=[1010, 1510, 2010],
        ge=1000,
        le=9999
    )
    industry: Optional[int] = Field(
        default=None,
        description="業界（6桁の数値コード）",
        examples=[101010, 151010, 201010],
        ge=100_000,
        le=999_999
    )
    sub_industry: Optional[int] = Field(
        default=None,
        description="サブ業界（8桁の数値コード）",
        examples=[10101010, 15101010, 20101010],
        ge=10_000_000,
        le=99_999_999
    )
    
    # 価格・時価総額情報
    price: Optional[confloat(ge=0.0)] = Field(
        default=None,
        description="価格",
        examples=[192.53, 150.25]
    )
    currency: Optional[constr(pattern=r"^[A-Z]{3}$")] = Field(
        default=None,
        description="通貨コード",
        examples=["USD", "JPY", "EUR"]
    )
    market_cap_usd: Optional[confloat(ge=0.0)] = Field(
        default=None,
        description="時価総額（USD）",
        examples=[2916000000000.0, 1500000000000.0]
    )
    
    # 包含率情報
    foreign_inclusion_factor: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="外国包含率",
        examples=[1.0, 0.5, 0.25]
    )
    domestic_inclusion_factor: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="国内包含率",
        examples=[1.0, 0.5, 0.25]
    )
    
    # データ日付
    data_date: date = Field(
        description="データ日付",
        examples=[date(2023, 12, 31)]
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
    
    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        """通貨コードを正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()
    
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
    def _validate_security_consistency(self) -> "MSCISecurityRecord":
        """証券情報の整合性を検証。"""
        # 少なくとも1つの識別子が必要
        identifiers = [self.msci_security_code, self.cusip, self.isin, self.sedol]
        if not any(identifiers):
            raise ValueError("少なくとも1つの証券識別子が必要です")
        
        # 価格と通貨の整合性
        if self.price is not None and self.currency is None:
            raise ValueError("価格が設定されている場合は通貨コードが必要です")
        
        return self
    
    @computed_field
    @property
    def primary_identifier(self) -> Optional[str]:
        """主要識別子（優先順位: ISIN → CUSIP → SEDOL → MSCIコード）。"""
        return self.isin or self.cusip or self.sedol or self.msci_security_code
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.retrieved_period)
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()
