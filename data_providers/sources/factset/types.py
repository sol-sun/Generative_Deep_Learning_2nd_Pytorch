"""
FactSet データプロバイダー型定義
==============================

FactSet固有のデータレコード型定義。

使用例:
    from data_providers.sources.factset.types import (
        FactSetIdentityRecord,
        FactSetFinancialRecord,
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    computed_field,
    field_serializer,
)
from pydantic.types import conint, constr

from wolf_period import WolfPeriod
from gppm.utils.country_code_manager import convert_to_alpha2, Alpha2Code, Alpha3Code


class FactSetIdentityRecord(BaseModel):
    """FactSet 企業識別子レコード（読み取り専用）。

    概要:
    - 企業名や各種コード（ISIN/SEDOL/CUSIP/TICKERなど）、主力証券情報を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - 企業の基本情報（名前、各種識別コード）を統合管理
    - 主力証券の自動識別機能
    - データの取得時期を明確に記録
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
    fsym_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="FactSet Symbol ID",
        examples=["000C7F-E"]
    )
    factset_entity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="FactSet Entity ID",
        examples=["001C7F-E"]
    )
    company_name: Optional[constr(min_length=1, max_length=512)] = Field(
        default=None,
        description="企業名",
        examples=["Apple Inc."]
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
        description="ティッカーシンボル（取引所サフィックス含む）",
        examples=["AAPL", "AAPL.OQ", "7203.T"]
    )
    ticker_region: Optional[constr(min_length=1, max_length=30)] = Field(
        default=None,
        description="地域別ティッカー",
        examples=["AAPL-US", "7203-JP"]
    )
    
    # 地理情報（ISO国コード制約）
    headquarters_country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="本社所在地国コード（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-2に変換）",
        examples=["US", "USA", "JP", "JPN"]
    )
    exchange_country_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="上場国コード（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-2に変換）",
        examples=["US", "USA", "JP", "JPN"]
    )
    
    # 主力証券識別
    primary_equity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="主力に該当する証券のFSYM_ID",
        examples=["000C7F-E"]
    )
    is_primary_equity: bool = Field(
        default=False,
        description="主力証券フラグ"
    )
    
    # 証券情報
    security_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="証券名",
        examples=["Apple Inc. Common Stock"]
    )
    security_type: Optional[constr(min_length=1, max_length=10)] = Field(
        default=None,
        description="証券タイプ",
        examples=["EQ", "PREF"]
    )
    active_flag: bool = Field(
        default=True,
        description="アクティブフラグ"
    )
    universe_type: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="ユニバースタイプ",
        examples=["EQ", "PREF"]
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
    
    @field_validator("headquarters_country_code", "exchange_country_code", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[str]) -> Optional[str]:
        """国コードをISO 3166-1 alpha-2に正規化。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return convert_to_alpha2(v.strip())
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class FactSetFinancialRecord(BaseModel):
    """FactSet 財務データレコード（読み取り専用）。

    概要:
    - WolfPeriodで表される各期間について、売上・利益・資産・負債・税率などを保持します。
    - 金額や比率は浮動小数点数（float）で表現します。

    特徴:
    - 財務指標の自動計算（負債資本比率など）
    - 会計期や地域情報の付与
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
    fsym_id: constr(min_length=1, max_length=20) = Field(
        description="FactSet Symbol ID",
        examples=["000C7F-E"]
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
    
    # 損益計算書項目（float型）
    ff_sales: Optional[float] = Field(
        default=None,
        description="売上高",
        examples=[365817000000.0]
    )
    ff_oper_inc: Optional[float] = Field(
        default=None,
        description="営業利益",
        examples=[114301000000.0]
    )
    ff_net_inc: Optional[float] = Field(
        default=None,
        description="純利益",
        examples=[99803000000.0]
    )
    ff_ebit_oper: Optional[float] = Field(
        default=None,
        description="営業EBIT",
        examples=[123136000000.0]
    )
    
    # 貸借対照表項目
    ff_assets: Optional[float] = Field(
        default=None,
        description="総資産",
        examples=[352755000000.0]
    )
    ff_assets_curr: Optional[float] = Field(
        default=None,
        description="流動資産",
        examples=[143566000000.0]
    )
    ff_liabs_curr_misc: Optional[float] = Field(
        default=None,
        description="その他流動負債",
        examples=[58829000000.0]
    )
    ff_pay_acct: Optional[float] = Field(
        default=None,
        description="買掛金",
        examples=[62611000000.0]
    )
    ff_debt: Optional[float] = Field(
        default=None,
        description="総負債",
        examples=[123930000000.0]
    )
    ff_debt_lt: Optional[float] = Field(
        default=None,
        description="長期負債",
        examples=[106550000000.0]
    )
    ff_debt_st: Optional[float] = Field(
        default=None,
        description="短期負債",
        examples=[17380000000.0]
    )
    
    # 税務・利息項目
    ff_inc_tax: Optional[float] = Field(
        default=None,
        description="法人税",
        examples=[16741000000.0]
    )
    ff_eq_aff_inc: Optional[float] = Field(
        default=None,
        description="持分法投資損益",
        examples=[0.0]
    )
    ff_int_exp_tot: Optional[float] = Field(
        default=None,
        description="総支払利息",
        examples=[3933000000.0]
    )
    ff_int_exp_debt: Optional[float] = Field(
        default=None,
        description="有利子負債利息（年次データ、ffillで補完）",
        examples=[3800000000.0]
    )
    
    # 市場データ
    ff_mkt_val: Optional[float] = Field(
        default=None,
        description="市場価値（派生データ）",
        examples=[2916000000000.0]
    )
    ff_price_close_fp: Optional[float] = Field(
        default=None,
        description="期末終値",
        examples=[192.53]
    )
    ff_com_shs_out: Optional[float] = Field(
        default=None,
        description="発行済み普通株式数",
        examples=[15441883000.0]
    )
    
    # 財務比率（小数点形式）
    ff_roic: Optional[float] = Field(
        default=None,
        description="投下資本利益率（小数点形式）",
        examples=[0.2856]
    )
    ff_tax_rate: Optional[float] = Field(
        default=None,
        description="税率（小数点形式）",
        examples=[0.1437]
    )
    ff_eff_int_rate: Optional[float] = Field(
        default=None,
        description="実効金利（小数点形式）",
        examples=[0.0318]
    )
    
    # 計算済み指標（遅延評価）
    fixed_assets_total: Optional[float] = Field(
        default=None,
        description="固定資産合計（計算値）"
    )
    invested_capital_operational: Optional[float] = Field(
        default=None,
        description="投下資本（運用ベース、計算値）"
    )
    operating_income_after_tax: Optional[float] = Field(
        default=None,
        description="営業利益（税引後、計算値）"
    )
    
    # 会計期間（WolfPeriodから導出）
    fterm_2: Optional[conint(ge=190001, le=299912)] = Field(
        default=None,
        description="会計期間（YYYYMM形式）",
        examples=[202312]
    )
    
    # 地域情報
    region: Optional[constr(min_length=1, max_length=50)] = Field(
        default=None,
        description="地域区分",
        examples=["North America", "Asia Pacific"]
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
    def debt_to_equity_ratio(self) -> Optional[float]:
        """負債資本比率の計算（遅延評価、float）。"""
        if self.ff_debt is None or self.ff_mkt_val is None or self.ff_mkt_val == 0:
            return None
        try:
            return round(self.ff_debt / self.ff_mkt_val, 4)
        except Exception:
            return None
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.period)
    
    @field_serializer("period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()
