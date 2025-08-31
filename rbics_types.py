"""
RBICS型定義
============

FactSet REVEREデータ（RBICS）の型定義とバリデーション
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import List, Optional, Dict, Any, Union
from enum import Enum

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    computed_field,
    model_validator,
    field_serializer,
)
from pydantic.types import conint, constr, confloat

from wolf_period import WolfPeriod, WolfPeriodRange, Frequency
from gppm.utils.country_code_manager import (
    convert_to_alpha3,
    Alpha2Code,
    Alpha3Code
)


class RBICSLevel(str, Enum):
    """RBICS階層レベル"""
    L1 = "L1"  # セクター
    L2 = "L2"  # 業界グループ  
    L3 = "L3"  # 業界
    L4 = "L4"  # サブ業界
    L5 = "L5"  # 詳細サブ業界
    L6 = "L6"  # 最詳細分類


class SegmentType(str, Enum):
    """セグメントタイプ"""
    REVENUE = "revenue"     # 売上セグメント
    FOCUS = "focus"         # フォーカス（主力事業）
    STRUCTURE = "structure" # マスタ構造


class RBICSStructureRecord(BaseModel):
    """RBICS構造マスタレコード（読み取り専用）。

    概要:
    - RBICS階層構造の定義情報（L1～L6の階層とその名称）を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - 6階層のRBICS分類コードと名称の管理
    - 階層間の一貫性検証
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
    
    # RBICS階層コード（L1～L6）
    l1_id: Optional[constr(min_length=2, max_length=2)] = Field(
        default=None,
        description="L1コード（セクター、2桁）",
        examples=["10", "15", "20"]
    )
    l2_id: Optional[constr(min_length=4, max_length=4)] = Field(
        default=None,
        description="L2コード（業界グループ、4桁）",
        examples=["1010", "1510", "2010"]
    )
    l3_id: Optional[constr(min_length=6, max_length=6)] = Field(
        default=None,
        description="L3コード（業界、6桁）",
        examples=["101010", "151010", "201010"]
    )
    l4_id: Optional[constr(min_length=8, max_length=8)] = Field(
        default=None,
        description="L4コード（サブ業界、8桁）",
        examples=["10101010", "15101010", "20101010"]
    )
    l5_id: Optional[constr(min_length=10, max_length=10)] = Field(
        default=None,
        description="L5コード（詳細サブ業界、10桁）",
        examples=["1010101010", "1510101010", "2010101010"]
    )
    l6_id: constr(min_length=12, max_length=12) = Field(
        description="L6コード（最詳細分類、12桁）",
        examples=["101010101010", "151010101010", "201010101010"]
    )
    
    # RBICS階層名称（L1～L6）
    l1_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="L1名称（セクター名）",
        examples=["Energy", "Materials", "Industrials"]
    )
    l2_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="L2名称（業界グループ名）",
        examples=["Energy Equipment & Services", "Materials", "Capital Goods"]
    )
    l3_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="L3名称（業界名）",
        examples=["Oil & Gas Drilling", "Chemicals", "Aerospace & Defense"]
    )
    l4_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="L4名称（サブ業界名）",
        examples=["Oil & Gas Drilling", "Diversified Chemicals", "Aerospace & Defense"]
    )
    l5_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="L5名称（詳細サブ業界名）",
        examples=["Oil & Gas Drilling", "Diversified Chemicals", "Aerospace & Defense"]
    )
    l6_name: constr(min_length=1, max_length=255) = Field(
        description="L6名称（最詳細分類名）",
        examples=["Oil & Gas Drilling", "Diversified Chemicals", "Aerospace & Defense"]
    )
    
    # 説明・詳細情報
    sector_description: Optional[constr(min_length=1, max_length=5000)] = Field(
        default=None,
        description="セクター説明",
        examples=["Energy sector includes companies involved in exploration and production of oil and gas"]
    )
    
    # 有効期間情報
    effective_start: Optional[date] = Field(
        default=None,
        description="有効開始日",
        examples=[date(2020, 1, 1)]
    )
    effective_end: Optional[date] = Field(
        default=None,
        description="有効終了日",
        examples=[date(2024, 12, 31)]
    )
    
    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )
    
    @model_validator(mode="after")
    def _validate_hierarchy_consistency(self) -> "RBICSStructureRecord":
        """RBICS階層の一貫性を検証。"""
        # L6は必須（最詳細分類）
        if not self.l6_id:
            raise ValueError("L6コードは必須です")
        
        # 階層の一貫性チェック（上位階層が存在する場合、下位階層も存在すべき）
        levels = [
            (self.l1_id, "L1"),
            (self.l2_id, "L2"),
            (self.l3_id, "L3"),
            (self.l4_id, "L4"),
            (self.l5_id, "L5"),
            (self.l6_id, "L6"),
        ]
        
        # 非Noneの階層が連続していることを確認
        has_value = [level[0] is not None for level in levels]
        if has_value != [True] * len(has_value):
            # 最後のTrueまでが連続していることを確認
            last_true_idx = max(i for i, v in enumerate(has_value) if v)
            if not all(has_value[:last_true_idx + 1]):
                raise ValueError("RBICS階層は連続している必要があります")
        
        return self
    
    @computed_field
    @property
    def primary_sector(self) -> str:
        """主要セクター（L1名称またはL6名称）。"""
        return self.l1_name or self.l6_name
    
    @computed_field
    @property
    def classification_level(self) -> RBICSLevel:
        """分類レベル（最も詳細な階層）。"""
        if self.l6_id:
            return RBICSLevel.L6
        elif self.l5_id:
            return RBICSLevel.L5
        elif self.l4_id:
            return RBICSLevel.L4
        elif self.l3_id:
            return RBICSLevel.L3
        elif self.l2_id:
            return RBICSLevel.L2
        else:
            return RBICSLevel.L1
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.retrieved_period)
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class RBICSCompanyRecord(BaseModel):
    """RBICS企業情報レコード（読み取り専用）。

    概要:
    - 企業の基本情報とRBICS分類情報を保持します。
    - 売上セグメント情報やフォーカス情報を含みます。

    特徴:
    - 企業の識別子情報（複数の証券識別子）
    - RBICS分類と売上シェア情報
    - 地域・国情報の統合管理
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 企業基本情報
    company_id: constr(min_length=1, max_length=20) = Field(
        description="REVERE Company ID",
        examples=["123456789"]
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
        description="ティッカーシンボル",
        examples=["AAPL", "MSFT", "7203"]
    )
    
    # 地理情報
    region_name: Optional[constr(min_length=1, max_length=100)] = Field(
        default=None,
        description="地域名",
        examples=["North America", "Asia Pacific", "Europe"]
    )
    region_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="地域コード",
        examples=["US", "USA", "JP", "JPN"]
    )
    hq_region_code: Optional[Union[Alpha2Code, Alpha3Code]] = Field(
        default=None,
        description="本社所在地域コード",
        examples=["US", "USA", "JP", "JPN"]
    )
    
    # RBICS分類情報
    segment_type: SegmentType = Field(
        description="セグメントタイプ",
        examples=[SegmentType.REVENUE, SegmentType.FOCUS]
    )
    
    # RBICS ID（セグメントタイプに応じて使い分け）
    focus_l6_id: Optional[constr(min_length=12, max_length=12)] = Field(
        default=None,
        description="会社単体に与えられるRBICS L6分類コード（12桁）",
        examples=["101010101010"]
    )
    revenue_l6_id: Optional[constr(min_length=12, max_length=12)] = Field(
        default=None,
        description="会社のセグメントに与えられるRBICS L6分類コード（12桁）",
        examples=["101010101010"]
    )
    
    # セグメント詳細情報（売上セグメントの場合）
    segment_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="セグメントID",
        examples=["SEG123456"]
    )
    segment_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="セグメント名",
        examples=["Consumer Electronics", "Software Services"]
    )
    revenue_share: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="売上シェア（小数点形式）",
        examples=[0.65, 0.25, 0.10]
    )
    
    # 期間情報
    period_end_date: Optional[date] = Field(
        default=None,
        description="期間終了日",
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
    
    @field_validator("ticker", mode="before")
    @classmethod
    def _normalize_ticker(cls, v: Optional[str]) -> Optional[str]:
        """ティッカーを正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()
    
    @field_validator("region_code", "hq_region_code", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[str]) -> Optional[str]:
        """国コードをISO 3166-1 alpha-3に正規化。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return convert_to_alpha3(v.strip())
    
    @model_validator(mode="after")
    def _validate_segment_consistency(self) -> "RBICSCompanyRecord":
        """セグメント情報の整合性を検証。"""
        # 売上セグメントの場合はセグメント詳細情報とREVENUE_L6_IDが必要
        if self.segment_type == SegmentType.REVENUE:
            if not self.segment_id or not self.segment_name:
                raise ValueError("売上セグメントにはセグメントIDと名称が必要です")
            if not self.revenue_l6_id:
                raise ValueError("売上セグメントにはREVENUE_L6_IDが必要です")
            if self.focus_l6_id:
                raise ValueError("売上セグメントではFOCUS_L6_IDは使用されません")
        
        # フォーカスセグメントの場合はFOCUS_L6_IDが必要
        elif self.segment_type == SegmentType.FOCUS:
            if not self.focus_l6_id:
                raise ValueError("フォーカスセグメントにはFOCUS_L6_IDが必要です")
            if self.revenue_l6_id:
                raise ValueError("フォーカスセグメントではREVENUE_L6_IDは使用されません")
            if self.segment_id or self.segment_name or self.revenue_share:
                raise ValueError("フォーカスセグメントではセグメント詳細情報は使用されません")
        
        # 少なくとも1つの識別子が必要
        identifiers = [self.factset_entity_id, self.cusip, self.isin, self.sedol, self.ticker]
        if not any(identifiers):
            raise ValueError("少なくとも1つの証券識別子が必要です")
        
        return self
    
    @computed_field
    @property
    def primary_identifier(self) -> Optional[str]:
        """主要識別子（優先順位: ISIN → CUSIP → SEDOL → Ticker → FactSet Entity ID）。"""
        return (
            self.isin or 
            self.cusip or 
            self.sedol or 
            self.ticker or 
            self.factset_entity_id
        )
    
    @computed_field
    @property
    def is_revenue_segment(self) -> bool:
        """売上セグメントかどうか（遅延評価）。"""
        return self.segment_type == SegmentType.REVENUE
    
    @computed_field
    @property
    def rbics_l6_id(self) -> Optional[str]:
        """適用可能なRBICS L6 ID（セグメントタイプに応じて自動選択）。"""
        if self.segment_type == SegmentType.REVENUE:
            return self.revenue_l6_id
        elif self.segment_type == SegmentType.FOCUS:
            return self.focus_l6_id
        return None
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.retrieved_period)
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class RBICSQueryParams(BaseModel):
    """RBICS クエリパラメータ（WolfPeriod対応）。

    目的:
    - 取得対象（企業/分類/地域）と期間を安全に指定
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
    factset_entity_ids: Optional[List[constr(min_length=1, max_length=20)]] = Field(
        default=None,
        description="FactSet Entity IDリスト",
        examples=[["001C7F-E", "002D8G-F"]]
    )
    
    # 証券識別子フィルタ
    security_ids: Optional[List[constr(min_length=1, max_length=50)]] = Field(
        default=None,
        description="証券識別子リスト",
        examples=[["US0378331005", "037833100"]]
    )
    identifier_type: constr(min_length=1, max_length=10) = Field(
        default="isin",
        description="識別子タイプ",
        examples=["isin", "cusip", "sedol", "ticker"]
    )
    
    # セグメントフィルタ
    segment_types: Optional[List[SegmentType]] = Field(
        default=None,
        description="セグメントタイプリスト",
        examples=[[SegmentType.REVENUE, SegmentType.FOCUS]]
    )
    include_revenue_segments: bool = Field(
        default=True,
        description="売上セグメントを含むかどうか"
    )
    include_focus_segments: bool = Field(
        default=True,
        description="フォーカスセグメントを含むかどうか"
    )
    
    # RBICS分類フィルタ
    rbics_levels: Optional[List[RBICSLevel]] = Field(
        default=None,
        description="取得するRBICS階層レベル",
        examples=[[RBICSLevel.L1, RBICSLevel.L6]]
    )
    rbics_l6_ids: Optional[List[constr(min_length=12, max_length=12)]] = Field(
        default=None,
        description="RBICS L6分類コードリスト（12桁）",
        examples=[["101010101010", "151010101010"]]
    )
    
    # 地域フィルタ
    region_codes: Optional[List[Union[Alpha2Code, Alpha3Code]]] = Field(
        default=None,
        description="地域コードリスト（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-3に変換）",
        examples=[["US", "USA", "JP", "JPN"]]
    )
    
    # WolfPeriodベースの期間フィルタ
    period: Optional[Union[WolfPeriod, WolfPeriodRange]] = Field(
        default=None,
        description="期間指定（WolfPeriodまたはWolfPeriodRange）",
        examples=["WolfPeriod.from_month(2023, 12)", "WolfPeriodRange(start, end)"]
    )
    period_start: Optional[WolfPeriod] = Field(
        default=None,
        description="開始期間（WolfPeriod）",
        examples=["WolfPeriod.from_month(2020, 1)"]
    )
    period_end: Optional[WolfPeriod] = Field(
        default=None,
        description="終了期間（WolfPeriod）",
        examples=["WolfPeriod.from_month(2023, 12)"]
    )
    
    # 後方互換性（非推奨）
    as_of_date: Optional[date] = Field(
        default=None,
        description="基準日（後方互換性用、非推奨。periodの使用を推奨）",
        examples=[date(2023, 12, 31)]
    )
    
    # データ品質フィルタ
    min_revenue_share: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="最小売上シェア（売上セグメント用）",
        examples=[0.05, 0.10]
    )
    exclude_zero_revenue: bool = Field(
        default=True,
        description="売上ゼロのセグメントを除外"
    )
    
    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    
    @field_validator("identifier_type", mode="before")
    @classmethod
    def _normalize_identifier_type(cls, v: str) -> str:
        """識別子タイプを正規化（小文字化）。"""
        if not isinstance(v, str):
            raise ValueError("識別子タイプは文字列である必要があります")
        return v.strip().lower()
    
    @field_validator("region_codes", mode="before")
    @classmethod
    def _normalize_region_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """地域コードリストをalpha-3に正規化。"""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("region_codesはリストである必要があります")
        
        normalized_codes = []
        for code in v:
            if isinstance(code, str):
                normalized = convert_to_alpha3(code.strip())
                if normalized:
                    normalized_codes.append(normalized)
        
        return normalized_codes if normalized_codes else None
    
    @model_validator(mode="after")
    def _validate_query_consistency(self) -> "RBICSQueryParams":
        """クエリパラメータの整合性を検証。"""
        # セグメントタイプの一貫性
        if self.segment_types is not None:
            if not self.include_revenue_segments and SegmentType.REVENUE in self.segment_types:
                raise ValueError("include_revenue_segmentsがFalseの場合、segment_typesにREVENUEを含めることはできません")
            if not self.include_focus_segments and SegmentType.FOCUS in self.segment_types:
                raise ValueError("include_focus_segmentsがFalseの場合、segment_typesにFOCUSを含めることはできません")
        
        # 識別子タイプの妥当性
        valid_types = ["isin", "cusip", "sedol", "ticker", "company_id", "factset_entity_id"]
        if self.identifier_type not in valid_types:
            raise ValueError(f"identifier_typeは以下のいずれかである必要があります: {valid_types}")
        
        return self
    
    @field_serializer("period", "period_start", "period_end")
    def _serialize_periods(self, period: Optional[Union[WolfPeriod, WolfPeriodRange]]) -> Optional[Dict[str, Any]]:
        """WolfPeriod/WolfPeriodRangeをシリアライズ。"""
        return period.model_dump() if period else None


__all__ = [
    "RBICSLevel",
    "SegmentType", 
    "RBICSStructureRecord",
    "RBICSCompanyRecord",
    "RBICSQueryParams",
]
