"""
RBICS プロバイダー用クエリパラメーター
====================================

目的
- RBICS固有のクエリパラメーター定義
- 複雑なRBICS検索条件の型安全な定義
- バリデーション・変換ロジック

使用例:
    from data_providers.sources.rbics.query_params import (
        RBICSStructureQueryParams,
        RBICSCompanyQueryParams,
        RBICSQueryParams,
    )
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
)

from wolf_period import WolfPeriod, WolfPeriodRange
from gppm.utils.country_code_manager import convert_to_alpha3

from .types import SegmentType, RBICSLevel


class RBICSStructureQueryParams(BaseModel):
    """RBICS構造マスタ用クエリパラメーター"""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        extra="forbid",
    )
    
    # 期間設定
    period: Optional[WolfPeriod] = Field(
        default=None,
        description="期間（WolfPeriod）",
        examples=["2023Q4", "2023-12"]
    )
    effective_date: Optional[date] = Field(
        default=None,
        description="有効日",
        examples=[date(2023, 12, 31)]
    )
    
    # レベル・コードフィルター
    rbics_levels: Optional[List[RBICSLevel]] = Field(
        default=None,
        description="RBICSレベルリスト",
        examples=[[RBICSLevel.L1, RBICSLevel.L2]]
    )
    rbics_codes: Optional[List[str]] = Field(
        default=None,
        description="RBICSコードリスト",
        examples=[["10", "1010", "101010"]]
    )
    parent_codes: Optional[List[str]] = Field(
        default=None,
        description="親コードリスト",
        examples=[["10", "1010"]]
    )
    
    # テキストフィルター
    description_contains: Optional[str] = Field(
        default=None,
        description="説明に含まれる文字列",
        examples=["Energy", "Technology"]
    )
    
    # 出力設定
    include_inactive: bool = Field(
        default=False,
        description="非アクティブなレコードを含む"
    )
    
    @field_validator("period", "effective_date", mode="after")
    @classmethod
    def _validate_period_consistency(cls, v, info):
        """期間と日付の重複指定チェック"""
        if hasattr(info, 'data') and info.data:
            period = info.data.get('period')
            effective_date = info.data.get('effective_date')
            if period and effective_date:
                raise ValueError("periodとeffective_dateは同時に指定できません")
        return v
    
    @field_serializer("period")
    def _serialize_period(self, period: Optional[WolfPeriod]) -> Optional[Dict[str, Any]]:
        """WolfPeriodをシリアライズ"""
        return period.model_dump() if period else None


class RBICSCompanyQueryParams(BaseModel):
    """RBICS企業データ用クエリパラメーター"""
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
        extra="forbid",
    )
    
    # 基本フィルター
    entity_ids: Optional[List[str]] = Field(
        default=None,
        description="エンティティIDリスト",
        #examples=[[]]
    )
    company_names: Optional[List[str]] = Field(
        default=None,
        description="企業名リスト",
        examples=[["Apple Inc.", "Microsoft Corporation"]]
    )
    
    # 期間設定
    period: Optional[WolfPeriod] = Field(
        default=None,
        description="期間（WolfPeriod）",
        examples=["2023Q4", "2023-12"]
    )
    fiscal_years: Optional[List[int]] = Field(
        default=None,
        description="会計年度リスト",
        examples=[[2020, 2021, 2022, 2023]]
    )
    
    # セグメント種別
    segment_types: Optional[List[SegmentType]] = Field(
        default=None,
        description="セグメントタイプリスト",
        examples=[[SegmentType.REVENUE, SegmentType.FOCUS]]
    )
    
    # RBICSレベル・コードフィルター
    rbics_levels: Optional[List[RBICSLevel]] = Field(
        default=None,
        description="RBICSレベルリスト",
        examples=[[RBICSLevel.L6]]
    )
    rbics_codes: Optional[List[str]] = Field(
        default=None,
        description="RBICSコードリスト",
        examples=[["101010101010"]]
    )
    
    # 地域フィルター
    countries: Optional[List[str]] = Field(
        default=None,
        description="国コードリスト（ISO 3166-1、入力: alpha-2/alpha-3両対応、内部: alpha-3に変換）",
        examples=[["US", "USA", "JP", "JPN"]]
    )
    regions: Optional[List[str]] = Field(
        default=None,
        description="地域リスト",
        examples=[["North America", "Asia Pacific"]]
    )
    
    # データ品質フィルター
    min_revenue_threshold: Optional[float] = Field(
        default=None,
        description="最小売上閾値",
        examples=[1000000.0]
    )
    exclude_zero_revenue: bool = Field(
        default=True,
        description="売上ゼロのレコードを除外"
    )
    require_segment_breakdown: bool = Field(
        default=False,
        description="セグメント内訳を必須とする"
    )
    
    # パフォーマンス設定
    batch_size: int = Field(
        default=5000,
        description="バッチサイズ"
    )
    max_workers: int = Field(
        default=4,
        description="最大ワーカー数"
    )
    use_cache: bool = Field(
        default=True,
        description="キャッシュを使用"
    )
    
    # 出力設定
    include_metadata: bool = Field(
        default=True,
        description="メタデータを含む"
    )
    aggregate_by_level: Optional[RBICSLevel] = Field(
        default=None,
        description="集計レベル",
        examples=[RBICSLevel.L6]
    )
    
    @field_validator("countries", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """国コードリストをalpha-3に正規化"""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("countriesはリストである必要があります")
        
        normalized_codes = []
        for code in v:
            if isinstance(code, str):
                normalized = convert_to_alpha3(code.strip())
                if normalized:
                    normalized_codes.append(normalized)
        
        return normalized_codes if normalized_codes else None
    
    @field_serializer("period")
    def _serialize_period(self, period: Optional[WolfPeriod]) -> Optional[Dict[str, Any]]:
        """WolfPeriodをシリアライズ"""
        return period.model_dump() if period else None


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
    factset_entity_ids: Optional[List[str]] = Field(
        default=None,
        description="FactSet Entity IDリスト",
        examples=[["001C7F-E", "002D8G-F"]]
    )
    
    # 証券識別子フィルタ
    security_ids: Optional[List[str]] = Field(
        default=None,
        description="証券識別子リスト",
        examples=[["US0378331005", "037833100"]]
    )
    identifier_type: str = Field(
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
    
    # 期間フィルタ
    period: Optional[WolfPeriod] = Field(
        default=None,
        description="期間（WolfPeriod）",
        examples=["2023Q4", "2023-12"]
    )
    as_of_date: Optional[date] = Field(
        default=None,
        description="基準日",
        examples=[date(2023, 12, 31)]
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
    
    # 地域フィルタ
    region_codes: Optional[List[str]] = Field(
        default=None,
        description="地域コードリスト",
        examples=[["US", "JP", "GB"]]
    )
    
    # データ品質フィルタ
    min_revenue_share: Optional[float] = Field(
        default=None,
        description="最小売上シェア",
        examples=[0.05]
    )
    exclude_zero_revenue: bool = Field(
        default=True,
        description="売上ゼロのレコードを除外"
    )
    
    # パフォーマンス設定
    batch_size: int = Field(
        default=5000,
        description="バッチサイズ"
    )
    max_workers: int = Field(
        default=4,
        description="最大ワーカー数"
    )
    
    @field_serializer("period", "period_start", "period_end")
    def _serialize_periods(self, period: Optional[WolfPeriod]) -> Optional[Dict[str, Any]]:
        """WolfPeriodをシリアライズ"""
        return period.model_dump() if period else None


# よく使用されるクエリのプリセット
def create_l6_revenue_query(
    entity_ids: Optional[List[str]] = None,
    fiscal_years: Optional[List[int]] = None
) -> RBICSCompanyQueryParams:
    """L6売上セグメント用クエリを作成"""
    return RBICSCompanyQueryParams(
        entity_ids=entity_ids,
        fiscal_years=fiscal_years,
        segment_types=[SegmentType.REVENUE],
        rbics_levels=[RBICSLevel.L6],
        exclude_zero_revenue=True,
        require_segment_breakdown=True
    )


def create_structure_hierarchy_query(
    start_level: RBICSLevel = RBICSLevel.L1,
    end_level: RBICSLevel = RBICSLevel.L6
) -> RBICSStructureQueryParams:
    """RBICS階層構造用クエリを作成"""
    levels = []
    level_values = [RBICSLevel.L1, RBICSLevel.L2, RBICSLevel.L3, 
                   RBICSLevel.L4, RBICSLevel.L5, RBICSLevel.L6]
    
    start_idx = level_values.index(start_level)
    end_idx = level_values.index(end_level)
    
    for i in range(start_idx, end_idx + 1):
        levels.append(level_values[i])
    
    return RBICSStructureQueryParams(
        rbics_levels=levels,
        include_inactive=False
    )
