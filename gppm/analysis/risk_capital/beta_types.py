"""
ベータ計算関連型定義
===================

ベータ計算に使用する型定義とレコードモデル。

使用例:
    from gppm.analysis.risk_capital.beta_types import BetaRecord
"""

from __future__ import annotations

from datetime import datetime, timezone, date
from typing import Optional, Dict, Any
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    computed_field,
    field_serializer,
)
from pydantic.types import conint, constr, confloat

from wolf_period import WolfPeriod


class BetaRecord(BaseModel):
    """ベータ計算結果レコード（読み取り専用）。

    概要:
    - ベータ計算の結果（ベータ値、R²、標準誤差、観測数）を保持します。
    - 計算条件（期間、インデックス、証券）のメタデータを含みます。

    特徴:
    - 統計的有意性の評価指標を含む
    - 計算条件の完全な記録
    - エラーケースの適切な処理
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 証券情報
    security_id: constr(min_length=1, max_length=50) = Field(
        description="証券識別子",
        examples=["US0378331005", "B0YQ5F0"]
    )
    identifier_type: constr(min_length=1, max_length=10) = Field(
        description="識別子タイプ",
        examples=["isin", "sedol", "cusip"]
    )
    
    # インデックス情報
    index_code: constr(min_length=1, max_length=10) = Field(
        description="インデックスコード",
        examples=["@DP", "@DT", "JPN"]
    )
    index_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="インデックス名称",
        examples=["MSCI World Index", "MSCI Japan"]
    )
    
    # 計算期間
    as_of_date: date = Field(
        description="計算基準日",
        examples=[date(2023, 12, 31)]
    )
    lookback_periods: conint(ge=1, le=1200) = Field(
        description="遡及期間数",
        examples=[60, 120, 240]
    )
    
    # ベータ計算結果
    beta_value: Optional[confloat(ge=-10.0, le=10.0)] = Field(
        default=None,
        description="ベータ値",
        examples=[1.234, 0.876, -0.123]
    )
    r_squared: Optional[confloat(ge=0.0, le=1.0)] = Field(
        default=None,
        description="決定係数（R²）",
        examples=[0.456, 0.789, 0.234]
    )
    standard_error: Optional[confloat(ge=0.0)] = Field(
        default=None,
        description="標準誤差",
        examples=[0.123, 0.456, 0.789]
    )
    observations: Optional[conint(ge=1)] = Field(
        default=None,
        description="観測数",
        examples=[60, 120, 240]
    )
    
    # 計算設定
    frequency: str = Field(
        description="計算頻度",
        examples=["monthly", "weekly", "daily"]
    )
    levered: bool = Field(
        description="レバードベータフラグ",
        examples=[True, False]
    )
    beta_type: str = Field(
        description="ベータタイプ",
        examples=["global", "local"]
    )
    
    # エラー情報
    error_message: Optional[constr(min_length=1, max_length=500)] = Field(
        default=None,
        description="エラーメッセージ（計算失敗時）",
        examples=["Insufficient data", "Index data not available"]
    )
    
    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )
    
    @field_validator("identifier_type", mode="before")
    @classmethod
    def _normalize_identifier_type(cls, v: str) -> str:
        """識別子タイプを正規化（小文字化）。"""
        if not isinstance(v, str):
            raise ValueError("識別子タイプは文字列である必要があります")
        return v.strip().lower()
    
    @field_validator("frequency", mode="before")
    @classmethod
    def _normalize_frequency(cls, v: str) -> str:
        """計算頻度を正規化（小文字化）。"""
        if not isinstance(v, str):
            raise ValueError("計算頻度は文字列である必要があります")
        return v.strip().lower()
    
    @model_validator(mode="after")
    def _validate_beta_consistency(self) -> "BetaRecord":
        """ベータ計算結果の整合性を検証。"""
        # 成功時はベータ値が必要
        if self.error_message is None and self.beta_value is None:
            raise ValueError("計算成功時はベータ値が必要です")
        
        # 失敗時はエラーメッセージが必要
        if self.error_message is not None and self.beta_value is not None:
            raise ValueError("計算失敗時はベータ値を設定できません")
        
        return self
    
    @computed_field
    @property
    def is_successful(self) -> bool:
        """計算が成功したかどうか（遅延評価）。"""
        return self.error_message is None and self.beta_value is not None
    
    @computed_field
    @property
    def is_statistically_significant(self) -> bool:
        """統計的有意性（R² > 0.1 かつ観測数 >= 24）。"""
        if not self.is_successful:
            return False
        return (self.r_squared or 0) > 0.1 and (self.observations or 0) >= 24
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.retrieved_period)
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()

