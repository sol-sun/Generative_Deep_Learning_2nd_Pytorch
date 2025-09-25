"""
FactSet Tradenames データプロバイダー型定義
========================================

FactSet Tradenames固有のデータレコード型定義。

使用例:
    from data_providers.sources.factset_tradenames.types import (
        FactSetTradenameRecord,
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


class FactSetTradenameRecord(BaseModel):
    """FactSet Tradename レコード（読み取り専用）。

    概要:
    - 企業の商品名・サービス名データを保持します。
    - 商品名の正規化と検索用のテキスト生成機能を含みます。

    特徴:
    - 商品名の正規化処理
    - 検索用テキストの自動生成
    - 期間管理によるデータの鮮度管理
    - RBICS分類との関連付け
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
    trade_id: constr(min_length=1, max_length=50) = Field(
        description="Trade ID",
        examples=["TRADE_001"]
    )
    factset_entity_id: constr(min_length=1, max_length=20) = Field(
        description="FactSet Entity ID",
        examples=["001C7F-E"]
    )
    product_id: Optional[float] = Field(
        default=None,
        description="Product ID",
        examples=[12345.0]
    )
    product_name: constr(min_length=1, max_length=500) = Field(
        description="Product Name",
        examples=["iPhone 15 Pro"]
    )
    l6_id: constr(min_length=1, max_length=20) = Field(
        description="RBICS L6 ID",
        examples=["123456"]
    )
    l6_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="RBICS L6 Name",
        examples=["Smartphones"]
    )

    # 期間情報
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start Date",
        examples=[datetime(2023, 1, 1)]
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End Date",
        examples=[datetime(2023, 12, 31)]
    )

    # フラグ
    multi_assign_flag: Optional[conint(ge=0, le=1)] = Field(
        default=None,
        description="Multi Assignment Flag",
        examples=[0, 1]
    )

    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )

    @field_validator("product_name", mode="before")
    @classmethod
    def _normalize_product_name(cls, v: str) -> str:
        """商品名を正規化（前後の空白除去）。"""
        if not isinstance(v, str):
            raise ValueError("商品名は文字列である必要があります")
        return v.strip()

    @field_validator("l6_id", mode="before")
    @classmethod
    def _normalize_l6_id(cls, v: str) -> str:
        """L6 IDを正規化（前後の空白除去）。"""
        if not isinstance(v, str):
            raise ValueError("L6 IDは文字列である必要があります")
        return v.strip()

    @computed_field
    @property
    def search_text(self) -> str:
        """検索用テキストの生成。"""
        if self.l6_name:
            return f"{self.l6_name} : {self.product_name}"
        return self.product_name

    @computed_field
    @property
    def is_active(self) -> bool:
        """アクティブな商品かどうかの判定。"""
        return self.end_date is None

    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()

    @field_serializer("start_date", "end_date")
    def _serialize_datetime(self, dt: Optional[datetime]) -> Optional[str]:
        """日時をシリアライズ。"""
        return dt.isoformat() if dt else None
