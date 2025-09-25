"""
FactSet Tradenames プロバイダー用クエリパラメーター
==============================================

FactSet Tradenames固有のクエリパラメーター定義。
複雑なFactSet Tradenames検索条件の型安全な定義とバリデーション・変換ロジック。

使用例:
    from data_providers.sources.factset_tradenames.query_params import FactSetTradenameQueryParams

    # クエリパラメータの作成
    params = FactSetTradenameQueryParams(
        factset_entity_ids=["001C7F-E", "002C7F-E"],
        l6_ids=["123456", "789012"],
        active_only=True,
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


class FactSetTradenameQueryParams(BaseModel):
    """FactSet Tradename クエリパラメータ。

    目的:
    - Tradenameデータ取得の条件を安全に指定
    - 企業・商品・期間によるフィルタリング
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
        description="特定のFactSet Entity IDリストでフィルタ",
        examples=[["001C7F-E", "002C7F-E"]]
    )

    # 商品フィルタ
    product_names: Optional[List[constr(min_length=1, max_length=500)]] = Field(
        default=None,
        description="特定の商品名リストでフィルタ",
        examples=[["iPhone", "iPad", "MacBook"]]
    )
    l6_ids: Optional[List[constr(min_length=1, max_length=20)]] = Field(
        default=None,
        description="特定のRBICS L6 IDリストでフィルタ",
        examples=[["123456", "789012"]]
    )

    # 期間フィルタ
    active_only: bool = Field(
        default=True,
        description="アクティブな商品のみを取得（end_date IS NULL）"
    )

    # データ品質フィルタ
    min_product_name_length: Optional[conint(ge=1, le=500)] = Field(
        default=None,
        description="最小商品名長",
        examples=[3]
    )
    exclude_empty_names: bool = Field(
        default=True,
        description="空の商品名を除外"
    )

    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    max_items_per_entity: Optional[conint(ge=1, le=1000)] = Field(
        default=100,
        description="企業あたりの最大商品数",
        examples=[100]
    )

    @field_validator("factset_entity_ids", "product_names", "l6_ids", mode="before")
    @classmethod
    def _normalize_string_lists(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """文字列リストを正規化（前後の空白除去）。"""
        if v is None:
            return None
        
        if not isinstance(v, list):
            raise ValueError("パラメータはリストである必要があります")
        
        normalized = []
        for item in v:
            if isinstance(item, str):
                normalized.append(item.strip())
        
        return normalized if normalized else None

    @field_validator("min_product_name_length")
    @classmethod
    def _validate_min_length(cls, v: Optional[int]) -> Optional[int]:
        """最小長の範囲チェック。"""
        if v is not None and (v < 1 or v > 500):
            raise ValueError("最小商品名長は1から500の範囲で指定してください")
        return v
