"""
ValueFlow: 型定義とバリデーション

このモジュールは、ValueFlowシステムで使用されるすべての型定義と
バリデーションロジックを提供します。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Literal, Optional, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    ConfigDict,
    TypeAdapter,
)


class DiffDirection(str, Enum):
    """差分の方向性を表す列挙型"""
    
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"
    
    def __str__(self) -> str:
        return self.value


class InferenceType(str, Enum):
    """推論の種類を表す列挙型"""
    
    ESTIMATED = "estimated"  # モデル推計値
    FORECAST = "forecast"    # 予想
    ACTUAL = "actual"        # 実績
    
    def __str__(self) -> str:
        return self.value


class ValueType(str, Enum):
    """値の型を表す列挙型"""
    
    FLOAT = "float"
    INT = "int"
    SERIES = "series"
    DATAFRAME = "dataframe"
    ARRAY = "array"
    STRING = "string"
    BOOLEAN = "boolean"
    
    def __str__(self) -> str:
        return self.value


class SourceInfo(BaseModel):
    """データソース情報"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    dataset: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="データセット名",
        examples=["factset_fundamentals", "bloomberg_prices", "custom_calculation"]
    )
    provider: Optional[str] = Field(
        None,
        max_length=50,
        description="データプロバイダー名",
        examples=["FactSet", "Bloomberg", "Reuters", "Custom"]
    )
    fields: Optional[list[str]] = Field(
        None,
        description="参照フィールド名の一覧",
        examples=[["revenue", "operating_income"], ["price", "volume"]]
    )
    note: Optional[str] = Field(
        None,
        max_length=500,
        description="補足情報",
        examples=["四半期データ", "調整済み実績値", "予想値（コンセンサス）"]
    )
    version: Optional[str] = Field(
        None,
        max_length=20,
        description="データセットのバージョン",
        examples=["v1.0", "2024Q1", "latest"]
    )
    
    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v: Optional[list[str]]) -> Optional[list[str]]:
        if v is not None:
            if len(v) == 0:
                raise ValueError("fields list cannot be empty")
            for field in v:
                if not isinstance(field, str) or len(field.strip()) == 0:
                    raise ValueError("all fields must be non-empty strings")
        return v


class CompanyRef(BaseModel):
    """企業参照情報"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    company_id: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="企業の内部一意IDやFSYM_ID等",
        examples=["FSYM123456", "BBG001234", "CUSTOM_001"]
    )
    name: Optional[str] = Field(
        None,
        max_length=200,
        description="企業名",
        examples=["Apple Inc.", "トヨタ自動車株式会社", "Microsoft Corporation"]
    )
    ticker: Optional[str] = Field(
        None,
        max_length=20,
        description="ティッカー",
        examples=["AAPL", "7203.T", "MSFT"]
    )
    weight: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="ポートフォリオウェイト等",
        examples=[0.05, 0.1, 0.25]
    )
    country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="国コード（ISO 3166-1 alpha-2/3）",
        examples=["US", "JP", "GBR", "DEU"]
    )
    sector: Optional[str] = Field(
        None,
        max_length=100,
        description="セクター",
        examples=["Technology", "Automotive", "Healthcare"]
    )
    market_cap: Optional[float] = Field(
        None,
        ge=0.0,
        description="時価総額（百万円）",
        examples=[1000000.0, 5000000.0]
    )
    
    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.upper()
            if len(v) not in [2, 3]:
                raise ValueError("country code must be 2 or 3 characters")
        return v


class DiffResult(BaseModel):
    """上書き（差し替え）時の差分情報"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    direction: DiffDirection = Field(
        ...,
        description="差分の方向性"
    )
    freq: Optional[str] = Field(
        None,
        max_length=20,
        description="差分の頻度",
        examples=["daily", "monthly", "quarterly", "annual", "one-time"]
    )
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="差分の説明",
        examples=["モデル推計値に差し替え", "四半期実績反映", "予想値更新"]
    )
    before_value: Any = Field(
        ...,
        description="上書き前の値"
    )
    after_value: Any = Field(
        ...,
        description="上書き後の値"
    )
    delta: Optional[Any] = Field(
        None,
        description="差分（型は値に依存）"
    )
    at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="差分生成時刻"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="差分の信頼度",
        examples=[0.95, 0.8, 0.6]
    )
    
    @model_validator(mode="after")
    def validate_values(self) -> "DiffResult":
        # 値の型が一致することを確認
        before_type = type(self.before_value)
        after_type = type(self.after_value)
        
        if before_type != after_type:
            # 数値型の場合は互換性をチェック
            if not (
                isinstance(self.before_value, (int, float, np.number)) and
                isinstance(self.after_value, (int, float, np.number))
            ):
                raise ValueError(
                    f"Value types must match: before={before_type}, after={after_type}"
                )
        
        return self


class ComputationRecord(BaseModel):
    """計算の記録メタデータ"""
    
    model_config = ConfigDict(
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="計算名やラベル",
        examples=["roic", "invested_capital", "wacc_calculation"]
    )
    func_repr: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="使用関数・式の簡易表現",
        examples=["lambda vals: vals[0] / vals[1]", "roic_calculation", "np.mean"]
    )
    inputs: list[str] = Field(
        default_factory=list,
        description="入力ノードIDの一覧"
    )
    note: Optional[str] = Field(
        None,
        max_length=500,
        description="計算に関する補足",
        examples=["四半期データを使用", "調整済み実績値", "予想値ベース"]
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="計算実行時刻"
    )
    execution_time_ms: Optional[float] = Field(
        None,
        ge=0.0,
        description="実行時間（ミリ秒）",
        examples=[1.5, 10.2, 100.0]
    )
    memory_usage_mb: Optional[float] = Field(
        None,
        ge=0.0,
        description="メモリ使用量（MB）",
        examples=[5.2, 50.0, 200.0]
    )
    
    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("inputs list cannot be empty")
        for input_id in v:
            if not isinstance(input_id, str) or len(input_id.strip()) == 0:
                raise ValueError("all input IDs must be non-empty strings")
        return v


class ValueMetadata(BaseModel):
    """値のメタデータ"""
    
    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid"
    )
    
    value_type: ValueType = Field(
        ...,
        description="値の型"
    )
    shape: Optional[tuple[int, ...]] = Field(
        None,
        description="値の形状（配列・DataFrameの場合）",
        examples=[(100,), (10, 5), (1000, 3, 2)]
    )
    dtype: Optional[str] = Field(
        None,
        description="データ型",
        examples=["float64", "int32", "object", "datetime64[ns]"]
    )
    min_value: Optional[float] = Field(
        None,
        description="最小値（数値の場合）"
    )
    max_value: Optional[float] = Field(
        None,
        description="最大値（数値の場合）"
    )
    mean_value: Optional[float] = Field(
        None,
        description="平均値（数値の場合）"
    )
    std_value: Optional[float] = Field(
        None,
        description="標準偏差（数値の場合）"
    )
    null_count: Optional[int] = Field(
        None,
        ge=0,
        description="null値の数"
    )
    unique_count: Optional[int] = Field(
        None,
        ge=0,
        description="ユニーク値の数"
    )


# 型エイリアス
ValueLike = Union[
    int, float, str, bool, 
    np.ndarray, pd.Series, pd.DataFrame,
    list[Any], dict[str, Any]
]

ComputeFunc = Callable[[list[ValueLike]], ValueLike]

# TypeAdapter for serialization
ValueLikeAdapter = TypeAdapter(ValueLike)