"""
ValueFlow: 型定義とバリデーション

このモジュールは、ValueFlowシステムで使用されるすべての型定義と
バリデーションロジックを提供し、堅牢なデータ契約のためにPydantic V2を活用します。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    field_validator,
    model_validator,
)


class DiffDirection(str, Enum):
    """差分の方向の列挙型。"""
    UP = "up"
    DOWN = "down"
    NEUTRAL = "neutral"

    def __str__(self) -> str:
        return self.value


class InferenceType(str, Enum):
    """推論のタイプの列挙型。"""
    ESTIMATED = "estimated"
    FORECAST = "forecast"
    ACTUAL = "actual"

    def __str__(self) -> str:
        return self.value


class ValueType(str, Enum):
    """値のタイプの列挙型。"""
    FLOAT = "float"
    INT = "int"
    STRING = "string"
    BOOLEAN = "boolean"
    ARRAY = "array"
    SERIES = "series"
    DATAFRAME = "dataframe"
    OBJECT = "object"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


class SourceInfo(BaseModel):
    """ソース情報のデータモデル。"""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    dataset: str = Field(
        ..., min_length=1, max_length=100,
        description="Name of the dataset.",
        examples=["factset_fundamentals", "bloomberg_prices"]
    )
    provider: Optional[str] = Field(
        None, max_length=50,
        description="Name of the data provider.",
        examples=["FactSet", "Bloomberg", "Custom"]
    )
    fields: Optional[List[str]] = Field(
        None, description="List of referenced field names.",
        examples=[["revenue", "operating_income"]]
    )
    note: Optional[str] = Field(
        None, max_length=500,
        description="Additional notes.",
        examples=["Quarterly data", "Adjusted actuals"]
    )
    version: Optional[str] = Field(
        None, max_length=20,
        description="Version of the dataset.",
        examples=["v1.0", "2024Q1"]
    )

    @field_validator("fields")
    @classmethod
    def validate_fields(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is not None:
            if not v:
                raise ValueError("fields list cannot be empty")
            if not all(isinstance(f, str) and f.strip() for f in v):
                raise ValueError("all fields must be non-empty strings")
        return v


class CompanyRef(BaseModel):
    """企業参照のデータモデル。"""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    company_id: str = Field(
        ..., min_length=1, max_length=50,
        description="Unique internal ID for the company (e.g., FSYM_ID).",
        examples=["FSYM123456", "BBG001234"]
    )
    name: Optional[str] = Field(None, max_length=200, description="Company name.")
    ticker: Optional[str] = Field(None, max_length=20, description="Ticker symbol.")
    weight: Optional[float] = Field(None, ge=0.0, le=1.0, description="Portfolio weight.")
    country: Optional[str] = Field(None, min_length=2, max_length=3, description="Country code (ISO 3166-1 alpha-2/3).")
    sector: Optional[str] = Field(None, max_length=100, description="Sector.")
    market_cap: Optional[float] = Field(None, ge=0.0, description="Market capitalization (in millions).")

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_upper = v.upper()
            if len(v_upper) not in [2, 3]:
                raise ValueError("country code must be 2 or 3 characters")
            return v_upper
        return v


class DiffResult(BaseModel):
    """値の上書き結果のデータモデル。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")

    direction: DiffDirection = Field(..., description="Direction of the difference.")
    freq: Optional[str] = Field(None, max_length=20, description="Frequency of the difference.", examples=["annual", "quarterly"])
    description: Optional[str] = Field(None, max_length=500, description="Description of the difference.")
    before_value: Any = Field(..., description="Value before the override.")
    after_value: Any = Field(..., description="Value after the override.")
    delta: Optional[Any] = Field(None, description="The calculated difference.")
    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the difference generation.")
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0, description="Confidence level of the difference.")


class ComputationRecord(BaseModel):
    """計算メタデータのデータモデル。"""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")

    name: str = Field(..., min_length=1, max_length=100, description="Name or label of the computation.")
    func_repr: str = Field(..., min_length=1, max_length=200, description="Simple representation of the function/expression used.")
    inputs: List[str] = Field(..., description="List of input Valuator IDs.")
    note: Optional[str] = Field(None, max_length=500, description="Notes about the computation.")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the computation.")
    execution_time_ms: Optional[float] = Field(None, ge=0.0, description="Execution time in milliseconds.")
    memory_usage_mb: Optional[float] = Field(None, ge=0.0, description="Memory usage in megabytes.")

    @field_validator("inputs")
    @classmethod
    def validate_inputs(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("inputs list cannot be empty")
        if not all(isinstance(i, str) and i.strip() for i in v):
            raise ValueError("all input IDs must be non-empty strings")
        return v


class ValueMetadata(BaseModel):
    """値メタデータのデータモデル。"""
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")

    value_type: ValueType = Field(..., description="Type of the value.")
    shape: Optional[Tuple[int, ...]] = Field(None, description="Shape of the value (for arrays/DataFrames).")
    dtype: Optional[str] = Field(None, description="Data type of the value elements.")
    min_value: Optional[float] = Field(None, description="Minimum value (for numeric types).")
    max_value: Optional[float] = Field(None, description="Maximum value (for numeric types).")
    mean_value: Optional[float] = Field(None, description="Mean value (for numeric types).")
    std_value: Optional[float] = Field(None, description="Standard deviation (for numeric types).")
    null_count: Optional[int] = Field(None, ge=0, description="Count of null values.")
    unique_count: Optional[int] = Field(None, ge=0, description="Count of unique values.")


# Valuatorが保持できる値の型エイリアス。
ValueLike = Union[int, float, str, bool, np.ndarray, pd.Series, pd.DataFrame, List[Any], Dict[str, Any]]

# 計算関数の型エイリアス。
ComputeFunc = Callable[[List[ValueLike]], ValueLike]

# ValueLike型の堅牢なシリアライゼーション/デシリアライゼーション用のTypeAdapter。
ValueLikeAdapter = TypeAdapter(ValueLike)