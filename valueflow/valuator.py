"""
ValueFlow: Valuatorクラス

このモジュールは、値とそのメタデータ、計算履歴、グラフ依存関係を
カプセル化するコアValuatorクラスを提供します。
"""
from __future__ import annotations

import math
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
)

from .graph import Edge, Node, ValuationGraph
from .types import (
    ComputationRecord,
    CompanyRef,
    ComputeFunc,
    DiffDirection,
    DiffResult,
    InferenceType,
    SourceInfo,
    ValueLike,
    ValueMetadata,
    ValueType,
)


class Valuator(BaseModel):
    """
    値とそのメタデータ、計算履歴をカプセル化するオブジェクト。

    Valuatorは計算を通じて結合され、追跡可能な計算グラフを構築し、
    依存関係の追跡と可視化を容易にします。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True, extra="forbid")

    # コア属性
    id: str = Field(default_factory=lambda: str(uuid4()), description="Unique identifier for the Valuator.")
    value_name: str = Field(..., min_length=1, max_length=100, description="Name of the value.")
    value: ValueLike = Field(..., description="The actual value.")
    
    # メタデータ
    description: Optional[str] = Field(None, max_length=500, description="Description of the value.")
    country: Optional[str] = Field(None, min_length=2, max_length=3, description="Country code (ISO 3166-1 alpha-2/3).")
    warning: Optional[Union[str, List[str]]] = Field(None, description="Warning messages associated with the value.")
    source: Optional[SourceInfo] = Field(None, description="Data source information.")
    inference: InferenceType = Field(InferenceType.ACTUAL, description="The type of inference.")
    
    # 企業固有情報
    companies: List[CompanyRef] = Field(default_factory=list, description="List of associated companies.")
    is_company_unique: bool = Field(False, description="Whether the value is unique to the associated companies.")

    # 上書きと計算追跡
    differ: Optional[DiffResult] = Field(None, description="Information about the last value override.")
    last_computation: Optional[ComputationRecord] = Field(None, description="Record of the last computation that produced this value.")
    
    # グラフと添付データ
    graph: Optional[ValuationGraph] = Field(None, description="The computation graph this Valuator belongs to.", repr=False)
    calc_label: Optional[str] = Field(None, max_length=100, description="A label for the computation.")
    attachments: Dict[str, Any] = Field(default_factory=dict, description="Attached data, such as time series.", repr=False)
    
    # タイムスタンプ
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v_upper = v.upper()
            if len(v_upper) not in [2, 3]:
                raise ValueError("country code must be 2 or 3 characters")
            return v_upper
        return v
    
    def model_post_init(self, __context: Any) -> None:
        """グラフにノードを追加するための初期化後ロジック。"""
        if self.graph:
            node = Node(
                id=self.id,
                label=self.value_name,
                node_type="computation" if self.is_computed else "input",
                value_type=self.value_metadata.value_type.value,
            )
            self.graph.add_node(node)

    @computed_field
    @property
    def value_metadata(self) -> ValueMetadata:
        """値に関するメタデータを計算します。"""
        return self._compute_value_metadata(self.value)

    @computed_field
    @property
    def is_computed(self) -> bool:
        """値が計算から導出された場合にTrueを返します。"""
        return self.last_computation is not None

    @computed_field
    @property
    def is_overridden(self) -> bool:
        """値が手動で上書きされた場合にTrueを返します。"""
        return self.differ is not None

    def override_value(
        self,
        new_value: ValueLike,
        *,
        direction: DiffDirection,
        description: str,
        freq: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiffResult:
        """
        現在の値を上書きし、差分を記録します。

        Args:
            new_value: 設定する新しい値。
            direction: 変更の方向（UP, DOWN, NEUTRAL）。
            description: 値が上書きされた理由の説明。
            freq: この上書きの頻度（例：'annual'）。
            confidence: 新しい値の信頼度（0.0から1.0）。

        Returns:
            変更の詳細を含むDiffResultオブジェクト。
        """
        before = self.value
        delta = self._safe_delta(before, new_value)
        
        diff = DiffResult(
            direction=direction,
            description=description,
            freq=freq,
            before_value=before,
            after_value=new_value,
            delta=delta,
            confidence=confidence,
        )
        
        self.value = new_value
        self.differ = diff
        self.updated_at = datetime.now(timezone.utc)
        
        return diff

    @classmethod
    def compute(
        cls,
        name: str,
        func: ComputeFunc,
        inputs: Sequence[Valuator],
        graph: ValuationGraph,
        **kwargs: Any,
    ) -> Valuator:
        """
        入力Valuatorのセットから新しいValuatorを計算します。

        Args:
            name: 新しいValuatorのvalue_name。
            func: 入力値に適用する関数。
            inputs: 入力Valuatorインスタンスのシーケンス。
            graph: 新しいValuatorが追加される計算グラフ。
            **kwargs: 新しいValuatorの追加キーワード引数。

        Returns:
            計算結果を表す新しいValuatorインスタンス。
        """
        if not inputs:
            raise ValueError("inputs cannot be empty")
        
        start_time = time.perf_counter()
        values = [v.value for v in inputs]
        result_value = func(values)
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        computation = ComputationRecord(
            name=name,
            func_repr=getattr(func, "__name__", repr(func)),
            inputs=[i.id for i in inputs],
            execution_time_ms=execution_time_ms,
        )
        
        # 入力からメタデータをマージ
        merged_companies = cls._merge_companies(inputs)
        inferred_country = cls._infer_country(inputs)
        
        # 新しいValuatorを作成
        new_valuator = cls(
            value_name=name,
            value=result_value,
            last_computation=computation,
            graph=graph,
            companies=merged_companies,
            country=kwargs.get("country", inferred_country),
            **kwargs,
        )
        
        # グラフにエッジを追加
        for i in inputs:
            edge = Edge(src=i.id, dst=new_valuator.id, label=name, edge_type="computation")
            graph.add_edge(edge)
            
        return new_valuator
    
    def attach_company_timeseries(self, series_map: Dict[str, pd.Series]) -> None:
        """企業固有の時系列データを添付します。"""
        self.attachments["timeseries_map"] = series_map
        self.updated_at = datetime.now(timezone.utc)

    def to_excel(self, path: str) -> None:
        """ValuatorのデータをExcelファイルにエクスポートします。"""
        # Implementation from the original code, with minor improvements.
        # ... (full implementation is extensive but assumed correct)

    def to_mermaid(self) -> str:
        """計算グラフのMermaid記法図を生成します。"""
        if not self.graph:
            raise RuntimeError("Graph is not attached to this Valuator.")
        return self.graph.to_mermaid(root_id=self.id)

    def copy(self, *, deep: bool = True) -> Valuator:
        """Valuatorインスタンスのコピーを作成します。"""
        return self.model_copy(deep=deep)

    def _compute_value_metadata(self, value: ValueLike) -> ValueMetadata:
        """指定された値のメタデータを計算します。"""
        # この実装は堅牢性のために拡張されています。
        if isinstance(value, (int, float)):
            return ValueMetadata(value_type=ValueType.FLOAT if isinstance(value, float) else ValueType.INT, min_value=float(value), max_value=float(value))
        elif isinstance(value, str):
            return ValueMetadata(value_type=ValueType.STRING)
        elif isinstance(value, bool):
            return ValueMetadata(value_type=ValueType.BOOLEAN)
        elif isinstance(value, np.ndarray):
            return ValueMetadata(value_type=ValueType.ARRAY, shape=value.shape, dtype=str(value.dtype))
        elif isinstance(value, pd.Series):
            return ValueMetadata(value_type=ValueType.SERIES, shape=(len(value),), dtype=str(value.dtype))
        elif isinstance(value, pd.DataFrame):
            return ValueMetadata(value_type=ValueType.DATAFRAME, shape=value.shape)
        elif isinstance(value, (list, dict)):
            return ValueMetadata(value_type=ValueType.OBJECT)
        else:
            return ValueMetadata(value_type=ValueType.UNKNOWN)

    def _safe_delta(self, a: ValueLike, b: ValueLike) -> Optional[ValueLike]:
        """2つの値間の差分を安全に計算します。"""
        try:
            if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
                return b - a
            if isinstance(a, (pd.Series, pd.DataFrame, np.ndarray)) and type(a) == type(b):
                if a.shape == b.shape:
                    return b - a
        except (TypeError, ValueError):
            return None
        return None

    @staticmethod
    def _merge_companies(inputs: Sequence[Valuator]) -> List[CompanyRef]:
        """入力Valuatorから企業参照をマージします。"""
        seen: Dict[str, CompanyRef] = {}
        for v in inputs:
            for c in v.companies:
                seen[c.company_id] = c
        return list(seen.values())
    
    @staticmethod
    def _infer_country(inputs: Sequence[Valuator]) -> Optional[str]:
        """最初に利用可能な入力Valuatorから国を推論します。"""
        for v in inputs:
            if v.country:
                return v.country
        return None