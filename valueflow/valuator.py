"""
ValueFlow: Valuator - 値とメタデータの管理

このモジュールは、値とメタデータ、計算履歴を持つValuatorクラスを提供します。
他のValuatorとの計算から計算グラフを構築し、依存関係の可視化/追跡が可能です。
"""

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Union
from uuid import UUID, uuid4

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
    computed_field,
    ConfigDict,
    TypeAdapter,
)

from .graph import ValuationGraph
from .types import (
    ComputeFunc,
    ComputationRecord,
    CompanyRef,
    DiffDirection,
    DiffResult,
    InferenceType,
    SourceInfo,
    ValueLike,
    ValueMetadata,
    ValueType,
    ValueLikeAdapter,
)


class Valuator(BaseModel):
    """値とメタデータ、計算履歴を持つValuator"""
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
        extra="forbid"
    )
    
    # 基本情報
    id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Valuatorの一意ID"
    )
    value_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="値の名前",
        examples=["operating_profit", "invested_capital", "roic", "wacc"]
    )
    value: ValueLike = Field(
        ...,
        description="実際の値"
    )
    
    # メタデータ
    description: Optional[str] = Field(
        None,
        max_length=500,
        description="値の説明",
        examples=["営業利益", "投下資本", "ROIC = 営業利益 / 投下資本"]
    )
    country: Optional[str] = Field(
        None,
        min_length=2,
        max_length=3,
        description="国コード（ISO 3166-1 alpha-2/3）",
        examples=["US", "JP", "GBR"]
    )
    warning: Optional[Union[str, List[str]]] = Field(
        None,
        description="警告メッセージ",
        examples=["データが古い", ["四半期データ", "調整済み実績値"]]
    )
    source: Optional[SourceInfo] = Field(
        None,
        description="データソース情報"
    )
    inference: InferenceType = Field(
        InferenceType.ACTUAL,
        description="推論の種類"
    )
    is_company_unique: Optional[bool] = Field(
        None,
        description="企業固有の値かどうか"
    )
    
    # 差分情報
    differ: Optional[DiffResult] = None
    
    # 企業情報
    companies: List[CompanyRef] = Field(
        default_factory=list,
        description="関連する企業のリスト"
    )
    
    # 計算グラフ
    graph: Optional[ValuationGraph] = None
    calc_label: Optional[str] = Field(
        None,
        max_length=100,
        description="計算ラベル",
        examples=["roic_calculation", "wacc_estimation"]
    )
    
    # 添付データ
    attachments: Dict[str, Any] = Field(
        default_factory=dict,
        description="添付データ（時系列データなど）"
    )
    
    # 計算履歴
    last_computation: Optional[ComputationRecord] = None
    
    # 作成・更新時刻
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="作成時刻"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="最終更新時刻"
    )
    
    @field_validator("warning")
    @classmethod
    def validate_warning(cls, v: Optional[Union[str, List[str]]]) -> Optional[Union[str, List[str]]]:
        if isinstance(v, list):
            if len(v) == 0:
                raise ValueError("warning list cannot be empty")
            for warning in v:
                if not isinstance(warning, str) or len(warning.strip()) == 0:
                    raise ValueError("all warnings must be non-empty strings")
        return v
    
    @field_validator("country")
    @classmethod
    def validate_country(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            v = v.upper()
            if len(v) not in [2, 3]:
                raise ValueError("country code must be 2 or 3 characters")
        return v
    
    @computed_field
    @property
    def value_metadata(self) -> ValueMetadata:
        """値のメタデータを計算"""
        return self._compute_value_metadata(self.value)
    
    @computed_field
    @property
    def is_computed(self) -> bool:
        """計算された値かどうか"""
        return self.last_computation is not None
    
    @computed_field
    @property
    def is_overridden(self) -> bool:
        """上書きされた値かどうか"""
        return self.differ is not None
    
    def model_post_init(self, __context: Any) -> None:
        """初期化後の処理"""
        if self.graph is not None:
            try:
                self.graph.add_node(
                    self.id,
                    label=self.value_name,
                    node_type="input" if not self.is_computed else "computation",
                    value_type=self.value_metadata.value_type.value
                )
            except Exception:
                pass
    
    def override_value(
        self,
        new_value: ValueLike,
        *,
        direction: DiffDirection = DiffDirection.NEUTRAL,
        freq: Optional[str] = None,
        description: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> DiffResult:
        """値を上書きし、差分情報を生成"""
        before = self.value
        delta = self._safe_delta(before, new_value)
        
        diff = DiffResult(
            direction=direction,
            freq=freq,
            description=description,
            before_value=before,
            after_value=new_value,
            delta=delta,
            confidence=confidence,
        )
        
        self.value = new_value
        self.differ = diff
        self.updated_at = datetime.now(timezone.utc)
        
        # グラフに上書きノードを追加
        if self.graph is not None:
            override_id = f"{self.id}_override_{int(time.time())}"
            self.graph.add_node(
                override_id,
                label=f"{self.value_name}_override",
                node_type="override",
                value_type=self.value_metadata.value_type.value
            )
            self.graph.add_edge(
                override_id,
                self.id,
                label="override",
                edge_type="override"
            )
        
        return diff
    
    @classmethod
    def compute(
        cls,
        *,
        name: str,
        func: ComputeFunc,
        inputs: Sequence[Valuator],
        graph: Optional[ValuationGraph] = None,
        description: Optional[str] = None,
        inference: InferenceType = InferenceType.ACTUAL,
        source: Optional[SourceInfo] = None,
        calc_label: Optional[str] = None,
        execution_time_ms: Optional[float] = None,
        memory_usage_mb: Optional[float] = None,
    ) -> Valuator:
        """複数のValuatorから新しいValuatorを計算"""
        if len(inputs) == 0:
            raise ValueError("inputs cannot be empty")
        
        start_time = time.time()
        start_memory = self._get_memory_usage() if hasattr(cls, '_get_memory_usage') else None
        
        try:
            values = [v.value for v in inputs]
            out_value = func(values)
        except Exception as e:
            raise RuntimeError(f"Computation failed: {e}") from e
        
        end_time = time.time()
        end_memory = self._get_memory_usage() if hasattr(cls, '_get_memory_usage') else None
        
        # 実行時間とメモリ使用量を計算
        if execution_time_ms is None:
            execution_time_ms = (end_time - start_time) * 1000
        
        if memory_usage_mb is None and start_memory is not None and end_memory is not None:
            memory_usage_mb = max(0, end_memory - start_memory)
        
        v = cls(
            value_name=name,
            value=out_value,
            description=description,
            inference=inference,
            source=source,
            calc_label=calc_label or name,
            graph=graph,
        )
        
        # 計算記録を作成
        func_repr = getattr(func, "__name__", repr(func))
        v.last_computation = ComputationRecord(
            name=name,
            func_repr=func_repr,
            inputs=[i.id for i in inputs],
            execution_time_ms=execution_time_ms,
            memory_usage_mb=memory_usage_mb,
        )
        
        # 企業情報をマージ
        v.companies = cls._merge_companies(inputs)
        v.country = cls._infer_country(inputs)
        
        # グラフにノードとエッジを追加
        if graph is not None:
            graph.add_node(
                v.id,
                label=name,
                node_type="computation",
                value_type=v.value_metadata.value_type.value
            )
            
            for i in inputs:
                if not graph.has_node(i.id):
                    graph.add_node(
                        i.id,
                        label=i.value_name,
                        node_type="input" if not i.is_computed else "computation",
                        value_type=i.value_metadata.value_type.value
                    )
                graph.add_edge(
                    i.id,
                    v.id,
                    label=name,
                    edge_type="computation"
                )
        
        return v
    
    def attach_company_timeseries(
        self,
        series_map: Dict[Union[str, CompanyRef], pd.Series],
        *,
        key: str = "timeseries_map"
    ) -> None:
        """企業の時系列データを添付"""
        normalized: Dict[str, pd.Series] = {}
        for k, s in series_map.items():
            if isinstance(k, CompanyRef):
                cid = k.company_id
            else:
                cid = str(k)
            normalized[cid] = s
        
        self.attachments[key] = normalized
        self.updated_at = datetime.now(timezone.utc)
    
    def get_company_timeseries(
        self,
        *,
        key: str = "timeseries_map"
    ) -> Optional[Dict[str, pd.Series]]:
        """企業の時系列データを取得"""
        v = self.attachments.get(key)
        if isinstance(v, dict):
            return {
                str(k): s for k, s in v.items()
                if isinstance(s, pd.Series)
            }
        return None
    
    def attach_data(
        self,
        key: str,
        data: Any,
        *,
        description: Optional[str] = None
    ) -> None:
        """任意のデータを添付"""
        self.attachments[key] = {
            "data": data,
            "description": description,
            "attached_at": datetime.now(timezone.utc)
        }
        self.updated_at = datetime.now(timezone.utc)
    
    def get_attached_data(self, key: str) -> Optional[Any]:
        """添付データを取得"""
        attachment = self.attachments.get(key)
        if isinstance(attachment, dict) and "data" in attachment:
            return attachment["data"]
        return attachment
    
    def to_mermaid(self) -> str:
        """Mermaid記法でグラフを出力"""
        if self.graph is None:
            raise RuntimeError("graph is not attached to this Valuator")
        return self.graph.to_mermaid(root_id=self.id)
    
    def to_networkx(self) -> Any:
        """NetworkXグラフを出力"""
        if self.graph is None:
            raise RuntimeError("graph is not attached to this Valuator")
        return self.graph.to_networkx()
    
    def to_excel(self, path: str) -> None:
        """Excelファイルに出力"""
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # メタデータシート
            meta_data = {
                "項目": [
                    "ID", "値名", "説明", "国", "推論種別", "計算ラベル",
                    "企業固有", "作成日時", "更新日時", "値の型", "値の形状"
                ],
                "値": [
                    self.id, self.value_name, self.description, self.country,
                    self.inference.value, self.calc_label, self.is_company_unique,
                    self.created_at.isoformat(), self.updated_at.isoformat(),
                    self.value_metadata.value_type.value,
                    str(self.value_metadata.shape) if self.value_metadata.shape else "N/A"
                ]
            }
            
            if self.differ:
                meta_data["項目"].extend([
                    "差分方向", "差分説明", "上書き前値", "上書き後値", "差分値"
                ])
                meta_data["値"].extend([
                    self.differ.direction.value, self.differ.description,
                    str(self.differ.before_value), str(self.differ.after_value),
                    str(self.differ.delta) if self.differ.delta is not None else "N/A"
                ])
            
            pd.DataFrame(meta_data).to_excel(
                writer, sheet_name="metadata", index=False
            )
            
            # 値のシート
            if isinstance(self.value, pd.DataFrame):
                self.value.to_excel(writer, sheet_name="value")
            elif isinstance(self.value, pd.Series):
                self.value.to_frame("value").to_excel(writer, sheet_name="value")
            else:
                value_data = pd.DataFrame({
                    "value_name": [self.value_name],
                    "value": [self.value],
                    "type": [type(self.value).__name__]
                })
                value_data.to_excel(writer, sheet_name="value", index=False)
            
            # 企業情報シート
            if self.companies:
                company_data = []
                for company in self.companies:
                    company_data.append({
                        "company_id": company.company_id,
                        "name": company.name,
                        "ticker": company.ticker,
                        "weight": company.weight,
                        "country": company.country,
                        "sector": company.sector,
                        "market_cap": company.market_cap
                    })
                pd.DataFrame(company_data).to_excel(
                    writer, sheet_name="companies", index=False
                )
            
            # 時系列データシート
            ts_map = self.get_company_timeseries()
            if ts_map:
                for cid, series in ts_map.items():
                    safe_name = f"ts_{str(cid)[:28]}"
                    series.to_frame(str(cid)).to_excel(
                        writer, sheet_name=safe_name
                    )
            
            # 計算履歴シート
            if self.last_computation:
                comp_data = pd.DataFrame({
                    "項目": [
                        "計算名", "関数", "入力ノード", "実行時間(ms)", "メモリ使用量(MB)",
                        "実行時刻", "備考"
                    ],
                    "値": [
                        self.last_computation.name,
                        self.last_computation.func_repr,
                        ", ".join(self.last_computation.inputs),
                        self.last_computation.execution_time_ms,
                        self.last_computation.memory_usage_mb,
                        self.last_computation.timestamp.isoformat(),
                        self.last_computation.note
                    ]
                })
                comp_data.to_excel(writer, sheet_name="computation", index=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        return self.model_dump()
    
    def to_json(self) -> str:
        """JSON形式に変換"""
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Valuator:
        """辞書から復元"""
        return cls.model_validate(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> Valuator:
        """JSONから復元"""
        return cls.model_validate_json(json_str)
    
    def _compute_value_metadata(self, value: ValueLike) -> ValueMetadata:
        """値のメタデータを計算"""
        if isinstance(value, (int, float)):
            return ValueMetadata(
                value_type=ValueType.FLOAT if isinstance(value, float) else ValueType.INT,
                min_value=float(value),
                max_value=float(value),
                mean_value=float(value),
                std_value=0.0,
                null_count=0,
                unique_count=1
            )
        elif isinstance(value, str):
            return ValueMetadata(
                value_type=ValueType.STRING,
                null_count=0 if value else 1,
                unique_count=1
            )
        elif isinstance(value, bool):
            return ValueMetadata(
                value_type=ValueType.BOOLEAN,
                null_count=0,
                unique_count=1
            )
        elif isinstance(value, np.ndarray):
            return ValueMetadata(
                value_type=ValueType.ARRAY,
                shape=value.shape,
                dtype=str(value.dtype),
                min_value=float(np.nanmin(value)) if value.size > 0 else None,
                max_value=float(np.nanmax(value)) if value.size > 0 else None,
                mean_value=float(np.nanmean(value)) if value.size > 0 else None,
                std_value=float(np.nanstd(value)) if value.size > 0 else None,
                null_count=int(np.isnan(value).sum()) if value.dtype.kind in 'fc' else 0,
                unique_count=len(np.unique(value))
            )
        elif isinstance(value, pd.Series):
            return ValueMetadata(
                value_type=ValueType.SERIES,
                shape=(len(value),),
                dtype=str(value.dtype),
                min_value=float(value.min()) if not value.empty else None,
                max_value=float(value.max()) if not value.empty else None,
                mean_value=float(value.mean()) if not value.empty else None,
                std_value=float(value.std()) if not value.empty else None,
                null_count=int(value.isnull().sum()),
                unique_count=len(value.unique())
            )
        elif isinstance(value, pd.DataFrame):
            return ValueMetadata(
                value_type=ValueType.DATAFRAME,
                shape=value.shape,
                dtype=str(value.dtypes.iloc[0]) if len(value.columns) > 0 else None,
                null_count=int(value.isnull().sum().sum()),
                unique_count=len(value.drop_duplicates())
            )
        else:
            return ValueMetadata(
                value_type=ValueType.STRING,  # fallback
                null_count=0,
                unique_count=1
            )
    
    def _safe_delta(self, a: ValueLike, b: ValueLike) -> Optional[ValueLike]:
        """安全に差分を計算"""
        try:
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                if any(
                    map(lambda x: isinstance(x, float) and (math.isnan(x) or math.isinf(x)), [a, b])
                ):
                    return None
                return b - a
            if isinstance(a, pd.Series) and isinstance(b, pd.Series):
                return b.subtract(a, fill_value=np.nan)
            if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
                return b.subtract(a, fill_value=np.nan)
            if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
                if a.shape == b.shape:
                    return b - a
                return None
        except Exception:
            return None
        return None
    
    @staticmethod
    def _merge_companies(inputs: Sequence[Valuator]) -> List[CompanyRef]:
        """企業情報をマージ"""
        seen: Dict[str, CompanyRef] = {}
        for v in inputs:
            for c in v.companies:
                seen[c.company_id] = c
        return list(seen.values())
    
    @staticmethod
    def _infer_country(inputs: Sequence[Valuator]) -> Optional[str]:
        """国を推論"""
        for v in inputs:
            if v.country:
                return v.country
        return None
    
    @staticmethod
    def _get_memory_usage() -> Optional[float]:
        """メモリ使用量を取得（簡易実装）"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return None