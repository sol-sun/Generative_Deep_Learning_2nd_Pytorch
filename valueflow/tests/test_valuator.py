"""
ValueFlow: Valuator のテスト

このモジュールは、Valuatorクラスの機能をテストします。
"""

import json
import math
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from valueflow import (
    Valuator,
    ValuationGraph,
    DiffDirection,
    InferenceType,
    CompanyRef,
    SourceInfo,
    ValueType,
)


class TestValuator:
    """Valuatorクラスのテスト"""
    
    def test_basic_creation(self):
        """基本的なValuatorの作成テスト"""
        valuator = Valuator(
            value_name="test_value",
            value=100.0,
            description="テスト値"
        )
        
        assert valuator.value_name == "test_value"
        assert valuator.value == 100.0
        assert valuator.description == "テスト値"
        assert valuator.inference == InferenceType.ACTUAL
        assert valuator.id is not None
        assert isinstance(valuator.created_at, datetime)
    
    def test_validation_errors(self):
        """バリデーションエラーのテスト"""
        # 空の名前
        with pytest.raises(ValidationError):
            Valuator(value_name="", value=100.0)
        
        # 無効な国コード
        with pytest.raises(ValidationError):
            Valuator(value_name="test", value=100.0, country="INVALID")
    
    def test_value_metadata_computation(self):
        """値のメタデータ計算テスト"""
        # 数値
        valuator = Valuator(value_name="float_value", value=123.45)
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.FLOAT
        assert metadata.min_value == 123.45
        assert metadata.max_value == 123.45
        assert metadata.mean_value == 123.45
        
        # 配列
        array_value = np.array([1, 2, 3, 4, 5])
        valuator = Valuator(value_name="array_value", value=array_value)
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.ARRAY
        assert metadata.shape == (5,)
        assert metadata.min_value == 1.0
        assert metadata.max_value == 5.0
        assert metadata.mean_value == 3.0
        
        # Series
        series_value = pd.Series([10, 20, 30, 40])
        valuator = Valuator(value_name="series_value", value=series_value)
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.SERIES
        assert metadata.shape == (4,)
        assert metadata.min_value == 10.0
        assert metadata.max_value == 40.0
        assert metadata.mean_value == 25.0
    
    def test_override_value(self):
        """値の上書きテスト"""
        valuator = Valuator(value_name="test", value=100.0)
        
        diff = valuator.override_value(
            150.0,
            direction=DiffDirection.UP,
            freq="annual",
            description="予想値に更新"
        )
        
        assert valuator.value == 150.0
        assert diff.direction == DiffDirection.UP
        assert diff.before_value == 100.0
        assert diff.after_value == 150.0
        assert diff.delta == 50.0
        assert diff.freq == "annual"
        assert diff.description == "予想値に更新"
        assert valuator.is_overridden
    
    def test_compute_method(self):
        """計算メソッドのテスト"""
        graph = ValuationGraph()
        
        # 入力値の作成
        a = Valuator(value_name="a", value=10.0, graph=graph)
        b = Valuator(value_name="b", value=5.0, graph=graph)
        
        # 計算関数
        def add_func(values: List[float]) -> float:
            return sum(values)
        
        # 計算実行
        result = Valuator.compute(
            name="sum",
            func=add_func,
            inputs=[a, b],
            graph=graph,
            description="加算"
        )
        
        assert result.value == 15.0
        assert result.value_name == "sum"
        assert result.is_computed
        assert result.last_computation is not None
        assert result.last_computation.name == "sum"
        assert len(result.last_computation.inputs) == 2
        assert a.id in result.last_computation.inputs
        assert b.id in result.last_computation.inputs
    
    def test_company_management(self):
        """企業データ管理のテスト"""
        company = CompanyRef(
            company_id="FSYM123",
            name="Test Company",
            ticker="TEST",
            country="US"
        )
        
        valuator = Valuator(
            value_name="company_value",
            value=100.0,
            companies=[company]
        )
        
        assert len(valuator.companies) == 1
        assert valuator.companies[0].company_id == "FSYM123"
        assert valuator.companies[0].name == "Test Company"
    
    def test_timeseries_attachment(self):
        """時系列データ添付のテスト"""
        valuator = Valuator(value_name="test", value=100.0)
        
        # 時系列データの作成
        timeseries = pd.Series(
            [100, 101, 99, 102],
            index=pd.date_range("2024-01-01", periods=4)
        )
        
        # 添付
        valuator.attach_company_timeseries({"FSYM123": timeseries})
        
        # 取得
        retrieved = valuator.get_company_timeseries()
        assert retrieved is not None
        assert "FSYM123" in retrieved
        assert len(retrieved["FSYM123"]) == 4
    
    def test_excel_export(self):
        """Excel出力のテスト"""
        valuator = Valuator(
            value_name="test_value",
            value=100.0,
            description="テスト値",
            inference=InferenceType.ACTUAL
        )
        
        # 企業情報の追加
        company = CompanyRef(
            company_id="FSYM123",
            name="Test Company",
            ticker="TEST"
        )
        valuator.companies = [company]
        
        # 時系列データの追加
        timeseries = pd.Series([100, 101, 99])
        valuator.attach_company_timeseries({"FSYM123": timeseries})
        
        # 一時ファイルに出力
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            valuator.to_excel(tmp.name)
            
            # ファイルが作成されたことを確認
            assert Path(tmp.name).exists()
            
            # 内容の確認
            with pd.ExcelFile(tmp.name) as xls:
                assert "metadata" in xls.sheet_names
                assert "value" in xls.sheet_names
                assert "companies" in xls.sheet_names
                assert "ts_FSYM123" in xls.sheet_names
    
    def test_mermaid_output(self):
        """Mermaid出力のテスト"""
        graph = ValuationGraph()
        
        a = Valuator(value_name="a", value=10.0, graph=graph)
        b = Valuator(value_name="b", value=5.0, graph=graph)
        
        def add_func(values: List[float]) -> float:
            return sum(values)
        
        result = Valuator.compute(
            name="sum",
            func=add_func,
            inputs=[a, b],
            graph=graph
        )
        
        mermaid = result.to_mermaid()
        assert "flowchart TD" in mermaid
        assert "a" in mermaid
        assert "b" in mermaid
        assert "sum" in mermaid
    
    def test_serialization(self):
        """シリアライゼーションテスト"""
        valuator = Valuator(
            value_name="test",
            value=100.0,
            description="テスト値"
        )
        
        # JSON変換
        json_str = valuator.to_json()
        assert isinstance(json_str, str)
        
        # 復元
        restored = Valuator.from_json(json_str)
        assert restored.value_name == valuator.value_name
        assert restored.value == valuator.value
        assert restored.description == valuator.description
        
        # 辞書変換
        dict_data = valuator.to_dict()
        assert isinstance(dict_data, dict)
        
        # 辞書から復元
        restored_from_dict = Valuator.from_dict(dict_data)
        assert restored_from_dict.value_name == valuator.value_name
    
    def test_complex_calculation_chain(self):
        """複雑な計算チェーンのテスト"""
        graph = ValuationGraph()
        
        # 基本値
        revenue = Valuator(value_name="revenue", value=1000.0, graph=graph)
        costs = Valuator(value_name="costs", value=800.0, graph=graph)
        assets = Valuator(value_name="assets", value=5000.0, graph=graph)
        
        # 営業利益の計算
        def operating_profit_func(values: List[float]) -> float:
            rev, cost = values
            return rev - cost
        
        operating_profit = Valuator.compute(
            name="operating_profit",
            func=operating_profit_func,
            inputs=[revenue, costs],
            graph=graph
        )
        
        # ROAの計算
        def roa_func(values: List[float]) -> float:
            profit, asset = values
            return profit / asset
        
        roa = Valuator.compute(
            name="roa",
            func=roa_func,
            inputs=[operating_profit, assets],
            graph=graph
        )
        
        assert operating_profit.value == 200.0
        assert roa.value == 0.04
        
        # グラフの確認
        assert graph.has_node(revenue.id)
        assert graph.has_node(roa.id)
        assert graph.has_edge(revenue.id, operating_profit.id)
        assert graph.has_edge(operating_profit.id, roa.id)
    
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
        # 計算エラー
        def error_func(values: List[float]) -> float:
            return values[0] / 0  # ゼロ除算
        
        a = Valuator(value_name="a", value=10.0)
        
        with pytest.raises(RuntimeError, match="Computation failed"):
            Valuator.compute(
                name="error_result",
                func=error_func,
                inputs=[a]
            )
        
        # グラフなしでのMermaid出力
        valuator = Valuator(value_name="test", value=100.0)
        with pytest.raises(RuntimeError, match="graph is not attached"):
            valuator.to_mermaid()
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # NaN値の処理
        valuator = Valuator(value_name="nan_value", value=float('nan'))
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.FLOAT
        
        # 無限大値の処理
        valuator = Valuator(value_name="inf_value", value=float('inf'))
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.FLOAT
        
        # 空の配列
        empty_array = np.array([])
        valuator = Valuator(value_name="empty_array", value=empty_array)
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.ARRAY
        assert metadata.shape == (0,)
        
        # 空のSeries
        empty_series = pd.Series([], dtype=float)
        valuator = Valuator(value_name="empty_series", value=empty_series)
        metadata = valuator.value_metadata
        assert metadata.value_type == ValueType.SERIES
        assert metadata.shape == (0,)
    
    def test_performance_metrics(self):
        """パフォーマンスメトリクスのテスト"""
        def slow_func(values: List[float]) -> float:
            import time
            time.sleep(0.01)  # 10ms待機
            return sum(values)
        
        a = Valuator(value_name="a", value=10.0)
        b = Valuator(value_name="b", value=20.0)
        
        result = Valuator.compute(
            name="slow_result",
            func=slow_func,
            inputs=[a, b]
        )
        
        assert result.last_computation is not None
        assert result.last_computation.execution_time_ms is not None
        assert result.last_computation.execution_time_ms >= 10.0  # 10ms以上


class TestValuationGraph:
    """ValuationGraphクラスのテスト"""
    
    def test_basic_operations(self):
        """基本的な操作のテスト"""
        graph = ValuationGraph()
        
        # ノードの追加
        graph.add_node("node1", label="Node 1")
        graph.add_node("node2", label="Node 2")
        
        assert graph.has_node("node1")
        assert graph.has_node("node2")
        assert not graph.has_node("node3")
        
        # エッジの追加
        edge_id = graph.add_edge("node1", "node2", label="test_edge")
        assert graph.has_edge("node1", "node2")
        assert not graph.has_edge("node2", "node1")
        
        # 前駆・後続ノード
        preds = list(graph.predecessors("node2"))
        succs = list(graph.successors("node1"))
        
        assert "node1" in preds
        assert "node2" in succs
    
    def test_topological_sort(self):
        """トポロジカルソートのテスト"""
        graph = ValuationGraph()
        
        # グラフの構築: A -> B -> C
        graph.add_node("A", label="A")
        graph.add_node("B", label="B")
        graph.add_node("C", label="C")
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        
        # ソート
        sorted_nodes = graph.topological_sort()
        assert sorted_nodes == ["A", "B", "C"]
        
        # 非循環性の確認
        assert graph.is_acyclic()
    
    def test_cycle_detection(self):
        """循環検出のテスト"""
        graph = ValuationGraph()
        
        # 循環グラフの構築: A -> B -> C -> A
        graph.add_node("A", label="A")
        graph.add_node("B", label="B")
        graph.add_node("C", label="C")
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("C", "A")
        
        # 循環の確認
        assert not graph.is_acyclic()
        
        # トポロジカルソートでエラー
        with pytest.raises(ValueError, match="contains cycles"):
            graph.topological_sort()
    
    def test_ancestors_descendants(self):
        """祖先・子孫ノードのテスト"""
        graph = ValuationGraph()
        
        # グラフの構築: A -> B -> C, A -> D
        graph.add_node("A", label="A")
        graph.add_node("B", label="B")
        graph.add_node("C", label="C")
        graph.add_node("D", label="D")
        
        graph.add_edge("A", "B")
        graph.add_edge("B", "C")
        graph.add_edge("A", "D")
        
        # 祖先の確認
        ancestors = graph.ancestors("C")
        assert "A" in ancestors
        assert "B" in ancestors
        assert "C" not in ancestors
        
        # 子孫の確認
        descendants = graph.descendants("A")
        assert "B" in descendants
        assert "C" in descendants
        assert "D" in descendants
        assert "A" not in descendants
    
    def test_mermaid_output(self):
        """Mermaid出力のテスト"""
        graph = ValuationGraph()
        
        graph.add_node("A", label="Node A")
        graph.add_node("B", label="Node B")
        graph.add_edge("A", "B", label="edge_label")
        
        mermaid = graph.to_mermaid()
        assert "flowchart TD" in mermaid
        assert "Node A" in mermaid
        assert "Node B" in mermaid
        assert "edge_label" in mermaid
    
    def test_json_serialization(self):
        """JSONシリアライゼーションのテスト"""
        graph = ValuationGraph()
        
        graph.add_node("A", label="Node A", node_type="input")
        graph.add_node("B", label="Node B", node_type="computation")
        graph.add_edge("A", "B", label="compute", weight=1.0)
        
        # JSON出力
        json_str = graph.to_json()
        assert isinstance(json_str, str)
        
        # 復元
        new_graph = ValuationGraph()
        new_graph.from_json(json_str)
        
        assert new_graph.has_node("A")
        assert new_graph.has_node("B")
        assert new_graph.has_edge("A", "B")
        
        node_a = new_graph.get_node("A")
        assert node_a is not None
        assert node_a.label == "Node A"
        assert node_a.node_type == "input"
    
    def test_statistics(self):
        """統計情報のテスト"""
        graph = ValuationGraph()
        
        graph.add_node("A", label="A", node_type="input")
        graph.add_node("B", label="B", node_type="computation")
        graph.add_node("C", label="C", node_type="output")
        
        graph.add_edge("A", "B", edge_type="computation")
        graph.add_edge("B", "C", edge_type="computation")
        
        stats = graph.get_statistics()
        
        assert stats["node_count"] == 3
        assert stats["edge_count"] == 2
        assert stats["is_acyclic"] is True
        assert stats["max_in_degree"] == 1
        assert stats["max_out_degree"] == 1
        assert "input" in stats["node_types"]
        assert "computation" in stats["node_types"]
        assert "output" in stats["node_types"]


if __name__ == "__main__":
    pytest.main([__file__])
