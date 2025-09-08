"""
ValueFlow: パフォーマンステスト

このスクリプトは、ValueFlowのパフォーマンスを測定するベンチマークです。
"""

import time
from typing import List

import numpy as np
import pandas as pd
from valueflow import Valuator, ValuationGraph


def benchmark_basic_operations():
    """基本的な操作のベンチマーク"""
    print("=== 基本的な操作のベンチマーク ===")
    
    # Valuator作成のベンチマーク
    start_time = time.time()
    valuators = []
    for i in range(1000):
        valuator = Valuator(
            value_name=f"value_{i}",
            value=float(i),
            description=f"Value {i}"
        )
        valuators.append(valuator)
    end_time = time.time()
    
    print(f"Valuator作成 (1000個): {(end_time - start_time) * 1000:.2f}ms")
    print(f"1個あたり: {(end_time - start_time) * 1000 / 1000:.3f}ms")
    
    # 値の上書きのベンチマーク
    start_time = time.time()
    for valuator in valuators[:100]:
        valuator.override_value(valuator.value * 1.1)
    end_time = time.time()
    
    print(f"値の上書き (100個): {(end_time - start_time) * 1000:.2f}ms")
    print(f"1個あたり: {(end_time - start_time) * 1000 / 100:.3f}ms")


def benchmark_computation():
    """計算のベンチマーク"""
    print("\n=== 計算のベンチマーク ===")
    
    graph = ValuationGraph()
    
    # 大量の入力値を作成
    inputs = [Valuator(value_name=f"input_{i}", value=float(i), graph=graph) 
              for i in range(100)]
    
    # 単純な計算関数
    def simple_sum(values: List[float]) -> float:
        return sum(values)
    
    # 計算のベンチマーク
    start_time = time.time()
    result = Valuator.compute(
        name="sum_result",
        func=simple_sum,
        inputs=inputs,
        graph=graph
    )
    end_time = time.time()
    
    print(f"100個の値の合計計算: {(end_time - start_time) * 1000:.2f}ms")
    print(f"結果: {result.value}")
    
    # 複雑な計算関数
    def complex_calculation(values: List[float]) -> float:
        result = 0.0
        for i, val in enumerate(values):
            result += val ** 2 + np.sin(val) + np.log(val + 1)
        return result / len(values)
    
    # 複雑な計算のベンチマーク
    start_time = time.time()
    complex_result = Valuator.compute(
        name="complex_result",
        func=complex_calculation,
        inputs=inputs,
        graph=graph
    )
    end_time = time.time()
    
    print(f"複雑な計算 (100個の値): {(end_time - start_time) * 1000:.2f}ms")
    print(f"結果: {complex_result.value:.6f}")


def benchmark_graph_operations():
    """グラフ操作のベンチマーク"""
    print("\n=== グラフ操作のベンチマーク ===")
    
    graph = ValuationGraph()
    
    # 大量のノードの追加
    start_time = time.time()
    for i in range(1000):
        graph.add_node(f"node_{i}", label=f"Node {i}")
    end_time = time.time()
    
    print(f"ノード追加 (1000個): {(end_time - start_time) * 1000:.2f}ms")
    print(f"1個あたり: {(end_time - start_time) * 1000 / 1000:.3f}ms")
    
    # 大量のエッジの追加
    start_time = time.time()
    for i in range(999):
        graph.add_edge(f"node_{i}", f"node_{i+1}", label=f"edge_{i}")
    end_time = time.time()
    
    print(f"エッジ追加 (999個): {(end_time - start_time) * 1000:.2f}ms")
    print(f"1個あたり: {(end_time - start_time) * 1000 / 999:.3f}ms")
    
    # トポロジカルソート
    start_time = time.time()
    sorted_nodes = graph.topological_sort()
    end_time = time.time()
    
    print(f"トポロジカルソート (1000ノード): {(end_time - start_time) * 1000:.2f}ms")
    print(f"ソート結果の長さ: {len(sorted_nodes)}")
    
    # 祖先・子孫の計算
    start_time = time.time()
    ancestors = graph.ancestors("node_500")
    descendants = graph.descendants("node_500")
    end_time = time.time()
    
    print(f"祖先・子孫計算: {(end_time - start_time) * 1000:.2f}ms")
    print(f"祖先数: {len(ancestors)}")
    print(f"子孫数: {len(descendants)}")


def benchmark_serialization():
    """シリアライゼーションのベンチマーク"""
    print("\n=== シリアライゼーションのベンチマーク ===")
    
    # 複雑なValuatorの作成
    graph = ValuationGraph()
    
    # 複数の計算チェーンを作成
    inputs = [Valuator(value_name=f"input_{i}", value=float(i), graph=graph) 
              for i in range(50)]
    
    def calculation(values: List[float]) -> float:
        return sum(values) / len(values)
    
    result = Valuator.compute(
        name="result",
        func=calculation,
        inputs=inputs,
        graph=graph
    )
    
    # 時系列データの添付
    timeseries = pd.Series(
        np.random.randn(1000),
        index=pd.date_range("2024-01-01", periods=1000)
    )
    result.attach_company_timeseries({"TEST": timeseries})
    
    # JSON変換のベンチマーク
    start_time = time.time()
    json_data = result.to_json()
    end_time = time.time()
    
    print(f"JSON変換: {(end_time - start_time) * 1000:.2f}ms")
    print(f"JSONサイズ: {len(json_data)} 文字")
    
    # JSON復元のベンチマーク
    start_time = time.time()
    restored = Valuator.from_json(json_data)
    end_time = time.time()
    
    print(f"JSON復元: {(end_time - start_time) * 1000:.2f}ms")
    print(f"復元成功: {restored.value_name == result.value_name}")
    
    # 辞書変換のベンチマーク
    start_time = time.time()
    dict_data = result.to_dict()
    end_time = time.time()
    
    print(f"辞書変換: {(end_time - start_time) * 1000:.2f}ms")
    print(f"辞書キー数: {len(dict_data)}")


def benchmark_memory_usage():
    """メモリ使用量のベンチマーク"""
    print("\n=== メモリ使用量のベンチマーク ===")
    
    try:
        import psutil
        process = psutil.Process()
        
        # 初期メモリ使用量
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # 大量のValuatorを作成
        valuators = []
        for i in range(10000):
            valuator = Valuator(
                value_name=f"value_{i}",
                value=float(i),
                description=f"Value {i}"
            )
            valuators.append(valuator)
        
        # メモリ使用量の測定
        current_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = current_memory - initial_memory
        
        print(f"初期メモリ使用量: {initial_memory:.2f}MB")
        print(f"10000個のValuator作成後: {current_memory:.2f}MB")
        print(f"メモリ増加量: {memory_used:.2f}MB")
        print(f"1個あたり: {memory_used / 10000:.3f}MB")
        
        # グラフ付きValuatorのメモリ使用量
        graph = ValuationGraph()
        graph_valuators = []
        
        for i in range(1000):
            valuator = Valuator(
                value_name=f"graph_value_{i}",
                value=float(i),
                graph=graph
            )
            graph_valuators.append(valuator)
        
        # エッジの追加
        for i in range(999):
            graph.add_edge(f"graph_value_{i}", f"graph_value_{i+1}")
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        graph_memory_used = final_memory - current_memory
        
        print(f"グラフ付きValuator (1000個) + エッジ (999個) 作成後: {final_memory:.2f}MB")
        print(f"グラフ関連のメモリ増加量: {graph_memory_used:.2f}MB")
        print(f"1個あたり: {graph_memory_used / 1000:.3f}MB")
        
    except ImportError:
        print("psutilがインストールされていません。メモリ使用量の測定をスキップします。")


def benchmark_large_data():
    """大量データのベンチマーク"""
    print("\n=== 大量データのベンチマーク ===")
    
    # 大きな配列データ
    large_array = np.random.randn(10000)
    start_time = time.time()
    array_valuator = Valuator(
        value_name="large_array",
        value=large_array,
        description="Large array data"
    )
    end_time = time.time()
    
    print(f"大きな配列 (10000要素) のValuator作成: {(end_time - start_time) * 1000:.2f}ms")
    
    # 大きなDataFrameデータ
    large_df = pd.DataFrame({
        'col1': np.random.randn(1000),
        'col2': np.random.randn(1000),
        'col3': np.random.randn(1000),
        'col4': np.random.randn(1000),
        'col5': np.random.randn(1000)
    })
    
    start_time = time.time()
    df_valuator = Valuator(
        value_name="large_dataframe",
        value=large_df,
        description="Large DataFrame data"
    )
    end_time = time.time()
    
    print(f"大きなDataFrame (1000x5) のValuator作成: {(end_time - start_time) * 1000:.2f}ms")
    
    # メタデータの計算時間
    start_time = time.time()
    metadata = array_valuator.value_metadata
    end_time = time.time()
    
    print(f"配列メタデータ計算: {(end_time - start_time) * 1000:.2f}ms")
    print(f"  - 形状: {metadata.shape}")
    print(f"  - 最小値: {metadata.min_value:.3f}")
    print(f"  - 最大値: {metadata.max_value:.3f}")
    print(f"  - 平均値: {metadata.mean_value:.3f}")
    print(f"  - 標準偏差: {metadata.std_value:.3f}")


def main():
    """メイン関数"""
    print("ValueFlow パフォーマンステスト")
    print("=" * 50)
    
    benchmark_basic_operations()
    benchmark_computation()
    benchmark_graph_operations()
    benchmark_serialization()
    benchmark_memory_usage()
    benchmark_large_data()
    
    print("\n" + "=" * 50)
    print("パフォーマンステスト完了")


if __name__ == "__main__":
    main()
