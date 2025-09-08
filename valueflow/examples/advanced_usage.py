"""
ValueFlow: 高度な使用例

このスクリプトは、ValueFlowの高度な機能を示す例です。
複雑な計算チェーン、パフォーマンス最適化、エラーハンドリングなどを含みます。
"""

import time
from typing import List

import numpy as np
import pandas as pd
from valueflow import (
    Valuator,
    ValuationGraph,
    DiffDirection,
    InferenceType,
    CompanyRef,
    SourceInfo,
)


def main():
    """高度な使用例の実行"""
    print("=== ValueFlow 高度な使用例 ===\n")
    
    # 1. 複雑な計算チェーンの構築
    print("1. 複雑な計算チェーンの構築")
    g = ValuationGraph()
    
    # 財務データの作成
    revenue = Valuator(value_name="revenue", value=1000.0, graph=g)
    cogs = Valuator(value_name="cogs", value=600.0, graph=g)
    operating_expenses = Valuator(value_name="operating_expenses", value=200.0, graph=g)
    depreciation = Valuator(value_name="depreciation", value=50.0, graph=g)
    interest_expense = Valuator(value_name="interest_expense", value=30.0, graph=g)
    tax_rate = Valuator(value_name="tax_rate", value=0.25, graph=g)
    
    # 資産データ
    total_assets = Valuator(value_name="total_assets", value=5000.0, graph=g)
    current_assets = Valuator(value_name="current_assets", value=2000.0, graph=g)
    current_liabilities = Valuator(value_name="current_liabilities", value=1000.0, graph=g)
    
    # 負債データ
    total_debt = Valuator(value_name="total_debt", value=1500.0, graph=g)
    equity = Valuator(value_name="equity", value=3500.0, graph=g)
    
    # 計算関数の定義
    def gross_profit_func(values: List[float]) -> float:
        """売上総利益の計算"""
        rev, cost = values
        return rev - cost
    
    def operating_profit_func(values: List[float]) -> float:
        """営業利益の計算"""
        gp, op_exp, dep = values
        return gp - op_exp - dep
    
    def ebit_func(values: List[float]) -> float:
        """EBITの計算"""
        op_profit, int_exp = values
        return op_profit - int_exp
    
    def net_income_func(values: List[float]) -> float:
        """純利益の計算"""
        ebit, tax = values
        return ebit * (1 - tax)
    
    def roa_func(values: List[float]) -> float:
        """ROAの計算"""
        ni, assets = values
        return ni / assets
    
    def roe_func(values: List[float]) -> float:
        """ROEの計算"""
        ni, equity_val = values
        return ni / equity_val
    
    def current_ratio_func(values: List[float]) -> float:
        """流動比率の計算"""
        ca, cl = values
        return ca / cl
    
    def debt_to_equity_func(values: List[float]) -> float:
        """負債対自己資本比率の計算"""
        debt, equity_val = values
        return debt / equity_val
    
    def wacc_func(values: List[float]) -> float:
        """WACCの計算（簡易版）"""
        cost_of_equity, cost_of_debt, tax, equity_ratio, debt_ratio = values
        return (cost_of_equity * equity_ratio + 
                cost_of_debt * (1 - tax) * debt_ratio)
    
    def value_creation_func(values: List[float]) -> float:
        """価値創造指標の計算"""
        roic, wacc = values
        return roic - wacc
    
    # 計算チェーンの実行
    print("計算チェーンの実行中...")
    
    # 売上総利益
    gross_profit = Valuator.compute(
        name="gross_profit",
        func=gross_profit_func,
        inputs=[revenue, cogs],
        graph=g,
        description="売上総利益 = 売上高 - 売上原価"
    )
    
    # 営業利益
    operating_profit = Valuator.compute(
        name="operating_profit",
        func=operating_profit_func,
        inputs=[gross_profit, operating_expenses, depreciation],
        graph=g,
        description="営業利益 = 売上総利益 - 販管費 - 減価償却費"
    )
    
    # EBIT
    ebit = Valuator.compute(
        name="ebit",
        func=ebit_func,
        inputs=[operating_profit, interest_expense],
        graph=g,
        description="EBIT = 営業利益 - 支払利息"
    )
    
    # 純利益
    net_income = Valuator.compute(
        name="net_income",
        func=net_income_func,
        inputs=[ebit, tax_rate],
        graph=g,
        description="純利益 = EBIT × (1 - 税率)"
    )
    
    # ROA
    roa = Valuator.compute(
        name="roa",
        func=roa_func,
        inputs=[net_income, total_assets],
        graph=g,
        description="ROA = 純利益 / 総資産"
    )
    
    # ROE
    roe = Valuator.compute(
        name="roe",
        func=roe_func,
        inputs=[net_income, equity],
        graph=g,
        description="ROE = 純利益 / 自己資本"
    )
    
    # 流動比率
    current_ratio = Valuator.compute(
        name="current_ratio",
        func=current_ratio_func,
        inputs=[current_assets, current_liabilities],
        graph=g,
        description="流動比率 = 流動資産 / 流動負債"
    )
    
    # 負債対自己資本比率
    debt_to_equity = Valuator.compute(
        name="debt_to_equity",
        func=debt_to_equity_func,
        inputs=[total_debt, equity],
        graph=g,
        description="負債対自己資本比率 = 総負債 / 自己資本"
    )
    
    # WACC（簡易版）
    cost_of_equity = Valuator(value_name="cost_of_equity", value=0.08, graph=g)
    cost_of_debt = Valuator(value_name="cost_of_debt", value=0.04, graph=g)
    equity_ratio = Valuator(value_name="equity_ratio", value=0.7, graph=g)
    debt_ratio = Valuator(value_name="debt_ratio", value=0.3, graph=g)
    
    wacc = Valuator.compute(
        name="wacc",
        func=wacc_func,
        inputs=[cost_of_equity, cost_of_debt, tax_rate, equity_ratio, debt_ratio],
        graph=g,
        description="WACC = 自己資本コスト × 自己資本比率 + 負債コスト × (1 - 税率) × 負債比率"
    )
    
    # 価値創造指標
    value_creation = Valuator.compute(
        name="value_creation",
        func=value_creation_func,
        inputs=[roa, wacc],
        graph=g,
        description="価値創造指標 = ROA - WACC"
    )
    
    # 結果の表示
    print("\n財務指標の計算結果:")
    print(f"売上総利益: {gross_profit.value:.2f}")
    print(f"営業利益: {operating_profit.value:.2f}")
    print(f"EBIT: {ebit.value:.2f}")
    print(f"純利益: {net_income.value:.2f}")
    print(f"ROA: {roa.value:.3f}")
    print(f"ROE: {roe.value:.3f}")
    print(f"流動比率: {current_ratio.value:.2f}")
    print(f"負債対自己資本比率: {debt_to_equity.value:.3f}")
    print(f"WACC: {wacc.value:.3f}")
    print(f"価値創造指標: {value_creation.value:.3f}")
    
    # 2. パフォーマンス最適化の例
    print("\n2. パフォーマンス最適化の例")
    
    def optimized_calculation(values: List[float]) -> float:
        """最適化された計算関数"""
        # 重い計算処理をシミュレート
        result = 0.0
        for val in values:
            result += val ** 2
        return result / len(values)
    
    # 大量のデータでの計算
    large_data = [Valuator(value_name=f"data_{i}", value=float(i), graph=g) 
                  for i in range(100)]
    
    start_time = time.time()
    optimized_result = Valuator.compute(
        name="optimized_result",
        func=optimized_calculation,
        inputs=large_data[:10],  # 最初の10個のみ使用
        graph=g
    )
    end_time = time.time()
    
    print(f"最適化計算結果: {optimized_result.value:.3f}")
    print(f"実行時間: {(end_time - start_time) * 1000:.2f}ms")
    if optimized_result.last_computation:
        print(f"記録された実行時間: {optimized_result.last_computation.execution_time_ms:.2f}ms")
    
    # 3. エラーハンドリングの例
    print("\n3. エラーハンドリングの例")
    
    def error_prone_calculation(values: List[float]) -> float:
        """エラーが発生する可能性のある計算"""
        if len(values) < 2:
            raise ValueError("少なくとも2つの値が必要です")
        if values[1] == 0:
            raise ZeroDivisionError("ゼロ除算エラー")
        return values[0] / values[1]
    
    # 正常なケース
    try:
        normal_result = Valuator.compute(
            name="normal_result",
            func=error_prone_calculation,
            inputs=[Valuator(value_name="a", value=10.0, graph=g),
                   Valuator(value_name="b", value=2.0, graph=g)],
            graph=g
        )
        print(f"正常な計算結果: {normal_result.value}")
    except Exception as e:
        print(f"正常なケースでエラー: {e}")
    
    # エラーケース
    try:
        error_result = Valuator.compute(
            name="error_result",
            func=error_prone_calculation,
            inputs=[Valuator(value_name="a", value=10.0, graph=g),
                   Valuator(value_name="b", value=0.0, graph=g)],
            graph=g
        )
        print(f"エラーケースの結果: {error_result.value}")
    except Exception as e:
        print(f"エラーケース（期待通り）: {e}")
    
    # 4. 複数企業の管理
    print("\n4. 複数企業の管理")
    
    companies = [
        CompanyRef(company_id="FSYM001", name="Company A", ticker="AAA", country="US"),
        CompanyRef(company_id="FSYM002", name="Company B", ticker="BBB", country="JP"),
        CompanyRef(company_id="FSYM003", name="Company C", ticker="CCC", country="GB")
    ]
    
    # 企業固有の値
    company_specific_roa = Valuator(
        value_name="company_roa",
        value=0.15,
        companies=companies,
        is_company_unique=True,
        graph=g
    )
    
    # 企業別時系列データ
    company_timeseries = {}
    for i, company in enumerate(companies):
        timeseries = pd.Series(
            [0.12 + i*0.01, 0.13 + i*0.01, 0.14 + i*0.01],
            index=pd.date_range("2024-01-01", periods=3, freq="M")
        )
        company_timeseries[company.company_id] = timeseries
    
    company_specific_roa.attach_company_timeseries(company_timeseries)
    
    print(f"管理企業数: {len(company_specific_roa.companies)}")
    for company in company_specific_roa.companies:
        print(f"  - {company.name} ({company.ticker})")
    
    # 5. グラフの分析
    print("\n5. グラフの分析")
    
    stats = g.get_statistics()
    print(f"ノード数: {stats['node_count']}")
    print(f"エッジ数: {stats['edge_count']}")
    print(f"非循環: {stats['is_acyclic']}")
    print(f"最大入次数: {stats['max_in_degree']}")
    print(f"最大出次数: {stats['max_out_degree']}")
    
    # トポロジカルソート
    try:
        sorted_nodes = g.topological_sort()
        print(f"計算順序（最初の10個）: {sorted_nodes[:10]}")
    except ValueError as e:
        print(f"トポロジカルソートエラー: {e}")
    
    # 6. 可視化
    print("\n6. 可視化")
    
    # Mermaid記法での出力
    mermaid_diagram = value_creation.to_mermaid()
    print("価値創造指標の計算グラフ（Mermaid記法）:")
    print(mermaid_diagram[:500] + "..." if len(mermaid_diagram) > 500 else mermaid_diagram)
    
    # 7. シリアライゼーション
    print("\n7. シリアライゼーション")
    
    # 複雑なオブジェクトのシリアライゼーション
    complex_valuator = value_creation
    
    # JSON形式
    json_data = complex_valuator.to_json()
    print(f"JSON形式のサイズ: {len(json_data)} 文字")
    
    # 辞書形式
    dict_data = complex_valuator.to_dict()
    print(f"辞書形式のキー数: {len(dict_data)}")
    
    # 復元テスト
    restored_valuator = Valuator.from_json(json_data)
    print(f"復元成功: {restored_valuator.value_name == complex_valuator.value_name}")
    
    # 8. バッチ処理の例
    print("\n8. バッチ処理の例")
    
    def batch_calculation(values: List[float]) -> float:
        """バッチ計算関数"""
        return sum(values) / len(values)
    
    # 複数のバッチを作成
    batch_results = []
    for i in range(5):
        batch_inputs = [Valuator(value_name=f"batch_{i}_input_{j}", value=float(j), graph=g) 
                       for j in range(10)]
        
        batch_result = Valuator.compute(
            name=f"batch_{i}_result",
            func=batch_calculation,
            inputs=batch_inputs,
            graph=g
        )
        batch_results.append(batch_result)
    
    print(f"バッチ処理結果数: {len(batch_results)}")
    for i, result in enumerate(batch_results):
        print(f"  バッチ {i}: {result.value:.3f}")
    
    print("\n=== 高度な使用例完了 ===")


if __name__ == "__main__":
    main()
