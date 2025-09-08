"""
ValueFlow: 基本的な使用例

このスクリプトは、ValueFlowの基本的な機能を示す例です。
"""

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
    """基本的な使用例の実行"""
    print("=== ValueFlow 基本的な使用例 ===\n")
    
    # 1. グラフの作成
    print("1. グラフの作成")
    g = ValuationGraph()
    
    # 2. 基本値の作成
    print("2. 基本値の作成")
    operating_profit = Valuator(
        value_name="operating_profit",
        value=120.0,
        description="営業利益",
        graph=g,
        country="US"
    )
    
    invested_capital = Valuator(
        value_name="invested_capital",
        value=1000.0,
        description="投下資本",
        graph=g,
        country="US"
    )
    
    print(f"営業利益: {operating_profit.value}")
    print(f"投下資本: {invested_capital.value}")
    
    # 3. 計算による値の作成
    print("\n3. 計算による値の作成")
    
    def roic_calculation(values: list) -> float:
        """ROIC計算関数"""
        op, ic = values
        return op / ic
    
    roic = Valuator.compute(
        name="roic",
        func=roic_calculation,
        inputs=[operating_profit, invested_capital],
        graph=g,
        inference=InferenceType.ACTUAL,
        description="ROIC = 営業利益 / 投下資本"
    )
    
    print(f"ROIC: {roic.value:.3f}")
    print(f"計算された値: {roic.is_computed}")
    
    # 4. 値の上書き
    print("\n4. 値の上書き")
    diff = roic.override_value(
        0.15,
        direction=DiffDirection.UP,
        freq="annual",
        description="モデル推計値に差し替え"
    )
    
    print(f"上書き後のROIC: {roic.value}")
    print(f"差分: {diff.delta:.3f}")
    print(f"方向: {diff.direction}")
    print(f"上書きされた値: {roic.is_overridden}")
    
    # 5. 企業データの管理
    print("\n5. 企業データの管理")
    
    company = CompanyRef(
        company_id="FSYM123",
        name="Example Corp",
        ticker="EXAM",
        country="US",
        sector="Technology"
    )
    
    roic.companies = [company]
    print(f"関連企業: {roic.companies[0].name}")
    
    # 6. 時系列データの添付
    print("\n6. 時系列データの添付")
    
    timeseries = pd.Series(
        [0.12, 0.13, 0.14, 0.15, 0.16],
        index=pd.date_range("2024-01-01", periods=5, freq="M")
    )
    
    roic.attach_company_timeseries({"FSYM123": timeseries})
    print(f"時系列データ: {len(timeseries)} 期間")
    
    # 7. データソース情報の設定
    print("\n7. データソース情報の設定")
    
    source_info = SourceInfo(
        dataset="factset_fundamentals",
        provider="FactSet",
        fields=["operating_income", "total_assets"],
        note="四半期データ",
        version="2024Q1"
    )
    
    operating_profit.source = source_info
    print(f"データソース: {operating_profit.source.dataset}")
    
    # 8. 可視化
    print("\n8. 可視化")
    
    mermaid_diagram = roic.to_mermaid()
    print("Mermaid記法:")
    print(mermaid_diagram)
    
    # 9. グラフの統計情報
    print("\n9. グラフの統計情報")
    stats = g.get_statistics()
    print(f"ノード数: {stats['node_count']}")
    print(f"エッジ数: {stats['edge_count']}")
    print(f"非循環: {stats['is_acyclic']}")
    
    # 10. 値のメタデータ
    print("\n10. 値のメタデータ")
    metadata = roic.value_metadata
    print(f"値の型: {metadata.value_type}")
    print(f"最小値: {metadata.min_value}")
    print(f"最大値: {metadata.max_value}")
    print(f"平均値: {metadata.mean_value}")
    
    # 11. 計算履歴
    print("\n11. 計算履歴")
    if roic.last_computation:
        print(f"計算名: {roic.last_computation.name}")
        print(f"使用関数: {roic.last_computation.func_repr}")
        print(f"入力ノード: {roic.last_computation.inputs}")
        print(f"実行時間: {roic.last_computation.execution_time_ms:.2f}ms")
    
    # 12. シリアライゼーション
    print("\n12. シリアライゼーション")
    
    # JSON形式での保存
    json_data = roic.to_json()
    print(f"JSON形式のサイズ: {len(json_data)} 文字")
    
    # 辞書形式での保存
    dict_data = roic.to_dict()
    print(f"辞書形式のキー数: {len(dict_data)}")
    
    print("\n=== 基本的な使用例完了 ===")


if __name__ == "__main__":
    main()
