"""
ValueFlow: Traceable Computation Graph for Values

ValueFlowは、任意の値（数値、配列、pandasのSeries/DataFrameなど）に
メタ情報と計算履歴を付与し、Valuator同士の計算から計算グラフを構築して
依存関係の可視化/追跡を可能にするライブラリです。

主な機能:
- 値の上書き（override）時に差分情報（DiffResult）を生成
- 企業リスト、構成銘柄の時系列などの添付データを保持
- Excel出力、Mermaid記法での可視化をサポート
- 計算グラフの可視化と追跡

使用例:
    import pandas as pd
    from valueflow import Valuator, ValuationGraph, InferenceType, DiffDirection, CompanyRef

    # グラフの作成
    g = ValuationGraph()

    # 基本値の作成
    operating_profit = Valuator(
        value_name="operating_profit",
        value=120.0,
        description="営業利益",
        graph=g
    )
    
    invested_capital = Valuator(
        value_name="invested_capital",
        value=1000.0,
        description="投下資本",
        graph=g
    )

    # 計算関数の定義
    def roic_calculation(values: list) -> float:
        op, ic = values
        return op / ic

    # 計算による値の作成
    roic = Valuator.compute(
        name="roic",
        func=roic_calculation,
        inputs=[operating_profit, invested_capital],
        graph=g,
        inference=InferenceType.ACTUAL,
        description="ROIC = 営業利益 / 投下資本"
    )

    # 値の上書き
    roic.override_value(
        0.15,
        direction=DiffDirection.UP,
        freq="annual",
        description="モデル推計値に差し替え"
    )

    # 企業情報の添付
    company = CompanyRef(
        company_id="FSYM123",
        name="Company A",
        ticker="AAA"
    )
    roic.companies = [company]

    # 時系列データの添付
    timeseries = pd.Series(
        [100, 101, 99],
        index=pd.date_range("2024-01-01", periods=3)
    )
    roic.attach_company_timeseries({"FSYM123": timeseries})

    # 結果の出力
    print(roic.to_mermaid())
    roic.to_excel("/tmp/roic_analysis.xlsx")
"""

from .types import (
    DiffDirection,
    DiffResult,
    InferenceType,
    SourceInfo,
    CompanyRef,
    ComputationRecord,
    ValueMetadata,
    ValueType,
    ValueLike,
    ComputeFunc,
)
from .graph import ValuationGraph, Edge, Node
from .valuator import Valuator

__version__ = "1.0.0"

__all__ = [
    # 型定義
    "DiffDirection",
    "DiffResult", 
    "InferenceType",
    "SourceInfo",
    "CompanyRef",
    "ComputationRecord",
    "ValueMetadata",
    "ValueType",
    "ValueLike",
    "ComputeFunc",
    
    # グラフ関連
    "ValuationGraph",
    "Edge",
    "Node",
    
    # メインクラス
    "Valuator",
    
    # バージョン
    "__version__",
]