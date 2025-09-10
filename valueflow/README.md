はい、承知いたしました。`README.md`をコピー＆ペーストで使えるようにスニペット形式で書き出します。

````markdown
# ValueFlow: Traceable Computation Graph for Values

ValueFlowは、任意の値（数値、配列、pandasのSeries/DataFrameなど）にメタ情報と計算履歴を付与し、Valuator同士の計算から計算グラフを構築して依存関係の可視化/追跡を可能にするライブラリです。

## 主な機能

- **値の追跡**: 任意の値にメタデータと計算履歴を付与。
- **計算グラフ**: Valuator同士の計算から依存関係グラフを自動構築。
- **値の上書き**: 上書き時に差分情報（DiffResult）を自動生成。
- **企業データ管理**: 企業リストや時系列データを柔軟に保持。
- **可視化**: Mermaid記法、NetworkX、Excel出力をサポート。
- **型安全性**: Pydantic v2による完全な型検証とIDEでの補完。

## インストール

```bash
pip install numpy pandas pydantic
# networkxとopenpyxlはオプションです
pip install networkx matplotlib openpyxl
````

*(注: このライブラリは現在PyPIに公開されていません。上記のファイル構成でローカルに設置して使用します。)*

## 基本的な使用方法

### 1\. インポートとグラフの初期化

```python
import pandas as pd
from valueflow import Valuator, ValuationGraph, InferenceType, DiffDirection

# 計算グラフのインスタンスを作成
g = ValuationGraph()
```

### 2\. 基本的な値の作成

```python
# 営業利益Valuator
operating_profit = Valuator(
    value_name="operating_profit",
    value=120.0,
    description="営業利益",
    graph=g
)

# 投下資本Valuator
invested_capital = Valuator(
    value_name="invested_capital", 
    value=1000.0,
    description="投下資本",
    graph=g
)
```

### 3\. 計算による値の作成

```python
# ROICを計算する関数を定義
def roic_calculation(values: list) -> float:
    op, ic = values
    return op / ic

# Valuator.computeを使用して新しいValuatorを計算
roic = Valuator.compute(
    name="roic",
    func=roic_calculation,
    inputs=[operating_profit, invested_capital],
    graph=g,
    description="ROIC = 営業利益 / 投下資本"
)

print(f"ROIC: {roic.value:.2f}")
# > ROIC: 0.12
```

### 4\. 値の上書きと差分追跡

```python
# 値をモデル推計値に上書き
diff = roic.override_value(
    new_value=0.15,
    direction=DiffDirection.UP,
    description="モデル推計値に差し替え"
)

print(f"上書き後の値: {roic.value}")
# > 上書き後の値: 0.15
print(f"差分情報: {diff.delta:.2f}")
# > 差分情報: 0.03
```

### 5\. 可視化と出力

```python
# Mermaid記法での計算グラフの可視化
mermaid_diagram = roic.to_mermaid()
print(mermaid_diagram)

# Excelへの出力
# roic.to_excel("roic_analysis.xlsx")

# NetworkXグラフへの変換
# nx_graph = g.to_networkx()
```

## 高度な使用方法

### 複雑な計算チェーン

```python
# 複数の計算を組み合わせ
def wacc_calculation(values: list) -> float:
    cost_of_equity, cost_of_debt, tax_rate, equity_ratio, debt_ratio = values
    return (cost_of_equity * equity_ratio + 
            cost_of_debt * (1 - tax_rate) * debt_ratio)

# 入力値の作成
cost_of_equity = Valuator(value_name="cost_of_equity", value=0.08, graph=g)
cost_of_debt = Valuator(value_name="cost_of_debt", value=0.04, graph=g)
tax_rate = Valuator(value_name="tax_rate", value=0.25, graph=g)
equity_ratio = Valuator(value_name="equity_ratio", value=0.6, graph=g)
debt_ratio = Valuator(value_name="debt_ratio", value=0.4, graph=g)

# WACCの計算
wacc = Valuator.compute(
    name="wacc",
    func=wacc_calculation,
    inputs=[cost_of_equity, cost_of_debt, tax_rate, equity_ratio, debt_ratio],
    graph=g,
    description="WACC計算"
)

# ROICとWACCの比較
def value_creation(values: list) -> float:
    roic_val, wacc_val = values
    return roic_val - wacc_val

value_creation_metric = Valuator.compute(
    name="value_creation",
    func=value_creation,
    inputs=[roic, wacc],
    graph=g,
    description="価値創造指標"
)
```

### データソース情報の管理

```python
from valueflow import SourceInfo

# データソース情報の設定
source_info = SourceInfo(
    dataset="factset_fundamentals",
    provider="FactSet",
    fields=["operating_income", "total_assets"],
    note="四半期データ",
    version="2024Q1"
)

operating_profit.source = source_info
```

## データ型サポート

ValueFlowは以下のデータ型をサポートしています：

  - **基本型**: `int`, `float`, `str`, `bool`
  - **配列型**: `numpy.ndarray`
  - **時系列型**: `pandas.Series`, `pandas.DataFrame`
  - **コレクション型**: `list`, `dict`

<!-- end list -->

```python
import numpy as np

# 配列データ
array_data = Valuator(
    value_name="price_array",
    value=np.array([100, 101, 99, 102]),
    description="価格配列",
    graph=g
)

# DataFrameデータ
df_data = Valuator(
    value_name="financial_data",
    value=pd.DataFrame({
        'revenue': [1000, 1100, 1200],
        'profit': [100, 110, 120]
    }),
    description="財務データ",
    graph=g
)
```

## 可視化オプション

### Mermaid記法

```python
# 特定ノードをルートとしたサブグラフ
subgraph_diagram = g.to_mermaid(root_id="roic")
print(subgraph_diagram)
```

### NetworkX統合

```python
import networkx as nx
import matplotlib.pyplot as plt

# NetworkXグラフに変換
nx_graph = g.to_networkx()

# 可視化
pos = nx.spring_layout(nx_graph)
nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue')
plt.show()
```

## API リファレンス

### Valuator

  - `override_value(new_value, **kwargs)`: 値の上書き
  - `attach_company_timeseries(series_map)`: 時系列データの添付
  - `to_excel(path)`: Excel出力
  - `to_mermaid()`: Mermaid記法出力
  - `to_networkx()`: NetworkXグラフ出力
  - `copy()`: Valuatorのコピーを作成
  - `Valuator.compute(name, func, inputs, graph, **kwargs)`: 計算による値の作成

### ValuationGraph

  - `add_node(node)`: ノードの追加
  - `add_edge(edge)`: エッジの追加
  - `predecessors(node_id)`: 前駆ノードの取得
  - `successors(node_id)`: 後続ノードの取得
  - `ancestors(node_id)`: 祖先ノードの取得
  - `to_mermaid(root_id=None)`: Mermaid記法出力
  - `to_networkx()`: NetworkXグラフ出力
