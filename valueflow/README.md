# ValueFlow: Traceable Computation Graph for Values

ValueFlowは、任意の値（数値、配列、pandasのSeries/DataFrameなど）にメタ情報と計算履歴を付与し、Valuator同士の計算から計算グラフを構築して依存関係の可視化/追跡を可能にするライブラリです。

## 🚀 主な機能

- **値の追跡**: 任意の値にメタデータと計算履歴を付与
- **計算グラフ**: Valuator同士の計算から依存関係グラフを自動構築
- **値の上書き**: 上書き時に差分情報（DiffResult）を自動生成
- **企業データ管理**: 企業リスト、構成銘柄の時系列データを保持
- **可視化**: Mermaid記法、NetworkX、Excel出力をサポート
- **型安全性**: Pydantic v2による完全な型検証とバリデーション

## 📦 インストール

```bash
pip install valueflow
```

## 🎯 基本的な使用方法

### 1. 基本的な値の作成

```python
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
```

### 2. 計算による値の作成

```python
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

print(f"ROIC: {roic.value}")  # 0.12
```

### 3. 値の上書きと差分追跡

```python
# 値の上書き
diff = roic.override_value(
    0.15,
    direction=DiffDirection.UP,
    freq="annual", 
    description="モデル推計値に差し替え"
)

print(f"差分: {diff.delta}")  # 0.03
print(f"方向: {diff.direction}")  # up
```

### 4. 企業データの管理

```python
# 企業情報の作成
company = CompanyRef(
    company_id="FSYM123",
    name="Company A",
    ticker="AAA",
    country="US",
    sector="Technology"
)

# 企業情報の添付
roic.companies = [company]

# 時系列データの添付
timeseries = pd.Series(
    [100, 101, 99, 102, 98],
    index=pd.date_range("2024-01-01", periods=5)
)
roic.attach_company_timeseries({"FSYM123": timeseries})
```

### 5. 可視化と出力

```python
# Mermaid記法での可視化
mermaid_diagram = roic.to_mermaid()
print(mermaid_diagram)

# Excel出力
roic.to_excel("/tmp/roic_analysis.xlsx")

# NetworkXグラフ
nx_graph = roic.to_networkx()
```

## 🔧 高度な使用方法

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

### バリデーションとエラーハンドリング

```python
from pydantic import ValidationError

try:
    # 無効な値での作成
    invalid_valuator = Valuator(
        value_name="",  # 空の名前は無効
        value=100.0
    )
except ValidationError as e:
    print(f"バリデーションエラー: {e}")

# 型安全な値の作成
try:
    safe_valuator = Valuator(
        value_name="test_value",
        value=100.0,
        country="US",  # 自動的に大文字に変換
        inference=InferenceType.ACTUAL
    )
    print("正常に作成されました")
except ValidationError as e:
    print(f"エラー: {e}")
```

## 📊 データ型サポート

ValueFlowは以下のデータ型をサポートしています：

- **基本型**: `int`, `float`, `str`, `bool`
- **配列型**: `numpy.ndarray`
- **時系列型**: `pandas.Series`, `pandas.DataFrame`
- **コレクション型**: `list`, `dict`

```python
import numpy as np

# 配列データ
array_data = Valuator(
    value_name="price_array",
    value=np.array([100, 101, 99, 102]),
    description="価格配列"
)

# DataFrameデータ
df_data = Valuator(
    value_name="financial_data",
    value=pd.DataFrame({
        'revenue': [1000, 1100, 1200],
        'profit': [100, 110, 120]
    }),
    description="財務データ"
)
```

## 🔍 計算グラフの分析

```python
# グラフの統計情報
stats = g.get_statistics()
print(f"ノード数: {stats['node_count']}")
print(f"エッジ数: {stats['edge_count']}")
print(f"非循環: {stats['is_acyclic']}")

# トポロジカルソート
sorted_nodes = g.topological_sort()
print(f"計算順序: {sorted_nodes}")

# 特定ノードの祖先・子孫
ancestors = g.ancestors("roic")
descendants = g.descendants("operating_profit")
```

## 📈 パフォーマンス最適化

```python
# 実行時間とメモリ使用量の追跡
def optimized_calculation(values: list) -> float:
    # 重い計算処理
    result = sum(values) / len(values)
    return result

# 計算時のメトリクス取得
result = Valuator.compute(
    name="optimized_result",
    func=optimized_calculation,
    inputs=[operating_profit, invested_capital],
    graph=g
)

print(f"実行時間: {result.last_computation.execution_time_ms}ms")
print(f"メモリ使用量: {result.last_computation.memory_usage_mb}MB")
```

## 🧪 テストとデバッグ

```python
# 値のメタデータ確認
metadata = roic.value_metadata
print(f"値の型: {metadata.value_type}")
print(f"形状: {metadata.shape}")
print(f"最小値: {metadata.min_value}")
print(f"最大値: {metadata.max_value}")

# 計算履歴の確認
if roic.is_computed:
    print(f"計算名: {roic.last_computation.name}")
    print(f"使用関数: {roic.last_computation.func_repr}")
    print(f"入力ノード: {roic.last_computation.inputs}")

# 上書き履歴の確認
if roic.is_overridden:
    print(f"上書き前: {roic.differ.before_value}")
    print(f"上書き後: {roic.differ.after_value}")
    print(f"差分: {roic.differ.delta}")
```

## 🔄 シリアライゼーション

```python
# JSON形式での保存・読み込み
json_data = roic.to_json()
restored_roic = Valuator.from_json(json_data)

# 辞書形式での保存・読み込み
dict_data = roic.to_dict()
restored_roic = Valuator.from_dict(dict_data)

# グラフの保存・読み込み
graph_json = g.to_json()
new_graph = ValuationGraph()
new_graph.from_json(graph_json)
```

## 🎨 可視化オプション

### Mermaid記法

```python
# 完全なグラフ
full_diagram = g.to_mermaid()

# 特定ノードをルートとしたサブグラフ
subgraph_diagram = g.to_mermaid(root_id="roic")
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

## 🏢 企業データ管理

```python
# 複数企業の管理
companies = [
    CompanyRef(company_id="FSYM001", name="Company A", ticker="AAA"),
    CompanyRef(company_id="FSYM002", name="Company B", ticker="BBB"),
    CompanyRef(company_id="FSYM003", name="Company C", ticker="CCC")
]

# 企業固有の値
company_specific_roic = Valuator(
    value_name="company_roic",
    value=0.15,
    companies=companies,
    is_company_unique=True
)

# 企業別時系列データ
company_timeseries = {
    "FSYM001": pd.Series([0.12, 0.13, 0.14]),
    "FSYM002": pd.Series([0.10, 0.11, 0.12]),
    "FSYM003": pd.Series([0.16, 0.17, 0.18])
}

company_specific_roic.attach_company_timeseries(company_timeseries)
```

## 🔧 設定とカスタマイズ

```python
# カスタム計算関数
def custom_roic_calculation(values: list) -> float:
    """カスタムROIC計算（調整済み）"""
    operating_income, invested_capital, adjustments = values
    adjusted_income = operating_income + adjustments
    return adjusted_income / invested_capital

# 調整項目の追加
adjustments = Valuator(
    value_name="adjustments",
    value=10.0,
    description="調整項目",
    graph=g
)

# カスタム計算の実行
custom_roic = Valuator.compute(
    name="custom_roic",
    func=custom_roic_calculation,
    inputs=[operating_profit, invested_capital, adjustments],
    graph=g,
    description="調整済みROIC"
)
```

## 📚 API リファレンス

### Valuator

#### 基本メソッド

- `override_value(new_value, **kwargs)`: 値の上書き
- `attach_company_timeseries(series_map)`: 時系列データの添付
- `get_company_timeseries()`: 時系列データの取得
- `to_excel(path)`: Excel出力
- `to_mermaid()`: Mermaid記法出力
- `to_networkx()`: NetworkXグラフ出力

#### クラスメソッド

- `Valuator.compute(name, func, inputs, **kwargs)`: 計算による値の作成

### ValuationGraph

#### 基本メソッド

- `add_node(node_id, **attrs)`: ノードの追加
- `add_edge(src, dst, **attrs)`: エッジの追加
- `predecessors(node_id)`: 前駆ノードの取得
- `successors(node_id)`: 後続ノードの取得
- `ancestors(node_id)`: 祖先ノードの取得
- `descendants(node_id)`: 子孫ノードの取得
- `topological_sort()`: トポロジカルソート
- `is_acyclic()`: 非循環性の確認

#### 出力メソッド

- `to_mermaid(root_id=None)`: Mermaid記法出力
- `to_networkx()`: NetworkXグラフ出力
- `to_json()`: JSON出力
- `from_json(json_str)`: JSONからの復元

## 🤝 貢献

ValueFlowはオープンソースプロジェクトです。貢献を歓迎します！

1. リポジトリをフォーク
2. フィーチャーブランチを作成
3. 変更をコミット
4. プルリクエストを送信

## 📄 ライセンス

MIT License

## 🔗 関連リンク

- [GitHub Repository](https://github.com/your-org/valueflow)
- [Documentation](https://valueflow.readthedocs.io)
- [PyPI Package](https://pypi.org/project/valueflow)

## 📞 サポート

質問や問題がある場合は、GitHubのIssuesページでお知らせください。