# データプロバイダー (Data Providers)

このモジュールは、外部データソースからデータを取得・統合するためのプロバイダークラス群を提供します。各プロバイダーは特定のデータソースに特化し、統一されたインターフェースでデータアクセスを実現します。

## 概要

データプロバイダーは以下の設計思想に基づいて構築されています：

- **統一インターフェース**: すべてのプロバイダーが共通の基底クラスを継承
- **期間管理**: `wolf_period`ライブラリによる一貫した期間フィルタリング
- **パフォーマンス最適化**: バッチ処理、並列化、キャッシュ機能
- **型安全性**: Pydanticモデルによる厳密なデータ検証
- **拡張性**: 新しいデータソースの追加が容易

## アーキテクチャ

```
data_providers/
├── core/                    # 共通機能
│   └── base_provider.py    # 基底クラス
├── database/               # データベース接続
│   ├── db_connections.py   # 接続管理
│   └── sqlserver.py        # SQL Server実装
├── sources/                # データソース別プロバイダー
│   ├── factset/           # FactSetデータ
│   ├── ciq/               # CIQデータ
│   ├── msci/              # MSCIデータ
│   ├── rbics/             # RBICSデータ
│   ├── revere/            # REVEREデータ
│   └── segment/           # セグメントデータ
└── utils/                 # ユーティリティ
    └── data_validation.py # データ検証
```

## データソース一覧

### 1. FactSetProvider
**役割**: FactSetのシンボル/財務データを高速かつ安全に統合取得

**主要機能**:
- 企業識別子データの取得（ISIN/SEDOL/CUSIP/TICKER等）
- 財務データの取得（売上・利益・資産・負債・財務比率等）
- WolfPeriod/WolfPeriodRangeによる期間フィルタリング
- 地域・証券タイプによるフィルタリング

**使用例**:
```python
from data_providers.sources.factset.provider import FactSetProvider
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

# プロバイダーの初期化
provider = FactSetProvider(max_workers=4)

# 企業識別子データの取得
identity_records = provider.get_identity_records(
    country=["US", "JP"],      # 米国・日本の企業
    active_only=True,          # アクティブな証券のみ
)

# 財務データの取得（期間範囲）
period_range = WolfPeriodRange(
    start=WolfPeriod.from_month(2023, 1, freq=Frequency.M),
    stop=WolfPeriod.from_month(2023, 12, freq=Frequency.M),
)

financial_records = provider.get_financial_records(
    period_range=period_range,  # 期間範囲
    country=["US", "JP"],       # 米国・日本の企業
    active_only=True,           # アクティブな証券のみ
    batch_size=5000,            # バッチサイズ
)
```

### 2. MSCIProvider
**役割**: MSCI構成銘柄データとインデックスデータの取得・管理、グローバルベータ計算用のデータ提供

**主要機能**:
- インデックス構成銘柄の取得
- インデックス値の取得
- ベータ計算（複数期間対応）
- 地域・セクター別フィルタリング

**使用例**:
```python
from data_providers.sources.msci.provider import MSCIProvider
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

# プロバイダーの初期化
provider = MSCIProvider(max_workers=4)

# インデックス構成銘柄の取得（単一期間）
constituents = provider.get_index_constituents(
    index_name='WORLD',
    dividend_flag=False,
    period=WolfPeriod.from_month(2023, 12, freq=Frequency.M)
)

# インデックス値の取得（期間範囲）
start_period = WolfPeriod.from_month(2023, 1, freq=Frequency.M)
end_period = WolfPeriod.from_month(2023, 12, freq=Frequency.M)
index_values = provider.get_index_values(
    index_name='WORLD',
    dividend_flag=False,
    period=WolfPeriodRange(start_period, end_period)
)

# ベータ計算（複数期間）
beta_results = provider.get_beta(
    security_ids=['US0378331005', 'US0231351067'],
    period=[
        WolfPeriod.from_month(2023, 12, freq=Frequency.M),
        WolfPeriod.from_month(2023, 11, freq=Frequency.M)
    ],
    lookback_periods=60,
    index_name='WORLD',
    dividend_flag=False
)
```

### 3. RBICSProvider
**役割**: FactSet REVEREデータベースからRBICS（セクター体系）データを取得・管理

**主要機能**:
- RBICS構造マスタデータの高速取得
- 企業のRBICS売上セグメント情報の取得
- 企業のRBICSフォーカス情報の取得
- WolfPeriod/WolfPeriodRangeによる期間フィルタリング

**使用例**:
```python
from data_providers.sources.rbics.provider import RBICSProvider
from data_providers.sources.rbics.types import SegmentType
from wolf_period import WolfPeriod
from datetime import date

# プロバイダーの初期化
provider = RBICSProvider(max_workers=4)

# RBICS構造マスタの取得
structure_records = provider.get_structure_records(
    period=WolfPeriod.from_day(date(2023, 12, 31))
)

# 企業のRBICS売上セグメント情報の取得
revenue_records = provider.get_company_records(
    segment_types=[SegmentType.REVENUE],
    factset_entity_ids=["001C7F-E", "002D8G-F"],
    min_revenue_share=0.05
)
```

### 4. CIQProvider
**役割**: CIQ（S&P Capital IQ）から企業識別子データ（Company ID, Ticker, ISIN等）を取得

**主要機能**:
- 企業識別子データの取得
- 上場・非上場の自動判別
- 国コードの正規化（alpha-2/alpha-3対応）

**使用例**:
```python
from data_providers.sources.ciq.provider import CIQProvider

# プロバイダーの初期化
provider = CIQProvider(max_workers=4)

# 企業識別子データの取得
identity_records = provider.get_identity_records(
    country=["US", "JPN"],     # 米国・日本の企業（alpha-2/alpha-3混在OK）
    listed_only=True,          # 上場企業のみ
)
```

### 5. SegmentDataProvider
**役割**: FactSetからセグメントデータを取得し、各種比率を計算

**主要機能**:
- セグメントデータの取得（売上、営業利益、資産）
- セグメント比率の計算（売上比率、利益比率、資産比率）
- 会計期間の調整と正規化

**使用例**:
```python
from data_providers.sources.segment.provider import SegmentDataProvider

# プロバイダーの初期化
provider = SegmentDataProvider()

# セグメントデータの取得
segment_data = provider.get_segment_data()
```

### 6. RevereDataProvider
**役割**: FactSet REVEREデータベースからセグメントデータを取得・処理

**主要機能**:
- REVEREセグメントデータの取得
- RBICS分類との統合
- セグメント名の正規化と重複除去

**使用例**:
```python
from data_providers.sources.revere.provider import RevereDataProvider

# プロバイダーの初期化
provider = RevereDataProvider()

# REVEREデータの取得
revere_data = provider.get_revere_data(
    fsym_ids=["000C7F-E", "002D8G-F"],
    fiscal_year=2023,
    min_revenue_share=0.05
)
```

## データベース接続

### GibDB
**役割**: 特定のGIB_DBへの接続実装（読み取り専用）

**接続情報**:
- ホスト: `172.22.200.25`
- ポート: `1433`
- ユーザー: `READ_MASTER`
- パスワード: `MASTER_READ123`

**使用例**:
```python
from data_providers.database.db_connections import GibDB

# 接続情報は固定（読み取り専用）
db = GibDB()

# クエリ実行（Decimal型は自動的にfloat/intに変換される）
df = db.execute_query("SELECT * FROM table")

# Decimal型の変換を無効化する場合
df = db.execute_query("SELECT * FROM table", convert_decimal=False)
```

### SQLServer
**役割**: Microsoft SQL Serverへの接続とクエリ実行の共通実装

**主要機能**:
- SQL Serverへの接続と切断
- クエリ実行結果のDataFrame変換
- Decimal型データの自動数値変換（NUMERIC, DECIMAL, MONEY型等に対応）

**特徴**:
- 抽象基底クラス（ABCMeta）として設計
- 接続パラメータの統一管理
- エラーハンドリングとログ出力
- リソース管理（カーソル・コネクションの自動クローズ）

## 共通機能 (BaseProvider)

すべてのプロバイダーは`BaseProvider`クラスを継承し、以下の共通機能を利用できます：

### データベース接続機能
- `GibDB`を継承したデータベース接続機能
- 自動的な接続管理とリソース解放

### テキスト正規化機能
```python
# Unicode正規化 (NFKC) と小文字化を適用
normalized_text = BaseProvider.normalize_text("Ａｐｐｌｅ Inc.")
# 結果: 'apple inc.'
```

### 重複データフィルタリング機能
```python
# 同一グループ内でFSYM_IDが複数存在する場合、
# PRIMARY_EQUITY_FLAG=1のレコードのみを残す
filtered_df = df.groupby('group_key').apply(BaseProvider.filter_func)
```

### 会計期間調整機能
```python
# 日本の会計年度基準 (4月開始) に基づいて処理
df_adjusted = BaseProvider.adjust_fiscal_term(df, date_col="DATE")
# FTERM_2: YYYYMM形式の会計期間
# FISCAL_YEAR: 会計年度
```

### 重複期間除去機能
```python
# 指定されたグループカラムと期間カラムで重複するデータを検出し、
# 最新の日付のレコードのみを残す
df_clean = BaseProvider.remove_duplicate_periods(
    df, 
    group_cols=["FSYM_ID", "FACTSET_ENTITY_ID"],
    date_col="DATE",
    period_col="FTERM_2"
)
```

## パフォーマンス最適化

### 並列処理
- 各プロバイダーで`max_workers`パラメータを調整
- `ThreadPoolExecutor`による計算/IOの分散
- 大量データでは`batch_size`の最適化

### キャッシュ戦略
- `@lru_cache`による重複計算/参照の抑制
- 頻繁に参照されるデータのキャッシュ化
- 期間フィルタの事前適用

### メモリ管理
- 大規模データセットの段階的処理
- 不要なデータの早期解放
- ベクトル化演算による処理速度向上

## エラーハンドリング

各プロバイダーは以下の例外を適切に処理します：

- **ValidationError**: レコード検証失敗（不正なデータ形式）
- **DatabaseError**: データベースアクセス失敗（接続・権限・SQL）
- **ValueError**: パラメータ検証失敗（不正な引数）
- **NotImplementedError**: 未実装機能呼び出し

## 設定と依存関係

### 主要依存パッケージ
- `pandas`: データ処理
- `pymssql`: SQL Server接続
- `pydantic`: データ検証
- `wolf_period`: 期間管理
- `concurrent.futures`: 並列処理

### 設定要件
- Python >= 3.13
- ネットワーク/認証: 社内DBやサードパーティデータに接続する場合、VPN・資格情報・IP制限等の要件を満たすこと
- ストレージ: 前処理結果の保存領域

## 開発ガイドライン

### 新しいプロバイダーの追加
1. `BaseProvider`を継承したクラスを作成
2. 適切な型定義（Pydanticモデル）を実装
3. クエリパラメータクラスを定義
4. バッチ処理とエラーハンドリングを実装
5. ドキュメントとテストを追加

### コードスタイル
- Python: 3.13以上
- 型ヒントの使用を推奨
- ログ出力は`get_logger(__name__)`を使用
- エラーハンドリングは適切な例外を発生
