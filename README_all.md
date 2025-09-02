# Global PPM (Global Portfolio Performance Management)

ベイジアン推論を核にした、企業セグメント/連結財務の時系列を統合解析する研究用ツール群です。FactSetやRBICS（REVERE）等のデータを取り込み、Stanで学習可能な入力データを生成し、その後の統計的推論へ繋げます。

本ドキュメントは「ゼロから動かす」ために必要な情報をすべて含みます。セットアップ、設定、CLIの逐次実行、生成物の解釈、Providerの仕様、トラブルシューティング、発展的な活用例まで段階的に解説します。初見の方は本章末の「クイックスタート（最短）」から試せます。

参考ファイル（主要実装の出発点）
- `src/gppm/cli/prepare_data.py:1`
- `src/gppm/cli/bayes_infer.py:1`
- `src/gppm/bayes/bayesian_data_processor.py:1`
- `src/gppm/pipeline/data_manager.py:1`
- `src/gppm/providers/factset_provider.py:1`

目標品質（完成基準 10 項目）
1) 目的が明確であること（何を解くか・得られる成果）
2) 再現可能性があること（要件・インストール・設定が具体的）
3) 手順が一貫していること（実行順・入出力が論理的）
4) 参照が具体的であること（コード参照が明示）
5) 初心者配慮（概念から実行まで迷わない）
6) エラーハンドリング（代表的失敗と対処がわかる）
7) 設定の透明性（YAML/環境変数の優先順位と意味）
8) Providerの見取り図（役割・入出力・注意点）
9) 将来拡張の道筋（推論エンジン連携計画）
10) 用語・表記の一貫性（記法・コマンドを統一）

このREADMEは上記すべてを満たすよう構成されています。

目次
1. 概要とアーキテクチャ
2. セットアップ（要件・インストール・検証）
3. 設定（gppm_config.yml と環境変数）
4. 実行チュートリアル（前処理 → ベイズ推論）
5. 生成物の構造（pickle のキーと意味）
6. プロバイダー仕様（FactSet/RBICS/REVERE/MSCI/CIQ/Segment/SQLServer）
7. 実践レシピ（E2Eサンプル・Notebook・ユースケース）
8. トラブルシューティング（FAQ付き）
9. 開発・テスト・コードスタイル
10. ライセンス/サポート

---

1. 概要とアーキテクチャ
------------------------

目的
- 企業の連結/セグメントROIC、製品シェア、WACCなどの情報を統合し、Stanに投入可能な時系列・階層データを作ること。
- 製品・地理・企業の階層に沿った因果/予測モデリングのための安定した土台を提供。

全体フロー
1) データ取得: Provider群がFactSet/REVERE/RBICS等からデータを取り出す
2) 統合処理: DataManagerが重複除去・キー整合・マスタ結合を行う
3) 前処理: BayesianDataProcessorがスコア計算、ROIC/WACC統合、ピボット化、Stan辞書の生成
4) 推論: Stan実装群（将来拡張）でサンプリング・推定を実行

主要モジュール（導線）
- 前処理パイプライン: `src/gppm/bayes/bayesian_data_processor.py:1`
- データ統合: `src/gppm/pipeline/data_manager.py:1`
- CLI（前処理）: `src/gppm/cli/prepare_data.py:1`
- CLI（推論・スキャフォールド）: `src/gppm/cli/bayes_infer.py:1`

設計方針
- 入力依存（Provider層）と統合/前処理（Pipeline/Processor層）を分離。
- 期間管理は `wolf_period` で一貫化し、月/四半期/会計年度の整合性を担保。
- 設定は `gppm_config.yml` と環境変数で制御し、CIやバッチ実行に適合。

---

クイックスタート（最短）
----------------------

前提: 依存インストール済み（`pip install -e .`）。実データ接続なしで、CSVから最小動作を確認します。

1) サンプルCSVを用意
- 例: `data/sample_segments.csv`（列例: `company_id,segment_id,yyyymm,revenue,roic`）

2) 前処理を実行
```bash
python -m gppm.cli.prepare_data \
  --config gppm_config.yml \
  --input-csv data/sample_segments.csv \
  --freq M \
  --fy-start 4 \
  --out-dir /tmp/gppm_output
```

3) 出力を確認
- `/tmp/gppm_output/processed_data/bayesian_dataset.pkl`
- `/tmp/gppm_output/logs/prepare_data.log`

4) Notebookで中身を覗く（任意）
- `notebooks/factset_dataset_refactored.ipynb:1` を開き、pickleを読み込んで概要を表示

2. セットアップ（要件・インストール・検証）
--------------------------------------------

2.1 要件
- Python >= 3.13（`pyproject.toml:1` に準拠）
- OS: Linux / macOS / Windows
- ネットワーク/認証: 社内DBやサードパーティデータに接続する場合、VPN・資格情報・IP制限等の要件を満たすこと
- ストレージ: 前処理結果のpickle保存領域（既定 `/tmp/gppm_output/processed_data/`）

2.2 インストール
```bash
# 仮想環境作成（任意）
python -m venv .venv
source .venv/bin/activate  # Windowsは .venv\Scripts\activate

# 開発インストール
pip install -e .

#（オプション）開発ツール
pip install -e .[dev]
```

2.3 動作検証（最小）
```bash
python - << 'PY'
from gppm.utils.config_manager import ConfigManager
cfg = ConfigManager().get_config()
print("config ok:", cfg.log_level, cfg.data.start_period, cfg.output.base_directory)
PY
```

2.4 代表的な依存パッケージ
- pandas, numpy, pyyaml
- pydantic v2, pydantic-settings
- SQL接続（環境に応じて）

---

3. 設定（gppm_config.yml と環境変数）
------------------------------------

3.1 設定ファイル（プロジェクトルートの `gppm_config.yml:1`）
```yaml
log_level: INFO
log_format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

processing:
  batch_size: 1000
  max_workers: 4

data:
  start_period: 201909  # YYYYMM
  end_period:   202406  # YYYYMM

output:
  base_directory: "/tmp/gppm_output"
  dataset_filename: "bayesian_dataset.pkl"

# 製品マッピング（pickle; 必須列: FACTSET_ENTITY_ID, PRODUCT_L6_ID, RELABEL_L6_ID）
mapping_df_path: null
```

3.2 環境変数による上書き（`GPPM_` 接頭辞、ネストは `__`）
```bash
export GPPM_LOG_LEVEL=DEBUG
export GPPM_DATA__START_PERIOD=202001
export GPPM_OUTPUT__BASE_DIRECTORY=/data/gppm
```

3.3 設定の詳細実装
- ローダ/バリデーション: `src/gppm/utils/config_manager.py:1`
- 優先順位: 環境変数 > YAML > デフォルト
- 追加の検証: 出力パス作成、期間の前後関係、マッピングファイル有無など

---

4. 実行チュートリアル（前処理 → ベイズ推論）
----------------------------------------------

4.1 前処理CLI（Stan用データ生成）
```bash
python -m gppm.cli.prepare_data
```
処理内容（抜粋）
- DataManagerでの取得/統合: 企業・財務・セグメント・RBICSマスタ・REVERE
- 製品スコア（`ProductScoreCalculator`）と地理正規化（`GeographicProcessor`）
- ROIC計算（連結/セグメント）
- 国別リスク係数からのWACC（連結）
- 期間・エンティティ整合、ピボットテーブル生成
- Stan辞書作成（観測・説明・添字のためのメタ）

実行ログの例（要約）
```
... 1. FactSetDataManagerからデータ取得中...
... 2. 製品スコア・ピボット作成...
... 3. ROIC/WACC 計算...
... 9. ピボットテーブル作成完了
... 11. Stan用データ構造作成完了
保存先: /tmp/gppm_output/processed_data/bayesian_dataset.pkl
```

4.2 生成結果の読み取り（Python API）
```python
from pathlib import Path
import pickle

p = Path("/tmp/gppm_output/processed_data/bayesian_dataset.pkl")
data = pickle.loads(p.read_bytes())

print(data.keys())          # dict_keys(['raw_data','pivot_tables','stan_data','product_names','entity_info','processing_info'])
print(data['stan_data'].keys())
print(data['processing_info'])
```

4.3 ベイズ推論 CLI（スキャフォールド）
```bash
python -m gppm.cli.bayes_infer \
  --data "/tmp/gppm_output/processed_data/bayesian_dataset.pkl" \
  --engine cmdstanpy \
  --model /path/to/model.stan \
  --iter 1000 --chains 4 --seed 42 \
  --output "/tmp/gppm_output/inference"
```
現状の動作
- 入力pickleの検証とサマリの生成
- `inference_meta.json` を出力（実サンプリングは未実装）
実装: `src/gppm/cli/bayes_infer.py:1`

4.4 将来の推論実装（方針）
- CmdStanPy/PyStan/Numpyro のいずれかを選択できる構成
- 学習成果物（posterior draws, summary, diagnostics）の保存
- 再現性（seed, chains, iter, thin, adapt等の制御）
- 事後チェック・予測（PPC）のスクリプト化

---

5. 生成物の構造（pickle のキーと意味）
--------------------------------------

5.1 stan_data（学習入力）
- 次元: `Segment_N`, `Company_N`, `Product_N`, `Time_N`
- 観測: `y_segment_obs`（セグメントROIC）, `y_consol_obs`（連結ROIC）, `y_consol_wacc_obs`（連結WACC; 任意）
- 添字: `segment_idx`, `consol_idx`, `time_idx` などの圧縮添字
- 説明変数: `Share`（時間×セグメント×製品）, `Share_consol`（時間×企業×製品）
- 欠損対応: 観測ベクトルとインデックスによりスパースな観測を表現

5.2 pivot_tables（確認・デバッグ・可視化に便利）
- `Y_segment`, `Y_consol`, `Y_consol_wacc`（必要に応じて）
- `X2_segment`, `X2_consol`（製品シェア）
- 整合済みの行列として形が崩れないことを保証

5.3 raw_data（最終整形済みのDataFrame群）
- `segment`, `consol`（ROIC・WACC結合済みのロング形式）
- `segment_product_share`, `consol_product_share`
- 期間列は `FTERM_2`（YYYYMM; `BayesianDataProcessor.convert_date_format`で統一）

5.4 メタ情報
- `processing_info`: 期間、件数、プロダクト数等の集約
- `product_names`, `entity_info`: ラベル付けや外部結合時の手掛かり

---

## プロバイダー仕様

本プロジェクトでは、外部データソースからデータを取得・統合するためのProviderクラス群を提供しています。各Providerは特定のデータソースに特化し、統一されたインターフェースでデータアクセスを実現します。

### FactSetProvider（企業ID/財務）

**役割**: FactSetのシンボル/財務データを高速かつ安全に統合取得

**主要機能**:
- 企業識別子レコードの取得（`get_identity_records`）
- 財務レコードの取得（`get_financial_records`）
- WolfPeriod/WolfPeriodRangeによる一貫した期間管理
- 大規模データ向けのバッチ/並列/キャッシュ最適化

**主要API**:
```python
from gppm.providers.factset_provider import FactSetProvider
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

# プロバイダーの初期化
provider = FactSetProvider(max_workers=4)

# 企業識別子データの取得
identity_records = provider.get_identity_records(
    country=["US", "JP"],      # 米国・日本の企業
    active_only=True,          # アクティブな証券のみ
)

# 財務データの取得（期間範囲）
period_range = WolfPeriodRange.from_periods(
    WolfPeriod.from_month(2023, 1, freq=Frequency.M),
    WolfPeriod.from_month(2023, 12, freq=Frequency.M),
)

financial_records = provider.get_financial_records(
    period_range=period_range,  # 期間範囲
    country=["US", "JP"],       # 米国・日本の企業
    active_only=True,           # アクティブな証券のみ
    batch_size=5000,            # バッチサイズ
)
```

**データモデル**:
- `FactSetIdentityRecord`: 企業識別子レコード（FSYM_ID、企業名、各種コード）
- `FactSetFinancialRecord`: 財務レコード（期間、売上、利益、資産等）
- `FactSetQueryParams`: クエリパラメータ（フィルタ、期間、性能チューニング）

**パフォーマンス最適化**:
- バッチ検証: レコード検証をまとめて実行
- 並列化: `ThreadPoolExecutor`による計算/IOの分散
- キャッシュ: `@lru_cache`による重複計算/参照の抑制
- ベクトル化: pandasによる列演算の最適化

### RBICSProvider（セクター分類）

**役割**: FactSet REVEREデータベースからRBICS（セクター体系）データを取得・管理

**主要機能**:
- RBICS構造マスタデータの高速取得
- 企業のRBICS売上セグメント情報の取得
- 企業のRBICSフォーカス情報の取得
- WolfPeriod/WolfPeriodRangeによる期間フィルタリング

**主要API**:
```python
from gppm.providers.rbics_provider import RBICSProvider
from gppm.providers.rbics_types import SegmentType
from wolf_period import WolfPeriod

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

**データモデル**:
- `RBICSStructureRecord`: RBICS構造マスタ（L1-L6階層、説明）
- `RBICSCompanyRecord`: 企業RBICS情報（セグメント、売上比率）
- `RBICSQueryParams`: クエリパラメータ

### MSCIProvider（指数・属性等）

**役割**: MSCI構成銘柄データとインデックスデータの取得・管理、グローバルベータ計算用のデータ提供

**主要機能**:
- インデックス構成銘柄の取得
- インデックス値の取得
- ベータ計算（複数期間対応）
- 地域・セクター別フィルタリング

**主要API**:
```python
from gppm.providers.msci_data_provider import MSCIProvider
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

**データモデル**:
- `MSCIIndexRecord`: インデックス情報（名前、地域、セクター）
- `MSCISecurityRecord`: 証券情報（ISIN、名前、重み）
- `MSCIBetaRecord`: ベータ計算結果（ベータ値、R²、標準誤差）

### CIQProvider（ID補完）

**役割**: CIQ（S&P Capital IQ）から企業識別子データ（Company ID, Ticker, ISIN等）を取得

**主要機能**:
- 企業識別子データの取得
- 上場・非上場の自動判別
- 国コードの正規化（alpha-2/alpha-3対応）

**主要API**:
```python
from gppm.providers.ciq_provider import CIQProvider

# プロバイダーの初期化
provider = CIQProvider(max_workers=4)

# 企業識別子データの取得
identity_records = provider.get_identity_records(
    country=["US", "JPN"],     # 米国・日本の企業（alpha-2/alpha-3混在OK）
    listed_only=True,          # 上場企業のみ
)
```

**データモデル**:
- `CIQIdentityRecord`: 企業識別子レコード（Company ID、名前、各種コード）
- `CIQFinancialRecord`: 財務データ用のプレースホルダー（将来対応予定）

### SegmentDataProvider（セグメント財務）

**役割**: FactSetからセグメントデータを取得し、各種比率を計算

**主要機能**:
- セグメントデータの取得（売上、営業利益、資産）
- セグメント比率の計算（売上比率、利益比率、資産比率）
- 会計期間の調整と正規化

**主要API**:
```python
from gppm.providers.segment_data_provider import SegmentDataProvider

# プロバイダーの初期化
provider = SegmentDataProvider()

# セグメントデータの取得
segment_data = provider.get_segment_data()
```

**処理内容**:
- セグメント名の正規化
- 売上・営業利益・資産の合計計算
- 各セグメントの比率計算
- Reconciling Itemsの処理

### RevereDataProvider（REVERE補助）

**役割**: FactSet REVEREデータベースからセグメントデータを取得・処理

**主要機能**:
- REVEREセグメントデータの取得
- RBICS分類との統合
- セグメント名の正規化と重複除去

**主要API**:
```python
from gppm.providers.revere_data_provider import RevereDataProvider

# プロバイダーの初期化
provider = RevereDataProvider()

# REVEREデータの取得
revere_data = provider.get_revere_data()
```

**処理内容**:
- セグメント名の正規化
- 売上比率と会計年度の計算
- WolfPeriodを使用した会計年度計算
- セグメントシェアの合計計算と重複除去

### SQLServer（DB基盤）

**役割**: Microsoft SQL Serverへの接続とクエリ実行の共通実装

**主要機能**:
- SQL Serverへの接続と切断
- クエリ実行結果のDataFrame変換
- Decimal型データの自動数値変換（NUMERIC, DECIMAL, MONEY型等に対応）

**主要API**:
```python
from gppm.providers.sqlserver import SQLServer

# データベース接続（抽象クラスのため、具体的な実装クラスを使用）
# 例: AISGGibDB
db = AISGGibDB()

# クエリ実行（Decimal型は自動的にfloat/intに変換される）
df = db.execute_query("SELECT * FROM table")

# Decimal型の変換を無効化する場合
df = db.execute_query("SELECT * FROM table", convert_decimal=False)
```

**特徴**:
- 抽象基底クラス（ABCMeta）として設計
- 接続パラメータの統一管理
- エラーハンドリングとログ出力
- リソース管理（カーソル・コネクションの自動クローズ）

### AISGGibDB（GIB_DB接続）

**役割**: 特定のGIB_DBへの接続実装（読み取り専用）

**実装**:
```python
from gppm.providers.aisg_gib_db import AISGGibDB

# 接続情報は固定（読み取り専用）
db = AISGGibDB()
```

**注意事項**:
- 読み取り専用アクセス
- 接続情報はハードコード（本番環境では環境変数等で管理推奨）
- ネットワーク要件（VPN、IP制限等）の確認が必要

---

## プロバイダーの統合利用

### DataManagerでの統合

各Providerは`FactSetDataManager`を通じて統合的に利用されます：

```python
from gppm.pipeline.data_manager import FactSetDataManager

# DataManagerの初期化
dm = FactSetDataManager()

# 統合データの取得
initial_data = dm.initialize_data()
print(initial_data.keys())
# 出力: dict_keys(['entity', 'financial', 'segment', 'revere', 'rbics_master'])

# 各データの確認
print(f"企業数: {len(initial_data['entity'])}")
print(f"財務データ件数: {len(initial_data['financial'])}")
print(f"セグメントデータ件数: {len(initial_data['segment'])}")
```

### パフォーマンス最適化

**並列処理**:
- 各Providerで`max_workers`パラメータを調整
- 大量データでは`batch_size`の最適化

**キャッシュ戦略**:
- 頻繁に参照されるデータのキャッシュ化
- 期間フィルタの事前適用

**メモリ管理**:
- 大規模データセットの段階的処理
- 不要なデータの早期解放

---

7. 実践レシピ（E2Eサンプル・ユースケース）
------------------------------------------

7.1 最小E2E（CLI×2）
```bash
# 1) 前処理
python -m gppm.cli.prepare_data

# 2) 推論（スキャフォールド）
python -m gppm.cli.bayes_infer \
  --data "/tmp/gppm_output/processed_data/bayesian_dataset.pkl" \
  --engine cmdstanpy --model /path/to/model.stan \
  --iter 1000 --chains 4 --seed 42
```

7.2 Notebookでの確認（例）
```python
import pickle, pandas as pd
data = pickle.load(open('/tmp/gppm_output/processed_data/bayesian_dataset.pkl','rb'))
pd.DataFrame(data['stan_data'].items(), columns=['key','value']).head()
```

7.3 自前の前処理ステップを差し込む
```python
from gppm.bayes.bayesian_data_processor import BayesianDataProcessor
from gppm.pipeline.data_manager import FactSetDataManager

dm = FactSetDataManager()
proc = BayesianDataProcessor(mapping_df_path='/path/to/mapping_df.pkl')

data = proc.load_from_data_manager(dm)
data = proc.filter_data_by_period(data, 201909, 202406)
data = proc.filter_valid_entities(data, na_threshold=0.5)
# ... 独自変換 ...
```

7.4 Stanモデルの最小雛形（イメージ）
```stan
data {
  int<lower=1> Segment_N;
  int<lower=1> Time_N;
  int<lower=1> N_obs;
  array[N_obs] int seg_id;
  array[N_obs] int t_id;
  vector[N_obs] y; // ROIC segment
}
parameters {
  vector[Segment_N] alpha;
}
model {
  y ~ normal(alpha[seg_id], 0.1);
}
```

7.5 パフォーマンス最適化のヒント
- バッチ/並列の調整（`processing.batch_size`, `processing.max_workers`）
- 中間データのキャッシュ化（必要に応じて）
- データ量に合わせた期間絞り込み

---

8. トラブルシューティング（FAQ付き）
------------------------------------

Q. 設定ファイルが見つからない
- A. ルートに `gppm_config.yml` を配置してください（例は本READMEの設定章）。

Q. マッピングファイルが見つからない
- A. `mapping_df_path` を指し示してください。必須列は `FACTSET_ENTITY_ID, PRODUCT_L6_ID, RELABEL_L6_ID`。

Q. 期間が逆（start > end）でエラー
- A. `data.start_period <= data.end_period` となるよう修正してください。

Q. 出力ディレクトリ作成に失敗
- A. `output.base_directory` の権限/空き容量を確認。別ディスクを指定して回避可能。

Q. DB接続に失敗する
- A. VPN・認証・FW設定を確認。`src/gppm/providers/sqlserver.py:1` の接続実装の要件に留意。

Q. Stan推論はどこで行う？
- A. `src/gppm/cli/bayes_infer.py:1` は現在スキャフォールドです。CmdStanPy等の連携を今後追加します。

---

用語集（最小）
--------------

- ベイジアン推論: 事前分布とデータから事後分布を求める枠組み。
- Stan: ベイジアンモデリング言語/コンパイラ。将来的に推論で使用予定。
- Provider: 外部データ取得クラス群（FactSet/RBICS/MSCI/CIQ等）。
- `wolf_period`: 期間（D/W/M/Q/Y）を週開始・会計年度込みで厳密に扱うユーティリティ。
- 前処理（prepare_data）: 生データを学習可能な辞書/行列へ変換する工程。

9. 開発・テスト・スタイル
--------------------------

- Python: 3.13以上
- テスト（存在する場合）: `pytest`（`pyproject.toml:1` 参照）
- スタイル/静的解析: `black`/`ruff`/`mypy` 設定を `pyproject.toml:1` に定義
- ログ取得は `gppm.utils.config_manager.get_logger()` を利用

参考スクリプト
```bash
python -m pytest -q
ruff check src/
black --check src/
mypy src/
```

---

10. ライセンス/サポート
-----------------------

- ライセンス: MIT（`LICENSE:1`）
- 質問/要望: Issues へ

---

付録A. 主要関数のスニペット集
------------------------------

DataManager（`src/gppm/pipeline/data_manager.py:1`）
```python
from gppm.pipeline.data_manager import FactSetDataManager
dm = FactSetDataManager()
initial = dm.initialize_data()
print(initial.keys())  # entity, financial, segment, revere, rbics_master
```

BayesianDataProcessor（`src/gppm/bayes/bayesian_data_processor.py:1`）
```python
from gppm.bayes.bayesian_data_processor import BayesianDataProcessor
bp = BayesianDataProcessor(mapping_df_path='/path/to/mapping.pkl')
data = bp.process_full_pipeline(data_manager=dm, save_path='/tmp/pp.pkl', start_period=201909, end_period=202406)
```

Provider API（FactSet/RBICS/…）
```python
from gppm.providers.factset_provider import FactSetProvider
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

p = FactSetProvider()
pr = WolfPeriodRange(
    WolfPeriod.from_month(2024, 1, freq=Frequency.M),
    WolfPeriod.from_month(2024, 12, freq=Frequency.M),
)
records = p.get_financial_records(period_range=pr, country=["US"], active_only=True)
```

---

付録B. モデル入力の検算Tips
-------------------------

- `stan_data`の各次元の整合（Segment_N × Time_N 等）
- 欠損の埋め方（ffill/bfillの後、観測インデックス化）
- 製品シェアの正規化チェック（各時点での合計）
- 連結とセグメントの時点・エンティティ合わせ（`align_pivot_tables`）

---

付録C. よくある拡張
-------------------

- 追加説明変数の組み込み（地理ウェイト、バリュエーション等）
- 企業×製品の階層化、相関構造の事前分布
- 予測/PPCの自動可視化

---

付録D. 設定パラメータ詳細（リファレンス）
------------------------------------------

log_level
- 種別: str（DEBUG/INFO/WARNING/ERROR/CRITICAL）
- 既定: INFO
- 影響: ログ出力の詳細度

processing.batch_size
- 種別: int (>0)
- 目的: バッチ処理サイズ（メモリ/速度のバランス）

processing.max_workers
- 種別: int (1..32)
- 目的: 並列ワーカー数（I/Oと計算の分散）

data.start_period, data.end_period
- 種別: int（YYYYMM）
- 制約: end >= start
- 目的: 前処理対象期間の絞り込み

output.base_directory
- 種別: 文字列/Path
- 目的: 出力ルート（`processed_data/` が自動付与）

output.dataset_filename
- 種別: str
- 目的: 出力pickle名（例: bayesian_dataset.pkl）

mapping_df_path
- 種別: 文字列/Path/Null
- 必須列: `FACTSET_ENTITY_ID, PRODUCT_L6_ID, RELABEL_L6_ID`
- 目的: 製品マッピング（再ラベル）用データ

---

付録E. CLIオプション（bayes_infer）
----------------------------------

`python -m gppm.cli.bayes_infer --help`

--data PATH
- 前処理pickleへのパス。未指定時は既定 `/tmp/gppm_output/processed_data/bayesian_dataset.pkl` を探索

--engine {cmdstanpy,pystan,numpyro}
- 将来切替予定の推論バックエンド（現状は検証とメタ出力のみ）

--model PATH
- Stanモデルファイル（.stan）。現状は未指定可（検証のみ）

--iter, --chains, --seed
- 反復数・チェーン数・乱数シード

--output PATH
- 推論結果の出力ディレクトリ（メタJSONなど）

---

付録F. 代表的な例外と対処
------------------------

FileNotFoundError（設定ファイル）
- `gppm_config.yml` をルートに配置

FileNotFoundError（マッピングファイル）
- `mapping_df_path`