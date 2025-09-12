# データモニタリング機能

## 概要

データ処理パイプラインでの企業数変化、国別分布、その他の統計情報をwith文を使ったコンテキストマネージャーでモニタリングする機能です。

## 主な機能

### 1. 基本的なモニタリング
- データ数の変化追跡
- 処理時間の測定
- エラーハンドリング

### 2. 列分布モニタリング
- 指定された列の値分布
- 分布の変化追跡
- カスタム列の可視化

### 3. パフォーマンスモニタリング
- 処理時間の詳細測定
- メモリ使用量の追跡（オプション）
- オーバーヘッドの最小化

### 4. 設定可能性
- モニタリングレベルの調整
- データタイプ別の設定
- YAML設定ファイル対応

## 使用方法

### 最もシンプルな使用方法（推奨）

```python
from gppm.core.monitoring import monitor

# 最も簡単な使用方法
with monitor("データ処理") as m:
    # 初期データの設定
    m.set_initial_data(data, entity_id_col="entity_id")
    
    # データ処理
    processed_data = process_data(data)
    
    # 変化の記録
    m.log_count_change(processed_data, "データ処理完了", entity_id_col="entity_id")
```

### 設定を指定した使用方法

```python
from gppm.core.monitoring import monitor, MonitoringConfig, MonitoringLevel

# カスタム設定を作成
config = MonitoringConfig(
    level=MonitoringLevel.DETAILED,
    enable_performance_monitoring=True,
    enable_memory_monitoring=False
)

# 設定を指定してモニタリング
with monitor("データ処理", config=config) as m:
    m.set_initial_data(data, entity_id_col="entity_id")
    processed_data = process_data(data)
    m.log_count_change(processed_data, "データ処理完了", entity_id_col="entity_id")
```

### ロガーも指定した使用方法

```python
from gppm.core.monitoring import monitor, MonitoringConfig, MonitoringLevel
import logging

# 設定とロガーを指定
config = MonitoringConfig(level=MonitoringLevel.DETAILED)
logger = logging.getLogger(__name__)

with monitor("データ処理", config=config, logger=logger) as m:
    m.set_initial_data(data, entity_id_col="entity_id")
    processed_data = process_data(data)
    m.log_count_change(processed_data, "データ処理完了", entity_id_col="entity_id")
```

### 従来の詳細制御が必要な場合

```python
from gppm.core.monitoring import monitor_data_processing
import logging

# ロガーの作成
logger = logging.getLogger("my_processing")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# モニタリングの実行
with monitor_data_processing("データ処理", logger) as monitor:
    # 初期データの設定
    monitor.set_initial_data(data, entity_id_col="entity_id")
    
    # データ処理
    processed_data = process_data(data)
    
    # 変化の記録
    monitor.log_count_change(processed_data, "データ処理完了", entity_id_col="entity_id")
```

### 列分布モニタリング

```python
# シンプルな方法
with monitor("列分布モニタリング") as m:
    m.set_initial_data(
        data, 
        entity_id_col="entity_id",
        track_columns=["country_code", "industry_code"]
    )
    
    # 指定された列の分布が自動的に記録される
    processed_data = process_data(data)
    m.log_count_change(processed_data, "処理完了", entity_id_col="entity_id")
```

### 設定ファイルの使用

```python
from gppm.core.monitoring import load_config_from_file, MonitoringConfig, MonitoringLevel

# 設定ファイルから読み込み
config = load_config_from_file("monitoring_config.yaml")

# または直接設定を作成
config = MonitoringConfig(
    level=MonitoringLevel.DETAILED,
    enable_performance_monitoring=True,
    enable_memory_monitoring=False
)

# 設定に基づいたモニタリング
with monitor("設定ベースモニタリング", config=config) as m:
    # 処理の実行
    pass
```

## 設定ファイル例

```yaml
# monitoring_config.yaml
level: detailed
enable_performance_monitoring: true
enable_memory_monitoring: false

log_file_path: "/tmp/monitoring_log.json"
log_level: INFO

default_track_columns: ["entity_id", "country_code"]
enable_entity_tracking: false

distribution_monitoring:
  top_values: 10
  track_changes: true
```

## モニタリングレベル

### MINIMAL
- 最小限のログ出力
- パフォーマンス影響を最小化
- 基本的なエラー情報のみ

### STANDARD（デフォルト）
- 標準的なログ出力
- 企業数変化の追跡
- 処理時間の記録

### DETAILED
- 詳細なログ出力
- 列分布の記録
- メタデータの保存

### DEBUG
- デバッグ用の全ログ
- メモリ使用量の追跡
- 詳細なエラー情報

## パフォーマンス考慮事項

### オーバーヘッドの最小化
- 非同期ログ出力
- 必要最小限のデータコピー
- 設定可能なモニタリング項目

### メモリ効率性
- データの部分的なコピーのみ
- ストリーミング処理対応
- ガベージコレクションの最適化

### ログファイル管理
- 自動ログローテーション
- ファイルサイズ制限
- エラー時のフォールバック

## エラーハンドリング

### 例外安全性
- モニタリングエラーがメイン処理に影響しない
- ログファイルの破損防止
- 権限エラーの適切な処理

### ログの信頼性
- ログファイルのバックアップ
- 書き込み権限の確認
- JSONエンコードエラーの処理

## テスト

### 単体テストの実行

```bash
python -m pytest src/gppm/utils/test_data_monitoring.py -v
```

### パフォーマンステスト

```bash
python -m pytest src/gppm/utils/test_data_monitoring.py::TestPerformanceImpact -v
```

## トラブルシューティング

### よくある問題

1. **ログファイルが作成されない**
   - ディレクトリの書き込み権限を確認
   - パスの存在を確認

2. **パフォーマンスが低下する**
   - モニタリングレベルをMINIMALに設定
   - 非同期ログ出力を有効化

3. **メモリ使用量が増加する**
   - メモリモニタリングを無効化
   - データコピーを最小限に抑制

4. **列分布が表示されない**
   - track_columnsパラメータで列名を指定
   - 列名がデータフレームに存在することを確認

### デバッグ方法

```python
# デバッグレベルのログを有効化
import logging
logging.getLogger("data_monitoring").setLevel(logging.DEBUG)

# 詳細なモニタリング設定
config = create_detailed_config()
```

## 拡張性

### カスタムメトリクスの追加

```python
# カスタムメトリクスの定義
config.custom_metrics = ["custom_metric_1", "custom_metric_2"]

# メトリクスの記録
monitor.stats.metadata["custom_metric_1"] = calculate_metric()
```

### カスタム設定の作成

```python
from gppm.core.monitoring import monitor, MonitoringConfig, MonitoringLevel

# カスタム設定を作成
config = MonitoringConfig(
    level=MonitoringLevel.DETAILED,
    enable_performance_monitoring=True,
    enable_memory_monitoring=False,
    default_track_columns=["entity_id", "country_code", "industry_code"],
    enable_entity_tracking=False
)

# 設定をYAMLファイルに保存
config.to_yaml("my_custom_config.yaml")

# カスタム設定を使用
with monitor("カスタム設定テスト", config=config) as m:
    m.set_initial_data(data)
    # 処理の実行
    pass
```