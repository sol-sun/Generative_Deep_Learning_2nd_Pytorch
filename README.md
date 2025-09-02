# WolfPeriod / WolfPeriodRange

統一的な期間処理（D/W/M/Q/Y）を提供するユーティリティ群です。WolfPeriod（単一期間）とWolfPeriodRange（期間範囲）で、週開始日・会計年度開始月を明示した型安全な操作を実現します。pandasのPeriodと親和性を保ちつつ、会計年度や週開始の概念を一次元で取り扱えるのが特徴です。

## 目次
1. [概要と設計方針](#概要と設計方針)
2. [インストールと開発環境](#インストールと開発環境)
3. [クイックスタート](#クイックスタート)
4. [APIリファレンス: WolfPeriod](#apiリファレンス-wolfperiod)
5. [APIリファレンス: WolfPeriodRange](#apiリファレンス-wolfperiodrange)
6. [カレンダー設定](#カレンダー設定)
7. [pandas連携とエイリアス](#pandas連携とエイリアス)
8. [エッジケース](#エッジケース)
9. [変換・シリアライゼーション](#変換シリアライゼーション)
10. [等価性・順序・算術の詳細](#等価性順序算術の詳細)
11. [実践レシピ](#実践レシピ)
12. [性能とベストプラクティス](#性能とベストプラクティス)
13. [FAQ](#faq)
14. [テストと品質](#テストと品質)
15. [用語集](#用語集)

---

## 概要と設計方針

### 目的
- 日/週/月/四半期/年（会計年度）の期間を、単一の不変（イミュータブル）オブジェクトで扱う
- 週開始日（Weekday）と会計年度開始月（fy_start_month）を常に保持し、比較や連携時のあいまいさを排除
- pandas Periodとの相互変換を備え、データ分析ツールチェーンと無理なく接続

### 設計の要点
- **型安全**: 列挙型（Frequency, Weekday, QuarterIndex）とpydantic v2の検証で堅牢化
- **明示性**: 週開始・会計年度開始をラベルに埋め込み、人間可読性を重視
- **一貫性**: 比較・差分は「同一モード（freq, week_start, fy_start_month）」でのみ許可
- **イミュータブル**: 値オブジェクトとして扱い、予期せぬ副作用を防止

### なぜWolfPeriodか
- pandas Periodは強力だが、FY開始や週開始が暗黙的になりやすい
- WolfPeriodは FY開始/週開始を「値」に内包し、比較・演算の安全性を保証
- 相互変換が容易で、既存のpandas資産を活かしつつ曖昧さを排除

---

## インストールと開発環境

本モジュールは同一リポジトリのサブパッケージです。プロジェクトルートで開発インストールしてください。

```bash
pip install -e .[dev]
```

### 依存
- pydantic v2
- pandas（連携時）
- Python 3.13+

---

## クイックスタート

### 最小コード（2分）
最小コードで「月次の直近3期間」を列挙します。

```python
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

end = WolfPeriod.from_month(2024, 6, freq=Frequency.M)
rg  = WolfPeriodRange(start=end-2, stop=end)
print([p.label for p in rg])  # ['2024-04','2024-05','2024-06']
```

### 単一期間の作成
```python
from datetime import date
from wolf_period import WolfPeriod, Frequency, Weekday

# 日次
p_d = WolfPeriod.from_day(date(2024, 2, 29))  # うるう年対応
print(p_d.label)  # 2024-02-29

# 週次（週開始=月）
p_w = WolfPeriod.from_week(date(2024, 1, 15), Weekday.MON)
print(p_w.label)  # W[Mon] 2024-01-15 (週の開始日ラベル)

# 月次（freq必須）
p_m = WolfPeriod.from_month(2024, 4, freq=Frequency.M)
print(p_m.label)  # 2024-04

# 四半期（FY=4月開始）
p_q = WolfPeriod.from_quarter(2024, 1, fy_start_month=4)
print(p_q.label)  # FY2024-Q1(開始=04月)

# 会計年度（FY=4月開始）
p_y = WolfPeriod(freq=Frequency.Y, y=2024, fy_start_month=4)
print(p_y.label)  # FY2024(開始=04月)
```

### 算術演算と比較
```python
# 加減算（同一モードのみ）
print(p_m + 1)      # 翌月
print((p_m + 1) - p_m)  # 1

# 比較（同一モードのみ）
assert p_m < (p_m + 1)
```

### 範囲の反復
```python
from wolf_period import WolfPeriodRange

rg = WolfPeriodRange(
    WolfPeriod.from_month(2024, 1, freq=Frequency.M),
    WolfPeriod.from_month(2024, 12, freq=Frequency.M),
)
for p in rg:
    print(p.label)
```

---

## APIリファレンス: WolfPeriod

### クラス定義
- **WolfPeriod**（`src/wolf_period/periods.py`）
  - 不変（イミュータブル）な期間オブジェクト
  - **フィールド**（頻度に応じて使用）：
    - `freq: Frequency` — D/W/M/Q/Y
    - `y: int` — ラベル年（Q/Yは会計年度）
    - `m: Optional[int]` — 月（Mまたはanchor）
    - `d: Optional[int]` — 日（D/Wのanchor）
    - `week_start: Weekday` — 週開始（Wで使用）
    - `fy_start_month: int` — 会計年度開始月（Q/Yで使用）
    - `q: Optional[QuarterIndex]` — 四半期（Qで使用）

### 主なコンストラクタ
- `WolfPeriod.from_day(d: date, *, freq=None, week_start=Weekday.MON, fy_start_month=4)`
- `WolfPeriod.from_week(any_day_in_week: date, week_start=Weekday.MON)`
- `WolfPeriod.from_month(y: int, m: int, *, freq: Frequency, fy_start_month=4)`
- `WolfPeriod.from_yyyymm(yyyymm: int | str, *, freq: Frequency, fy_start_month=4)`
- `WolfPeriod.from_quarter(fy_label: int, q: int|QuarterIndex, fy_start_month=4)`

### プロパティ
- `start_date: date` — 期間の開始日
- `end_date: date` — 期間の終了日
- `label: str` — 人間に読みやすいラベル（週開始/FY開始含む）
- `year: int` — 暦年（開始日の年）
- `fiscal_year: int` — 会計年度ラベル
- `ordinal: int` — 起点（1970）からの序数（モードごとに定義）

### 演算・比較
- `__add__(n: int) -> WolfPeriod` — n期間足す
- `__sub__(n: int|WolfPeriod)` — nなら減算、期間なら差分（モード一致必須）
- `__eq__/__lt__/__le__/__gt__/__ge__` — 順序比較（モード一致必須）

### 変換
- `to_pandas_period()` — pandas Periodに変換（Q/YはFYに応じたエイリアス）
- `to_dict()/from_dict()` — シリアライゼーション

### 例（四半期の境界計算）
```python
from wolf_period import QuarterIndex

# FY=10月開始でFY2024-Q1の期間
p = WolfPeriod.from_quarter(2024, QuarterIndex.Q1, fy_start_month=10)
print(p.start_date, p.end_date)
```

### バリデーション
- 月=1..12、日=1..31、FY開始=1..12、四半期=1..4 を厳密に検証
- 頻度とフィールドの組合せ不整合（例: Mで日付を持つ）は例外

---

## APIリファレンス: WolfPeriodRange

### クラス定義
- **WolfPeriodRange**（`src/wolf_period/ranges.py`）
  - 期間の範囲を効率的にイテレート・操作
  - `start: WolfPeriod`, `stop: Optional[WolfPeriod]`, `step: int`, `count: Optional[int]`

### 構築方法
```python
# 1) start/stop
WolfPeriodRange(start=p1, stop=p2, step=1)
WolfPeriodRange(p1, p2)  # 位置引数

# 2) start/count
WolfPeriodRange(start=p1, count=12, step=1)
```

### 主な機能
- **反復**: `for p in rg: ...`
- **長さ**: `len(rg)`（count または stop が必要）
- **インデックス**: `rg[0]`, `rg[-1]`
- **スライス**: `rg[:3]`, `rg[::2]`
- **包含判定**: `p in rg`（日付やTimestampも可）
- **ラベル**: `rg.labels()`
- **pandas連携**: `rg.to_index()`, `rg.to_frame()`

### 包含判定の挙動
- **WolfPeriod（同一モード）**: `ordinal`を用いた範囲・剰余チェック
- **datetime/date/Timestamp**: 現在モードに合うよう内部でWolfPeriodへ変換した上で判定

### 負のステップ
```python
rg = WolfPeriodRange(
  WolfPeriod.from_month(2024, 12, freq=Frequency.M),
  WolfPeriod.from_month(2024, 1, freq=Frequency.M),
  step=-1,
)
```

---

## カレンダー設定

### CalendarConfig（`types.py`）
- `week_start: Weekday = Weekday.MON`
- `fy_start_month: int = 4`
- `with_week_start(wd)`, `with_fy_start(m)` で派生設定を生成（不変）

### 使用例
```python
from wolf_period import CalendarConfig, Weekday

cfg = CalendarConfig()
cfg_sun = cfg.with_week_start(Weekday.SUN)
cfg_oct = cfg.with_fy_start(10)
```

### 週開始とpandasの関係
- 週次Periodは `W-MON` のようにエイリアスへ反映

### FY開始とpandasの関係
- **四半期**: `Q-<FYの終了月>` 例）FY=4月開始 → 終了月=3月 → `Q-MAR`
- **年次**: `A-<FYの終了月>` 例）FY=4月開始 → `A-MAR`

---

## pandas連携とエイリアス

### 単一期間
```python
import pandas as pd
from wolf_period import WolfPeriod, Frequency

p = WolfPeriod.from_month(2024, 1, freq=Frequency.M)
pp = p.to_pandas_period()
assert isinstance(pp, pd.Period)
```

### 範囲
```python
from wolf_period import WolfPeriodRange

rg = WolfPeriodRange(
  WolfPeriod.from_quarter(2024, 1, fy_start_month=4),
  WolfPeriod.from_quarter(2024, 4, fy_start_month=4),
)
idx = rg.to_index()   # Index(['FY2024-Q1(開始=04月)', ...])
df  = rg.to_frame()   # start/end/label を列に持つ
```

### エイリアスの決定ロジック
- **週次**: `W-<週開始3文字>`
- **四半期**: `Q-<FY終了月略称>`
- **年次**: `A-<FY終了月略称>`

---

## エッジケース

### うるう年
```python
p = WolfPeriod.from_day(date(2024, 2, 29))
print(p.start_date, p.end_date)  # 同日

pm = WolfPeriod.from_month(2024, 2, freq=Frequency.M)
print(pm.end_date.day)  # 29
```

### 週開始の違い
```python
WolfPeriod.from_week(date(2024, 1, 15), Weekday.SUN).label  # W[Sun] 2024-01-14
WolfPeriod.from_week(date(2024, 1, 15), Weekday.FRI).label  # W[Fri] 2024-01-12
```

### FYバウンダリ（4月/10月/1月）
```python
WolfPeriod.from_quarter(2024, 1, fy_start_month=4).label   # FY2024-Q1(開始=04月)
WolfPeriod.from_quarter(2024, 1, fy_start_month=10).label  # FY2024-Q1(開始=10月)
WolfPeriod.from_quarter(2024, 1, fy_start_month=1).label   # FY2024-Q1(開始=01月)
```

---

## 変換・シリアライゼーション

### 辞書変換
```python
from wolf_period import WolfPeriod, Frequency

p = WolfPeriod.from_month(2024, 1, freq=Frequency.M)
d = p.to_dict()
restored = WolfPeriod.from_dict(d)
```

### YYYYMM変換
```python
from wolf_period import WolfPeriod, Frequency

p = WolfPeriod.from_yyyymm(202404, freq=Frequency.M)
assert p.label == '2024-04'
```

---

## 等価性・順序・算術の詳細

### 等価性と順序
- 比較対象は同一モード（freq, week_start, fy_start_month）でなければならない
- 違うモード間の比較は例外（安全性のため）

### 差分計算
```python
# 月次の差分は暦の月数差
WolfPeriod.from_month(2024, 4, freq=Frequency.M) - WolfPeriod.from_month(2024, 1, freq=Frequency.M)  # 3

# 四半期の差分はFY基準の四半期数
WolfPeriod.from_quarter(2024, 4, 4) - WolfPeriod.from_quarter(2024, 1, 4)  # 3
```

### 序数（ordinal）
- **D**: 1970-01-01 からの日数
- **W**: 1970-01-01 を含む週の `week_start` 基準の序数
- **M**: 1970年1月を原点とする月序数
- **Q**: 1970年のQ1を原点とする四半期序数（FY基準）
- **Y**: 1970年を原点とする会計年度序数

---

## 実践レシピ

### 直近12ヶ月のラベル列
```python
import pandas as pd
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency

end = WolfPeriod.from_month(2024, 12, freq=Frequency.M)
rg = WolfPeriodRange(end - 11, end)
labels = rg.labels()
```

### 月次→四半期の集計キー生成
```python
def month_to_quarter(y, m, fy_start_month=4):
    from wolf_period import WolfPeriod, Frequency
    p = WolfPeriod.from_month(y, m, freq=Frequency.Q, fy_start_month=fy_start_month)
    return p.label
```

### 任意日付配列を週次キーへ
```python
from datetime import date
from wolf_period import WolfPeriod, Weekday

def to_week_label(dates, week_start=Weekday.MON):
    return [WolfPeriod.from_week(d, week_start).label for d in dates]
```

### pandas Periodからの移行
```python
import pandas as pd
from wolf_period import WolfPeriod, Frequency

pp = pd.Period('2024-03', freq='M')
p = WolfPeriod.from_month(pp.year, pp.month, freq=Frequency.M)
```

---

## 性能とベストプラクティス

- 値オブジェクトを集合・辞書キーに使う場合、モードを揃える
- 範囲生成では `count` を優先すると明示的で安全
- pandas連携は必要時のみ（大規模ループでは最小化）

---

## FAQ

### よくある質問

**Q. `from_month` で `freq` を省略できますか？**
- A. 省略できません。`Frequency.M/Q/Y` を必ず指定してください。

**Q. pandasの `Q-`/`A-` エイリアスはどう決まりますか？**
- A. `fy_start_month` から終了月を計算し、その略称を使います（例: FY=4月開始→終了=3月→`Q-MAR`）。

**Q. 範囲に `date` や `Timestamp` は使えますか？**
- A. 使えます。内部で現在のモードに合わせて `WolfPeriod` に変換してから判定します。

**Q. 比較でモードが異なるとどうなりますか？**
- A. 例外を投げます。モードを明示的に揃えてください。

**Q. 週開始やFY開始を変えた場合の表示は？**
- A. ラベルに `W[Sun]` や `開始=10月` など明示されます。

---

## テストと品質

### 推奨テスト
- うるう年の境界
- FYバウンダリ（1/4/10月開始）
- 週開始違いによるラベル比較
- 差分計算の妥当性
- pandasへの往復変換

### スタイル/型
- `pyproject.toml` に `black`/`ruff`/`mypy` 設定

---

## 用語集

- **Frequency**: D/W/M/Q/Y の列挙
- **Weekday**: MON..SUN の列挙
- **QuarterIndex**: Q1..Q4 の列挙（柔軟な生成）
- **CalendarConfig**: 週開始とFY開始の設定
- **ordinal**: 序数（起点からの整数）

---

## 付録

### pandasとの往復変換の落とし穴
- pandas Periodの四半期/年次は、デフォルトで暦年ベースの`Q-DEC`/`A-DEC`になりがち
- 会計年度ベースの分析では、WolfPeriodを原本としてpandasへの変換時にFY終了月を反映（`Q-MAR`/`A-MAR`等）
- 逆変換時（pandas→WolfPeriod）は、FY開始月を明示して対応づけること

### ISO週との関係
- WolfPeriodはISO週番号を直接は扱いませんが、`week_start`を`MON`にして`from_week`すれば、多くの分析で代替可能です
- ただしISO週の年境界（第1週が前年末にかかる等）を厳密に扱う必要がある場合は、別途ISO週変換の補助関数を用意してください

### 既知の制約
- 週次/WolfPeriodRangeでの大規模イテレーションは、用途に応じてステップ幅を見直してください（例: 大量データでは月次・四半期へ集約）
- 異なるモードの期間比較は例外になります。安全性のためであり、混在を許す設計ではありません

### 比較: pandas Period vs WolfPeriod
- **pandas Period**: 強力なリサンプリング/時系列機能。FYや週開始の取扱いはやや抽象的
- **WolfPeriod**: FY・週開始を一次元上に固定し、明示的・型安全に扱える。Periodとの相互変換で双方の長所を活かす設計
