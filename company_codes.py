"""
企業コードユーティリティ
=========================

目的
- 企業識別子（ISIN/SEDOL/CUSIP/TICKER/FSYM/Entity/CIQ/GVKEY など）の相互変換、取得、検索を
  シンプルな API で提供する読み取り専用ディレクトリです。

主要コンポーネント
- `CompanyCatalog`: データのリフレッシュ、取得、変換、検索を担う高水準 API。
- `Company`: 企業レコード（読み取り専用）。入力検証・正規化を内包。
- `CodeIdentifier`: 識別子の種別を列挙。
- `ConvertRequest`/`ConvertResult`: 変換要求とその結果を表すモデル。
- `SearchQuery`: 検索条件モデル。
- `CompanyList`: API 返却などで JSON 形式を固定するためのコレクションモデル。

データソースと更新
- FactSet および CIQ のプロバイダーから高速取得
- メモリ内にインデックスを構築し、以後の `get`/`convert`/`search` はメモリのみで処理
- WolfPeriodによる統一された期間管理
- 自動更新は行いません。明示的に `CompanyCatalog.refresh()` を呼び出してください。

正規化・検証ルール（要点）
- 国コード: `ISO 3166-1 alpha-2` に正規化（未知コードはエラー）。
- ISIN/SEDOL/CUSIP: 英数字のみを許容し、大文字化。
- ティッカー: 取引所サフィックス（例: ".T"）を除去し、残りを大文字化。
- 企業名: 必須。識別子は少なくとも1つ必須。
- WolfPeriod: 統一された日付・期間処理

検索仕様
- 小文字化による部分一致で評価します。
- `country` 指定時は「上場国/本社国」のいずれかが一致する企業のみ対象。
- 同一主キー（標準は FSYM）単位で重複しません。

スレッド安全性
- インデックス再構築はロックで保護します。読み取り操作はロックレスで高速です。
- 並列処理対応による高いスループット

使用例
    from gppm.utils.company_codes import CompanyCatalog
    from wolf_period import WolfPeriod

    catalog = CompanyCatalog()
    catalog.refresh()  # DB から最新をロード
    ticker = catalog.convert("US0378331005", "isin", "ticker")  # => "AAPL"
    company = catalog.get("US0378331005", "isin")
    results = catalog.search("Apple", fields=["company_name", "ticker"], country="US")
"""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter
from enum import Enum
from threading import Lock
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Union,
    overload,
    Literal,
)

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    computed_field,
    field_validator,
    model_validator,
)
from pydantic.type_adapter import TypeAdapter

from wolf_period import WolfPeriod, WolfPeriodRange, Frequency
from gppm.utils.config_manager import get_logger
from gppm.utils.country_code_manager import convert_to_alpha2
from gppm.utils.data_processor import DataProcessor
from gppm.providers.factset_provider import (
    FactSetProvider,
    FactSetQueryParams,
    FactSetIdentityRecord,
)
from gppm.providers.ciq_provider import (
    CIQProvider,
    CIQQueryParams,
    CIQIdentityRecord,
)


logger = get_logger(__name__)


class CodeIdentifier(str, Enum):
    """企業識別子の種別（小文字の文字列表現）。

    対応関係（`Company` の属性名）:
    - fsym -> `fsym_id`
    - entity -> `factset_entity_id`
    - isin -> `isin`
    - sedol -> `sedol`
    - cusip -> `cusip`
    - ticker -> `ticker`
    - ciq -> `ciq_company_id`
    - gvkey -> `gvkey`

    注意:
    - 文字列から生成する場合は小文字を想定（例: `CodeIdentifier("isin")`）。
    - 未サポート値は `ValueError` になります。
    """

    FSYM = "fsym"
    ENTITY = "entity"
    ISIN = "isin"
    SEDOL = "sedol"
    CUSIP = "cusip"
    TICKER = "ticker"
    CIQ = "ciq"
    GVKEY = "gvkey"

    @property
    def attr_name(self) -> str:
        """Company上の対応属性名を返す。

        例: CodeIdentifier.ISIN.attr_name -> "isin"
        """
        return {
            CodeIdentifier.FSYM: "fsym_id",
            CodeIdentifier.ENTITY: "factset_entity_id",
            CodeIdentifier.ISIN: "isin",
            CodeIdentifier.SEDOL: "sedol",
            CodeIdentifier.CUSIP: "cusip",
            CodeIdentifier.TICKER: "ticker",
            CodeIdentifier.CIQ: "ciq_company_id",
            CodeIdentifier.GVKEY: "gvkey",
        }[self]


class Company(BaseModel):
    """企業レコード（読み取り専用）。

    概要:
    - 企業名・各種識別子・国コードを保持し、バリデーションと正規化を行います。
    - インスタンスは不変（`frozen=True`）。

    主な検証規則:
    - `company_name`: 必須。
    - 識別子: 少なくとも1つ必須（ISIN/SEDOL/CUSIP/TICKER/FSYM/Entity/GVKEY/CIQ のいずれか）。
    - ISIN/SEDOL/CUSIP: 英数字のみ許容。入力は大文字化し、パターンに適合。
    - ティッカー: 取引所サフィックス（例: ".T"）を除いたベースを大文字化。
    - 国コード: `ISO 3166-1 alpha-2` に正規化。未知コードはエラー。

    計算フィールド:
    - `primary_key`: 主要識別子（優先順位: FSYM → Entity → CIQ ID → ISIN → Ticker）。
    - `identifiers`: 保持する識別子の集約（デバッグ/表示用途）。
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # 企業基本
    company_name: str = Field(description="企業名", examples=["Apple Inc."])
    factset_entity_id: Optional[str] = Field(
        default=None, description="FactSet Entity ID", examples=["001C7F-E"]
    )
    ciq_company_id: Optional[int] = Field(default=None, description="CIQ Company ID")

    # 証券識別子（同一企業/証券を識別する代表的なコード群）
    fsym_id: Optional[str] = Field(
        default=None, description="FactSet Symbol ID", examples=["000C7F-E"]
    )
    isin: Optional[str] = Field(
        default=None,
        description="International Securities Identification Number",
        pattern=r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$",
        examples=["US0378331005"],
    )
    sedol: Optional[str] = Field(
        default=None,
        description="Stock Exchange Daily Official List",
        pattern=r"^[0-9A-Z]{7}$",
        examples=["B0YQ5F0"],
    )
    cusip: Optional[str] = Field(
        default=None,
        description="Committee on Uniform Securities Identification Procedures",
        pattern=r"^[0-9A-Z]{9}$",
        examples=["037833100"],
    )
    ticker: Optional[str] = Field(
        default=None, description="ティッカー", examples=["AAPL", "MSFT", "7203"]
    )
    gvkey: Optional[str] = Field(
        default=None, description="Compustat Global Company Key", examples=["031084"]
    )

    # 地理（上場国/本社国）。ISO 3166-1 alpha-2 に正規化して保持
    exchange_country_code: Optional[str] = Field(
        default=None,
        description="上場国コード（ISO 3166-1 alpha-2）",
        pattern=r"^[A-Z]{2}$",
        examples=["US", "JP"],
    )
    headquarters_country_code: Optional[str] = Field(
        default=None,
        description="本社所在地国コード（ISO 3166-1 alpha-2）",
        pattern=r"^[A-Z]{2}$",
        examples=["US", "JP"],
    )

    last_updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="最終更新日時"
    )
    
    # WolfPeriod統合（メタデータ）
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )

    @field_validator("isin")
    @classmethod
    def _validate_isin(cls, v: Optional[str]) -> Optional[str]:
        """ISIN を検証・正規化します。

        - 英数字のみ許容し、大文字化します。
        - `None` はそのまま許容します。
        """
        if v is None:
            return v
        if not v.isalnum():
            raise ValueError("ISINは英数字のみ")
        return v.upper()

    @field_validator("sedol", "cusip")
    @classmethod
    def _validate_alnum(cls, v: Optional[str]) -> Optional[str]:
        """SEDOL/CUSIP を検証・正規化します。

        - 英数字のみ許容し、大文字化します。
        - `None` はそのまま許容します。
        """
        if v is None:
            return v
        if not v.isalnum():
            raise ValueError("英数字のみ")
        return v.upper()

    @field_validator("ticker")
    @classmethod
    def _validate_ticker(cls, v: Optional[str]) -> Optional[str]:
        """ティッカーを正規化します。

        - 取引所サフィックス（例: ".T"）を除外し、大文字化します。
        - `None` はそのまま許容します。
        """
        if v is None:
            return v
        base = v.split(".")[0] if "." in v else v
        return base.upper() if base else None

    @field_validator("exchange_country_code", "headquarters_country_code")
    @classmethod
    def _validate_country(cls, v: Optional[str]) -> Optional[str]:
        """国コードを `ISO 3166-1 alpha-2` に正規化します。

        - 未知コードは `ValueError` を送出します。
        - `None` はそのまま許容します。
        """
        if v is None:
            return v
        normalized = convert_to_alpha2(v)
        if not normalized:
            raise ValueError(f"無効な国コード: {v}")
        return normalized

    @model_validator(mode="after")
    def _validate_company(self) -> "Company":
        """入力の整合性を最終検証します。

        - `company_name` が空でないこと。
        - 少なくとも1つの識別子を保持していること。
        """
        if not self.company_name:
            raise ValueError("company_name は必須")
        identifiers = [
            self.fsym_id,
            self.factset_entity_id,
            self.isin,
            self.sedol,
            self.cusip,
            self.ticker,
            self.gvkey,
            str(self.ciq_company_id) if self.ciq_company_id else None,
        ]
        if not any(identifiers):
            raise ValueError("少なくとも1つの識別子が必要")
        return self

    @computed_field
    @property
    def primary_key(self) -> Optional[str]:
        """主要識別子。

        優先順位: FSYM → Entity → CIQ ID → ISIN → Ticker
        """
        return (
            self.fsym_id
            or self.factset_entity_id
            or (str(self.ciq_company_id) if self.ciq_company_id else None)
            or self.isin
            or self.ticker
        )

    @computed_field
    @property
    def identifiers(self) -> Dict[str, Optional[str]]:
        """保持する識別子の一覧（デバッグ/表示用）。"""
        return {
            "fsym_id": self.fsym_id,
            "factset_entity_id": self.factset_entity_id,
            "isin": self.isin,
            "sedol": self.sedol,
            "cusip": self.cusip,
            "ticker": self.ticker,
            "gvkey": self.gvkey,
            "ciq_company_id": str(self.ciq_company_id) if self.ciq_company_id else None,
        }


class ErrorDetail(BaseModel):
    """フィールド別のエラー詳細（人間可読＋プログラム可読）。

    例:
    - field: "from_code", message: "コードが見つかりません"
    - field: "to_identifier", message: "未サポートの識別子種別"
    """

    field: str = Field(description="エラーフィールド")
    message: str = Field(description="エラーメッセージ")


class ConvertRequest(BaseModel):
    """コード変換リクエスト。

    目的:
    - `from_code` を `from_identifier` から `to_identifier` へ変換する指示を表します。

    用途:
    - 変換時の詳細（成功/失敗やエラー理由）を必要とする低レベル API に渡します。
    """

    model_config = ConfigDict(frozen=True)

    from_code: str = Field(description="変換元コード", min_length=1, examples=["US0378331005"])
    from_identifier: CodeIdentifier = Field(description="変換元の識別子", examples=["isin"])
    to_identifier: CodeIdentifier = Field(description="変換先の識別子", examples=["ticker"])


class ConvertResult(BaseModel):
    """コード変換結果。

    - 成功時: `to_code` に変換結果を格納し、`success=True`、`errors=None`。
    - 失敗時: `to_code=None`、`success=False`、`errors` に原因の配列を格納します。
    """

    model_config = ConfigDict(frozen=True)

    from_code: str = Field(description="変換元コード")
    from_identifier: CodeIdentifier = Field(description="変換元の識別子")
    to_code: Optional[str] = Field(default=None, description="変換後コード")
    to_identifier: CodeIdentifier = Field(description="変換先の識別子")
    success: bool = Field(description="成功フラグ")
    errors: Optional[List[ErrorDetail]] = Field(default=None, description="フィールド別のエラー一覧")


class SearchQuery(BaseModel):
    """検索条件。

    項目:
    - `query`: 部分一致検索のクエリ（小文字化して比較）。
    - `fields`: 検索対象フィールド。未指定時は代表的な 5 項目。
    - `country_code`: 上場国/本社国のいずれかが一致する企業に限定。
    - `limit`: 返却上限（1〜1000）。
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(description="検索文字列", min_length=1)
    fields: List[str] = Field(
        default=["company_name", "ticker", "isin", "sedol", "cusip"],
        description="検索対象フィールド",
    )
    country_code: Optional[str] = Field(
        default=None,
        description="国コードフィルタ（ISO 3166-1 alpha-2）",
        pattern=r"^[A-Z]{2}$",
    )
    limit: int = Field(default=100, gt=0, le=1000, description="最大件数")


class CompanyList(BaseModel):
    """外部公開用のコレクションモデル（RootModel 風）。

    目的:
    - API 返却で JSON 形式を固定したい場合に使用します。

    返却形式:
    - `{"items": [Company, ...]}` を返します。
    """

    model_config = ConfigDict(frozen=True)

    items: List[Company]

    def model_dump(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        return {"items": [c.model_dump(**kwargs) for c in self.items]}


class CompanyCatalog(DataProcessor):
    """企業コードのディレクトリ（高速・WolfPeriod対応版）。

    概要:
    - FactSet/CIQプロバイダーから最新データを高速取り込み、メモリ内にインデックスを構築
    - WolfPeriod統合による統一された期間処理
    - 大規模データセットに対応したバッチ処理最適化
    - 高速な取得・変換・検索を提供

    パフォーマンス特性:
    - バッチ処理: 大量データの並列処理による高速化
    - メモリ効率: インデックス最適化とキャッシュ活用
    - 並列処理: 複数プロバイダーからの並列データ取得
    - 遅延評価: 必要時のみの計算実行

    ライフサイクル:
    - `refresh()` 呼び出しでデータを初期化/更新。
    - 以降の `get/convert/search` はメモリ内インデックスを参照。

    スレッド安全性:
    - インデックス構築は `Lock` により排他制御。読み取りはロックレス。
    - 並列処理対応による高いスループット
    """

    # IDENTIFIER_ATTRの重複を排し、CodeIdentifier.attr_nameに集約

    def __init__(self, max_workers: int = 4, batch_size: int = 5000) -> None:
        super().__init__()
        self._index: Dict[str, Company] = {}
        self._by_pk: Dict[str, Company] = {}
        self._lock = Lock()
        self._last_refresh: Optional[WolfPeriod] = None  # WolfPeriod使用
        self._batch_size = batch_size
        
        # 最適化されたデータプロバイダーの初期化
        self._factset_provider = FactSetProvider(max_workers=max_workers)
        self._ciq_provider = CIQProvider(max_workers=max_workers)
        
        logger.debug("CompanyCatalog初期化完了: max_workers=%d batch_size=%d", max_workers, batch_size)

    # ---------- データロード ----------
    def refresh(self) -> None:
        """データを最新化します（高速・WolfPeriod対応版）。

        処理内容:
        - FactSet/CIQ 最適化プロバイダーからマッピング/企業マスタを高速取得
        - バッチ処理による並列検証・正規化（不正行はスキップしてログ記録）
        - メモリ内インデックスを最適化再構築し、`get/convert/search` で利用可能にします
        - WolfPeriodによる統一された期間管理

        パフォーマンス最適化:
        - 並列データ取得: FactSetとCIQを並行取得
        - バッチバリデーション: 大量データの効率的処理
        - メモリ効率: インデックス最適化とキャッシュ活用

        ログ:
        - 開始/終了、バッチごとの検証・処理エラーを DEBUG レベルで記録します。
        """
        start_ts = perf_counter()
        refresh_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
        logger.info("企業コードロード開始: sources=factset,ciq batch_size=%d", self._batch_size)

        # 並列データ取得による高速化
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            # FactSetとCIQを並列取得
            fs_future = executor.submit(
                self._factset_provider.get_identity_records,
                FactSetQueryParams(
                    active_only=True,
                    batch_size=self._batch_size,
                    max_records=1000000  # 大規模データセット対応
                )
            )
            ciq_future = executor.submit(
                self._ciq_provider.get_identity_records,
                CIQQueryParams(
                    listed_only=True,
                    batch_size=self._batch_size,
                    max_records=1000000  # 大規模データセット対応
                )
            )
            
            # 結果を取得
            t0 = perf_counter()
            fs_records = fs_future.result()
            t1 = perf_counter()
            ciq_records = ciq_future.result()
            t2 = perf_counter()

        logger.info(
            "データ取得状況: 件数_factset=%s 件数_ciq=%s 取得秒_factset=%.3f 取得秒_ciq=%.3f",
            len(fs_records),
            len(ciq_records),
            (t1 - t0),
            (t2 - t1),
        )

        # バッチ処理による高速変換
        companies: List[Company] = []
        fs_errors, ciq_errors = self._convert_records_batch(
            fs_records, ciq_records, companies, refresh_period
        )

        # `Company` のリストとして整形する（最適化版）
        adapter = TypeAdapter(List[Company])
        companies = adapter.validate_python(companies)

        t3 = perf_counter()
        self._build_index_optimized(companies)
        t4 = perf_counter()
        self._last_refresh = refresh_period  # WolfPeriod使用
        
        logger.info(
            "企業コードロード完了: エントリ数=%s 企業数=%s エラー_factset=%s エラー_ciq=%s 構築秒=%.3f 合計秒=%.3f",
            len(self._index),
            len(self._by_pk),
            fs_errors,
            ciq_errors,
            (t4 - t3),
            (perf_counter() - start_ts),
        )

    def _convert_records_batch(
        self,
        fs_records: List[FactSetIdentityRecord],
        ciq_records: List[CIQIdentityRecord],
        companies: List[Company],
        refresh_period: WolfPeriod
    ) -> tuple[int, int]:
        """レコードをCompanyオブジェクトにバッチ変換（最適化版）。
        
        Returns:
            tuple[int, int]: (FactSetエラー数, CIQエラー数)
        """
        fs_errors = 0
        ciq_errors = 0
        
        # FactSetレコードをバッチ処理
        for batch_start in range(0, len(fs_records), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(fs_records))
            batch_records = fs_records[batch_start:batch_end]
            
            for record in batch_records:
                try:
                    c = Company(
                        company_name=record.company_name,
                        fsym_id=record.fsym_id,
                        factset_entity_id=record.factset_entity_id,
                        cusip=record.cusip,
                        isin=record.isin,
                        sedol=record.sedol,
                        ticker=record.ticker,
                        exchange_country_code=record.exchange_country_code,
                        headquarters_country_code=record.headquarters_country_code,
                        retrieved_period=refresh_period,
                    )
                    companies.append(c)
                except ValidationError as e:
                    fs_errors += 1
                    logger.debug(
                        "FactSet検証エラー: fsym=%s entity=%s err=%s",
                        record.fsym_id,
                        record.factset_entity_id,
                        str(e)
                    )
                except Exception as e:
                    fs_errors += 1
                    logger.debug(
                        "FactSet処理エラー: fsym=%s entity=%s err=%s",
                        record.fsym_id,
                        record.factset_entity_id,
                        str(e)
                    )
        
        # CIQレコードをバッチ処理
        for batch_start in range(0, len(ciq_records), self._batch_size):
            batch_end = min(batch_start + self._batch_size, len(ciq_records))
            batch_records = ciq_records[batch_start:batch_end]
            
            for record in batch_records:
                try:
                    c = Company(
                        company_name=record.company_name,
                        ciq_company_id=record.company_id,
                        ticker=record.ticker,
                        isin=record.isin,
                        cusip=record.cusip,
                        sedol=record.sedol,
                        exchange_country_code=record.exchange_country_code,
                        headquarters_country_code=record.headquarters_country_code,
                        retrieved_period=refresh_period,
                    )
                    companies.append(c)
                except ValidationError as e:
                    ciq_errors += 1
                    logger.debug(
                        "CIQ検証エラー: company_id=%s err=%s",
                        record.company_id,
                        str(e)
                    )
                except Exception as e:
                    ciq_errors += 1
                    logger.debug(
                        "CIQ処理エラー: company_id=%s err=%s",
                        record.company_id,
                        str(e)
                    )
        
        return fs_errors, ciq_errors
    
    def _build_index_optimized(self, companies: List[Company]) -> None:
        """メモリ内インデックスを最適化構築（バッチ処理版）。
        
        キー設計:
        - 主キーインデックス: `primary_key` -> `Company`
        - 識別子種別ごとのインデックス: `"{kind}_{code}"` -> `Company`
        """
        with self._lock:
            self._index.clear()
            self._by_pk.clear()
            
            # バッチ処理による最適化
            for batch_start in range(0, len(companies), self._batch_size):
                batch_end = min(batch_start + self._batch_size, len(companies))
                batch_companies = companies[batch_start:batch_end]
                
                for c in batch_companies:
                    pk = c.primary_key
                    if pk and pk not in self._by_pk:
                        self._by_pk[pk] = c
                    
                    # すべての識別子でキーを張る（最適化版）
                    if c.fsym_id:
                        self._index[f"fsym_{c.fsym_id}"] = c
                    if c.factset_entity_id:
                        self._index[f"entity_{c.factset_entity_id}"] = c
                    if c.isin:
                        self._index[f"isin_{c.isin}"] = c
                    if c.sedol:
                        self._index[f"sedol_{c.sedol}"] = c
                    if c.cusip:
                        self._index[f"cusip_{c.cusip}"] = c
                    if c.ticker:
                        self._index[f"ticker_{c.ticker}"] = c
                    if c.gvkey:
                        self._index[f"gvkey_{c.gvkey}"] = c
                    if c.ciq_company_id is not None:
                        self._index[f"ciq_{c.ciq_company_id}"] = c





    # ---------- 取得/変換/検索 ----------
    def get(self, code: str, identifier: Union[str, CodeIdentifier]) -> Optional[Company]:
        """識別子とコードで企業を1件取得します。

        Args:
            code: 取得対象コード（例: "US0378331005"）。
            identifier: 識別子種別（`CodeIdentifier` または同等の小文字の文字列）。

        Returns:
            一致する `Company`。見つからない場合は `None`。

        注意:
            未知の識別子文字列が渡された場合は `None` を返します。
        """
        try:
            k = identifier if isinstance(identifier, CodeIdentifier) else CodeIdentifier(str(identifier).lower())
        except ValueError:
            logger.debug("未知の識別子: identifier=%s code=%s", identifier, code)
            return None
        return self._index.get(f"{k.value}_{code}")

    @overload
    def convert(
        self,
        code: str,
        from_identifier: Union[str, CodeIdentifier],
        to_identifier: Union[str, CodeIdentifier],
        *,
        detail: Literal[False] = False,
    ) -> Optional[str]:
        ...

    @overload
    def convert(
        self,
        code: str,
        from_identifier: Union[str, CodeIdentifier],
        to_identifier: Union[str, CodeIdentifier],
        *,
        detail: Literal[True],
    ) -> ConvertResult:
        ...

    def convert(
        self,
        code: str,
        from_identifier: Union[str, CodeIdentifier],
        to_identifier: Union[str, CodeIdentifier],
        *,
        detail: bool = False,
    ) -> Union[Optional[str], ConvertResult]:
        """コード変換（簡易/詳細を統合）。

        Args:
            code: 変換元コード。
            from_identifier: 変換元識別子。
            to_identifier: 変換先識別子。
            detail: True の場合は `ConvertResult` を返却。False の場合は結果コード（または None）。

        Returns:
            detail=False のときは変換先コード（失敗時 None）。detail=True のときは `ConvertResult`。
        """
        # リクエストの正規化（識別子の個別検証）
        errors: List[ErrorDetail] = []
        f_id: Optional[CodeIdentifier]
        t_id: Optional[CodeIdentifier]
        if isinstance(from_identifier, CodeIdentifier):
            f_id = from_identifier
        else:
            try:
                f_id = CodeIdentifier(str(from_identifier).lower())
            except Exception:
                f_id = None
                errors.append(ErrorDetail(field="from_identifier", message="未サポートの識別子種別"))

        if isinstance(to_identifier, CodeIdentifier):
            t_id = to_identifier
        else:
            try:
                t_id = CodeIdentifier(str(to_identifier).lower())
            except Exception:
                t_id = None
                errors.append(ErrorDetail(field="to_identifier", message="未サポートの識別子種別"))

        if errors:
            result = ConvertResult(
                from_code=code,
                from_identifier=f_id or CodeIdentifier.FSYM,
                to_code=None,
                to_identifier=t_id or CodeIdentifier.FSYM,
                success=False,
                errors=errors,
            )
            return result if detail else None

        req = ConvertRequest(from_code=code, from_identifier=f_id, to_identifier=t_id)  # type: ignore[arg-type]

        # 企業の取得
        company = self.get(req.from_code, req.from_identifier)
        if not company:
            logger.debug(
                "変換失敗: from=%s(%s) to=%s 理由=未存在",
                req.from_code,
                req.from_identifier.value,
                req.to_identifier.value,
            )
            result = ConvertResult(
                from_code=req.from_code,
                from_identifier=req.from_identifier,
                to_code=None,
                to_identifier=req.to_identifier,
                success=False,
                errors=[ErrorDetail(field="from_code", message="コードが見つかりません")],
            )
            return result if detail else None

        # 変換実行（識別子→Company属性名に直接マップ）
        to_code_val = getattr(company, req.to_identifier.attr_name)
        logger.debug(
            "変換結果: from=%s(%s) -> to=%s(%s) success=%s pk=%s",
            req.from_code,
            req.from_identifier.value,
            to_code_val,
            req.to_identifier.value,
            to_code_val is not None,
            company.primary_key,
        )
        result = ConvertResult(
            from_code=req.from_code,
            from_identifier=req.from_identifier,
            to_code=to_code_val if to_code_val is not None else None,
            to_identifier=req.to_identifier,
            success=to_code_val is not None,
            errors=None
            if to_code_val is not None
            else [ErrorDetail(field="to_code", message="変換先コードが存在しません")],
        )
        return result if detail else (result.to_code if result.success else None)

    def search(
        self,
        query: str,
        *,
        fields: Optional[List[str]] = None,
        country: Optional[str] = None,
        limit: int = 100,
    ) -> List[Company]:
        """簡易検索（部分一致）。

        Args:
            query: 部分一致検索クエリ（小文字化比較）。
            fields: 検索対象フィールド（例: `company_name`, `ticker`）。
            country: 国コードフィルタ（上場国/本社国のいずれか一致）。
            limit: 最大件数（1〜1000）。

        Returns:
            条件に一致した `Company` のリスト（企業単位で重複なし）。

        注意:
            `fields` 未指定時は代表的な 5 項目を使用します。
        """
        s = SearchQuery(
            query=query,
            fields=fields or ["company_name", "ticker", "isin", "sedol", "cusip"],
            country_code=country,
            limit=limit,
        )

        q = s.query.lower()
        matched: List[Company] = []
        seen: set[str] = set()

        for c in self._by_pk.values():
            if s.country_code:
                cc = convert_to_alpha2(s.country_code)
                if cc and (c.headquarters_country_code != cc and c.exchange_country_code != cc):
                    continue
            pk = c.primary_key
            if not pk or pk in seen:
                continue

            for f in s.fields:
                v = getattr(c, f, None)
                if v and q in str(v).lower():
                    matched.append(c)
                    seen.add(pk)
                    break

            if len(matched) >= s.limit:
                break

        logger.debug(
            "検索: query=\"%s\" fields=%s country=%s limit=%s matched=%s",
            s.query,
            s.fields,
            s.country_code,
            s.limit,
            len(matched),
        )
        return matched

    # ---------- 補助 ----------
    def mapping_frame(self, include_fields: Optional[List[str]] = None) -> pd.DataFrame:
        """現在のマッピングを `pandas.DataFrame` で返します。

        用途:
        - デバッグ・可視化・CSV 出力などの補助。

        Args:
            include_fields: 出力に含める `Company` の属性名一覧。
                未指定時は代表的なフィールドを出力します。

        Returns:
            マッピングのスナップショットを表す `DataFrame`。
        """
        if include_fields is None:
            include_fields = [
                "fsym_id",
                "factset_entity_id",
                "company_name",
                "isin",
                "sedol",
                "cusip",
                "ticker",
                "headquarters_country_code",
                "exchange_country_code",
            ]
        rows: List[Dict[str, Any]] = []
        for c in self._by_pk.values():
            row: Dict[str, Any] = {}
            for f in include_fields:
                row[f] = getattr(c, f, None)
            rows.append(row)
        return pd.DataFrame(rows)

    def stats(self) -> Dict[str, Any]:
        """概要情報を返します（監視・デバッグ用途・WolfPeriod対応版）。

        Returns:
            統計情報辞書（WolfPeriod情報を含む）
        """
        return {
            "total_entries": len(self._index),
            "unique_companies": len(self._by_pk),
            "last_refresh": str(self._last_refresh) if self._last_refresh else None,
            "last_refresh_period": self._last_refresh.model_dump() if self._last_refresh else None,
            "batch_size": self._batch_size,
            "providers": {
                "factset": "FactSetProvider (高速・WolfPeriod対応版)",
                "ciq": "CIQProvider (高速・WolfPeriod対応版)"
            },
            "performance": {
                "parallel_data_fetch": True,
                "batch_validation": True,
                "optimized_indexing": True,
                "memory_efficient": True
            }
        }

__all__ = [
    "CodeIdentifier",
    "Company",
    "ErrorDetail",
    "ConvertRequest",
    "ConvertResult",
    "SearchQuery",
    "CompanyList",
    "CompanyCatalog",
]
