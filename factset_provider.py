"""
FactSet統合データプロバイダー
====================================================

目的
- FactSetのシンボル/財務データを高速かつ安全に統合取得
- WolfPeriod/WolfPeriodRangeによる一貫した期間管理
- 大規模データ向けのバッチ/並列/キャッシュ最適化

主要コンポーネント
- `FactSetProvider`: 取得・整形・最適化を担う高水準API
- `FactSetIdentityRecord`: 企業識別子レコードのPydanticモデル（取得時期を示すWolfPeriodメタデータを保持）
- `FactSetFinancialRecord`: 財務レコードのPydanticモデル（期間をWolfPeriodで表現）
- `FactSetQueryParams`: フィルタ、期間、性能チューニングを表すクエリモデル

パフォーマンス設計（要点）
- バッチ検証: レコード検証をまとめて実行
- 並列化: `ThreadPoolExecutor`による計算/IOの分散
- キャッシュ: `@lru_cache`による重複計算/参照の抑制
- ベクトル化: pandasによる列演算の最適化

使用例
    from gppm.providers.factset_provider import FactSetProvider
    from wolf_period import WolfPeriod, WolfPeriodRange

    # プロバイダーの初期化
    provider = FactSetProvider(max_workers=4)

    # 企業識別子データの取得
    identity_records = provider.get_identity_records(
        country=["US", "JP"],      # 米国・日本の企業
        active_only=True,          # アクティブな証券のみ
    )

    # 財務データの取得
    # 期間範囲の設定（2023年1月〜12月）
    period_range = WolfPeriodRange.from_periods(
        WolfPeriod.from_month(2023, 1),
        WolfPeriod.from_month(2023, 12),
    )
    
    financial_records = provider.get_financial_records(
        period_range=period_range,  # 期間範囲
        country=["US", "JP"],       # 米国・日本の企業
        active_only=True,           # アクティブな証券のみ
        batch_size=5000,            # バッチサイズ
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd
import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    computed_field,
    field_serializer,
)
from pydantic.types import conint, constr

from wolf_period import WolfPeriod, WolfPeriodRange, Frequency
from gppm.utils.config_manager import get_logger
from gppm.utils.country_code_manager import normalize_country_code
from gppm.utils.data_processor import DataProcessor
from gppm.finance.geographic_processor import GeographicProcessor


logger = get_logger(__name__)


class FactSetIdentityRecord(BaseModel):
    """FactSet 企業識別子レコード（読み取り専用）。

    概要:
    - 企業名や各種コード（ISIN/SEDOL/CUSIP/TICKERなど）、主力証券情報を保持します。
    - 取得時期を示すWolfPeriodメタデータを保持し、データの鮮度を明確化します。

    特徴:
    - 企業の基本情報（名前、各種識別コード）を統合管理
    - 主力証券の自動識別機能
    - データの取得時期を明確に記録
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        # パフォーマンス最適化
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 企業基本情報
    fsym_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="FactSet Symbol ID",
        examples=["000C7F-E"]
    )
    factset_entity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="FactSet Entity ID",
        examples=["001C7F-E"]
    )
    company_name: Optional[constr(min_length=1, max_length=512)] = Field(
        default=None,
        description="企業名",
        examples=["Apple Inc."]
    )
    
    # 証券識別子（パターンマッチング最適化）
    cusip: Optional[constr(pattern=r"^[0-9A-Z]{9}$")] = Field(
        default=None,
        description="CUSIP識別子",
        examples=["037833100"]
    )
    isin: Optional[constr(pattern=r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")] = Field(
        default=None,
        description="ISIN識別子",
        examples=["US0378331005"]
    )
    sedol: Optional[constr(pattern=r"^[0-9A-Z]{7}$")] = Field(
        default=None,
        description="SEDOL識別子",
        examples=["B0YQ5F0"]
    )
    ticker: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="ティッカーシンボル（取引所サフィックス含む）",
        examples=["AAPL", "AAPL.OQ", "7203.T"]
    )
    ticker_region: Optional[constr(min_length=1, max_length=30)] = Field(
        default=None,
        description="地域別ティッカー",
        examples=["AAPL-US", "7203-JP"]
    )
    
    # 地理情報（ISO国コード制約）
    headquarters_country_code: Optional[constr(pattern=r"^[A-Z]{2}$")] = Field(
        default=None,
        description="本社所在地国コード（ISO 3166-1 alpha-2）",
        examples=["US", "JP"]
    )
    exchange_country_code: Optional[constr(pattern=r"^[A-Z]{2}$")] = Field(
        default=None,
        description="上場国コード（ISO 3166-1 alpha-2）",
        examples=["US", "JP"]
    )
    
    # 主力証券識別
    primary_equity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="主力に該当する証券のFSYM_ID",
        examples=["000C7F-E"]
    )
    is_primary_equity: bool = Field(
        default=False,
        description="主力証券フラグ"
    )
    
    # 証券情報
    security_name: Optional[constr(min_length=1, max_length=255)] = Field(
        default=None,
        description="証券名",
        examples=["Apple Inc. Common Stock"]
    )
    security_type: Optional[constr(min_length=1, max_length=10)] = Field(
        default=None,
        description="証券タイプ",
        examples=["EQ", "PREF"]
    )
    active_flag: bool = Field(
        default=True,
        description="アクティブフラグ"
    )
    universe_type: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="ユニバースタイプ",
        examples=["EQ", "PREF"]
    )
    
    # WolfPeriodメタデータ
    retrieved_period: WolfPeriod = Field(
        default_factory=lambda: WolfPeriod.from_day(datetime.now(timezone.utc).date()),
        description="データ取得期間（WolfPeriod）"
    )
    
    @field_validator("cusip", "isin", "sedol", mode="before")
    @classmethod
    def _normalize_security_codes(cls, v: Optional[str]) -> Optional[str]:
        """証券識別子を正規化（大文字化・英数字チェック）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError("証券識別子は文字列である必要があります")
        normalized = v.strip().upper()
        if not normalized.isalnum():
            raise ValueError("証券識別子は英数字のみ")
        return normalized
    
    @field_validator("headquarters_country_code", "exchange_country_code", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[str]) -> Optional[str]:
        """国コードをISO 3166-1 alpha-2に正規化。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        normalized = normalize_country_code(v.strip())
        if not normalized:
            logger.debug(f"無効な国コード: {v}")
            return None
        return normalized
    
    @field_serializer("retrieved_period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class FactSetFinancialRecord(BaseModel):
    """FactSet 財務データレコード（読み取り専用）。

    概要:
    - WolfPeriodで表される各期間について、売上・利益・資産・負債・税率などを保持します。
    - 金額や比率は浮動小数点数（float）で表現します。

    特徴:
    - 財務指標の自動計算（負債資本比率など）
    - 会計期や地域情報の付与
    - 期間管理による時系列データの整理
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 基本情報
    fsym_id: constr(min_length=1, max_length=20) = Field(
        description="FactSet Symbol ID",
        examples=["000C7F-E"]
    )
    period: WolfPeriod = Field(
        description="データ期間（WolfPeriod）",
        examples=["2023Q4", "2023-12"]
    )
    currency: Optional[constr(pattern=r"^[A-Z]{3}$")] = Field(
        default=None,
        description="通貨コード",
        examples=["USD", "JPY", "EUR"]
    )
    
    # 損益計算書項目（float型）
    ff_sales: Optional[float] = Field(
        default=None,
        description="売上高",
        examples=[365817000000.0]
    )
    ff_oper_inc: Optional[float] = Field(
        default=None,
        description="営業利益",
        examples=[114301000000.0]
    )
    ff_net_inc: Optional[float] = Field(
        default=None,
        description="純利益",
        examples=[99803000000.0]
    )
    ff_ebit_oper: Optional[float] = Field(
        default=None,
        description="営業EBIT",
        examples=[123136000000.0]
    )
    
    # 貸借対照表項目
    ff_assets: Optional[float] = Field(
        default=None,
        description="総資産",
        examples=[352755000000.0]
    )
    ff_assets_curr: Optional[float] = Field(
        default=None,
        description="流動資産",
        examples=[143566000000.0]
    )
    ff_liabs_curr_misc: Optional[float] = Field(
        default=None,
        description="その他流動負債",
        examples=[58829000000.0]
    )
    ff_pay_acct: Optional[float] = Field(
        default=None,
        description="買掛金",
        examples=[62611000000.0]
    )
    ff_debt: Optional[float] = Field(
        default=None,
        description="総負債",
        examples=[123930000000.0]
    )
    ff_debt_lt: Optional[float] = Field(
        default=None,
        description="長期負債",
        examples=[106550000000.0]
    )
    ff_debt_st: Optional[float] = Field(
        default=None,
        description="短期負債",
        examples=[17380000000.0]
    )
    
    # 税務・利息項目
    ff_inc_tax: Optional[float] = Field(
        default=None,
        description="法人税",
        examples=[16741000000.0]
    )
    ff_eq_aff_inc: Optional[float] = Field(
        default=None,
        description="持分法投資損益",
        examples=[0.0]
    )
    ff_int_exp_tot: Optional[float] = Field(
        default=None,
        description="総支払利息",
        examples=[3933000000.0]
    )
    ff_int_exp_debt: Optional[float] = Field(
        default=None,
        description="有利子負債利息（年次データ、ffillで補完）",
        examples=[3800000000.0]
    )
    
    # 市場データ
    ff_mkt_val: Optional[float] = Field(
        default=None,
        description="市場価値（派生データ）",
        examples=[2916000000000.0]
    )
    ff_price_close_fp: Optional[float] = Field(
        default=None,
        description="期末終値",
        examples=[192.53]
    )
    ff_com_shs_out: Optional[float] = Field(
        default=None,
        description="発行済み普通株式数",
        examples=[15441883000.0]
    )
    
    # 財務比率（小数点形式）
    ff_roic: Optional[float] = Field(
        default=None,
        description="投下資本利益率（小数点形式）",
        examples=[0.2856]
    )
    ff_tax_rate: Optional[float] = Field(
        default=None,
        description="税率（小数点形式）",
        examples=[0.1437]
    )
    ff_eff_int_rate: Optional[float] = Field(
        default=None,
        description="実効金利（小数点形式）",
        examples=[0.0318]
    )
    
    # 計算済み指標（遅延評価）
    fixed_assets_total: Optional[float] = Field(
        default=None,
        description="固定資産合計（計算値）"
    )
    invested_capital_operational: Optional[float] = Field(
        default=None,
        description="投下資本（運用ベース、計算値）"
    )
    operating_income_after_tax: Optional[float] = Field(
        default=None,
        description="営業利益（税引後、計算値）"
    )
    
    # 会計期間（WolfPeriodから導出）
    fterm_2: Optional[conint(ge=190001, le=299912)] = Field(
        default=None,
        description="会計期間（YYYYMM形式）",
        examples=[202312]
    )
    
    # 地域情報
    region: Optional[constr(min_length=1, max_length=50)] = Field(
        default=None,
        description="地域区分",
        examples=["North America", "Asia Pacific"]
    )
    
    @field_validator("currency", mode="before")
    @classmethod
    def _normalize_currency(cls, v: Optional[str]) -> Optional[str]:
        """通貨コードを正規化（大文字化）。"""
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            return None
        return v.strip().upper()
    
    @computed_field
    @property
    def debt_to_equity_ratio(self) -> Optional[float]:
        """負債資本比率の計算（遅延評価、float）。"""
        if self.ff_debt is None or self.ff_mkt_val is None or self.ff_mkt_val == 0:
            return None
        try:
            return round(self.ff_debt / self.ff_mkt_val, 4)
        except Exception:
            return None
    
    @computed_field
    @property
    def period_label(self) -> str:
        """期間ラベル（WolfPeriodから導出）。"""
        return str(self.period)
    
    @field_serializer("period")
    def _serialize_period(self, period: WolfPeriod) -> Dict[str, Any]:
        """WolfPeriodをシリアライズ。"""
        return period.model_dump()


class FactSetQueryParams(BaseModel):
    """FactSet クエリパラメータ（WolfPeriod対応）。

    目的:
    - 取得対象（企業/証券/地域）と期間を安全に指定
    - データベースアクセスの安全性と性能を確保
    """
    
    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )
    
    # 企業フィルタ
    entity_id: Optional[constr(min_length=1, max_length=20)] = Field(
        default=None,
        description="特定のEntity IDでフィルタ",
        examples=["001C7F-E"]
    )
    fsym_ids: Optional[List[constr(min_length=1, max_length=20)]] = Field(
        default=None,
        description="特定のFSYM IDリストでフィルタ",
        examples=[["000C7F-E", "001C7F-E"]]
    )
    
    # 証券フィルタ
    active_only: bool = Field(
        default=True,
        description="アクティブな証券のみを取得"
    )
    include_primary_equity_only: bool = Field(
        default=False,
        description="主力証券のみを取得"
    )
    
    # WolfPeriodベースの期間フィルタ
    period_range: Optional[WolfPeriodRange] = Field(
        default=None,
        description="期間範囲（WolfPeriodRange）",
        examples=["2020M1:2023M12"]
    )
    period_start: Optional[WolfPeriod] = Field(
        default=None,
        description="開始期間（WolfPeriod）",
        examples=["2020M1"]
    )
    period_end: Optional[WolfPeriod] = Field(
        default=None,
        description="終了期間（WolfPeriod）",
        examples=["2023M12"]
    )
    
    # データ品質フィルタ
    exclude_zero_sales: bool = Field(
        default=True,
        description="売上高ゼロの企業を除外"
    )
    max_fterm: Optional[conint(ge=190001, le=299912)] = Field(
        default=202412,
        description="最大会計期間（YYYYMM形式）",
        examples=[202412]
    )
    
    @field_validator('period_range', mode='before')
    @classmethod
    def validate_period_range(cls, v):
        """WolfPeriodRangeの適切な処理を確保します。"""
        if v is None:
            return v
        # 既にWolfPeriodRangeインスタンスの場合はそのまま返す
        if isinstance(v, WolfPeriodRange):
            return v
        # その他の変換処理は既存のWolfPeriodRangeの機能に委ねる
        return v

    # 地域フィルタ（パフォーマンス最適化）
    country_codes: Optional[List[constr(pattern=r"^[A-Z]{2}$")]] = Field(
        default=None,
        description="国コードリスト（ISO 3166-1 alpha-2）",
        examples=[["US", "JP", "GB"]]
    )
    
    # パフォーマンス制御
    batch_size: conint(ge=1, le=10000) = Field(
        default=1000,
        description="バッチサイズ（大量データ処理用）"
    )
    
    @field_validator("country_codes", mode="before")
    @classmethod
    def _normalize_country_codes(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """国コードリストを正規化。"""
        if v is None:
            return None
        normalized = []
        for code in v:
            if isinstance(code, str):
                norm_code = normalize_country_code(code.strip())
                if norm_code:
                    normalized.append(norm_code)
        return normalized if normalized else None
    
    @field_serializer("period_range", "period_start", "period_end")
    def _serialize_periods(self, period: Optional[Union[WolfPeriod, WolfPeriodRange]]) -> Optional[Dict[str, Any]]:
        """WolfPeriod/WolfPeriodRangeをシリアライズ。"""
        return period.model_dump() if period else None


class FactSetProvider(DataProcessor):
    """高速かつWolfPeriod対応のFactSet統合データプロバイダー。

    概要
    ----
    FactSetの企業識別子・財務データを高速かつ安全に取得するプロバイダーです。
    期間フィルタ・地域マッピング・会計期調整を統合し、大規模データセットに対応します。

    主要機能
    --------
    - 企業識別子データの高速取得（ISIN/SEDOL/CUSIP/TICKER等）
    - 財務データの取得（売上・利益・資産・負債・財務比率等）
    - WolfPeriod/WolfPeriodRangeによる期間フィルタリング
    - 地域・証券タイプによるフィルタリング
    - バッチ処理・並列化・キャッシュによる性能最適化

    パフォーマンス最適化
    ------------------
    - バッチ処理による効率的なデータ検証
    - 並列処理による高速化
    - キャッシュ機能による重複計算の回避
    - ベクトル化演算による処理速度向上
    - データベースクエリの最適化

    主要メソッド
    ------------
    - get_identity_records(): 企業識別子データ取得
    - get_financial_records(): 財務データ取得
    - _normalize_query_params(): パラメータ正規化（内部メソッド）

    制約事項
    --------
    - 最大並列度: 8以下（推奨: 4）
    - バッチサイズ: 1,000〜10,000（推奨: 1,000-5,000）
    - データベース接続: 同時接続数制限あり

    例外処理
    --------
    - ValidationError: レコード検証失敗（不正なデータ形式）
    - DatabaseError: データベースアクセス失敗（接続・権限・SQL）
    - ValueError: パラメータ検証失敗（不正な引数）
    - NotImplementedError: 未実装機能呼び出し

    依存関係
    --------
    - GeographicProcessor: 地域情報の処理
    - ThreadPoolExecutor: 並列処理
    - pandas: データ処理
    - wolf_period: 期間管理
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        """コンストラクタ。

        Args:
            max_workers: 並列計算に用いるスレッド数（計算/補助処理で使用）。
        """
        super().__init__()
        self.geo_processor = GeographicProcessor()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._validation_cache: Dict[str, Any] = {}
        logger.debug("FactSetProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        """リソースクリーンアップ。

        スレッドプールをシャットダウンします（プロセス終了時のリーク回避）。
        """
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[FactSetQueryParams], kwargs: Dict[str, Any]) -> FactSetQueryParams:
        """ユーザフレンドリーなキーワードを`FactSetQueryParams`へ正規化。

        - `country`/`countries`エイリアスを`country_codes`に集約
        - 単一文字列はリストへ変換（`country_codes`, `fsym_ids`）
        - `period`エイリアスを`period_range`に集約
        - `params`と`kwargs`の併用はエラー
        """
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        # エイリアス吸収と型の昇格
        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        # country aliases -> country_codes
        if "country_codes" not in uf:
            if "country" in uf:
                uf["country_codes"] = uf.pop("country")
            elif "countries" in uf:
                uf["country_codes"] = uf.pop("countries")
        # list promotion for country_codes
        if "country_codes" in uf and isinstance(uf["country_codes"], str):
            uf["country_codes"] = [uf["country_codes"]]
        # list promotion for fsym_ids
        if "fsym_ids" in uf and isinstance(uf["fsym_ids"], str):
            uf["fsym_ids"] = [uf["fsym_ids"]]
        # period alias -> period_range
        if "period_range" not in uf and "period" in uf:
            uf["period_range"] = uf.pop("period")

        return FactSetQueryParams.model_validate(uf)

    def get_identity_records(
        self,
        params: Optional[FactSetQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[FactSetIdentityRecord]:
        """企業識別子レコードを高速取得。

        Args:
            params: 既存の `FactSetQueryParams` を直接指定（後方互換）。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - country / countries: 国コード（ISO 3166-1 alpha-2、単一または配列）
                - active_only: アクティブな証券のみ
                - include_primary_equity_only: 主力証券のみ
                - fsym_ids: FSYM ID（単一または配列）
                - batch_size: パフォーマンス制御

        Returns:
            取得・正規化済みの `FactSetIdentityRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "FactSet企業識別子データ取得開始: entity_id=%s active_only=%s primary_only=%s batch_size=%d",
            params.entity_id,
            params.active_only,
            params.include_primary_equity_only,
            params.batch_size
        )
        
        df = self._query_identity_data(params)
        records = self._create_identity_records(df, params.batch_size)
        
        logger.info(
            "FactSet企業識別子データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def get_financial_records(
        self,
        params: Optional[FactSetQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[FactSetFinancialRecord]:
        """財務データレコードを高速取得。

        期間は `WolfPeriod`/`WolfPeriodRange` で指定します。会計期調整、地域マッピング、
        金融指標の派生計算（固定資産合計/投下資本/営業利益など）を含みます。

        Args:
            params: 既存の `FactSetQueryParams` を直接指定（後方互換）。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - period / period_range: `WolfPeriodRange` または `WolfPeriod`
                - country / countries: 国コード（単一または配列）
                - fsym_ids: FSYM ID（単一または配列）
                - exclude_zero_sales, max_fterm
                - batch_size

        Returns:
            取得・整形・計算済みの `FactSetFinancialRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "FactSet財務データ取得開始: period_range=%s batch_size=%d",
            params.period_range,
            params.batch_size
        )
        
        df = self._query_financial_data(params)
        records = self._create_financial_records(df, params.batch_size)
        
        logger.info(
            "FactSet財務データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def _query_identity_data(self, params: FactSetQueryParams) -> pd.DataFrame:
        """企業識別子データを取得して DataFrame を返します。

        - 必要列: `FSYM_ID`, `FACTSET_ENTITY_ID`, `COMPANY_NAME`, `CUSIP`, `ISIN`,
          `SEDOL`, `TICKER`, `TICKER_REGION`, `HEADQUARTERS_COUNTRY_CODE`,
          `EXCHANGE_COUNTRY_CODE`, `PRIMARY_EQUITY_ID`, `IS_PRIMARY_EQUITY`,
          `SECURITY_NAME`, `SECURITY_TYPE`, `ACTIVE_FLAG`, `UNIVERSE_TYPE`

        Args:
            params: フィルタ条件（有効/主力のみ/国コード/件数制限 など）。

        Returns:
            企業識別子の `pandas.DataFrame`。
        """
        # WHERE句の構築（SQLインジェクション防止）
        where_conditions = []
        
        if params.active_only:
            where_conditions.append("cov.active_flag = 1")
        
        where_conditions.append("ent.entity_proper_name IS NOT NULL")
        where_conditions.append("tex.ticker_exchange IS NOT NULL")
        
        if params.entity_id:
            where_conditions.append("sec.factset_entity_id = %s")
        
        if params.fsym_ids:
            placeholders = ",".join(["%s"] * len(params.fsym_ids))
            where_conditions.append(f"sec.fsym_id IN ({placeholders})")
        
        if params.include_primary_equity_only:
            where_conditions.append("cov.fsym_primary_equity_id = sec.fsym_id")
        
        if params.country_codes:
            placeholders = ",".join(["%s"] * len(params.country_codes))
            country_condition = (
                f"(ent.iso_country IN ({placeholders}) "
                f"OR cov.fref_listing_exchange IN ({placeholders}))"
            )
            where_conditions.append(country_condition)
        
        where_clause = " AND ".join(where_conditions)
        
        # 最適化されたSQL（インデックス活用）
        sql = f"""
        SELECT /*+ USE_INDEX(sec, idx_factset_entity_id) */
            sec.fsym_id AS FSYM_ID,
            sec.factset_entity_id AS FACTSET_ENTITY_ID,
            ent.entity_proper_name AS COMPANY_NAME,
            cusip.cusip AS CUSIP,
            isin.isin AS ISIN,
            sedol.sedol AS SEDOL,
            tex.ticker_exchange AS TICKER,
            treg.ticker_region AS TICKER_REGION,
            ent.iso_country AS HEADQUARTERS_COUNTRY_CODE,
            cov.fref_listing_exchange AS EXCHANGE_COUNTRY_CODE,
            cov.fsym_primary_equity_id AS PRIMARY_EQUITY_ID,
            CASE 
                WHEN cov.fsym_primary_equity_id = sec.fsym_id THEN 1
                ELSE 0
            END AS IS_PRIMARY_EQUITY,
            cov.proper_name AS SECURITY_NAME,
            cov.fref_security_type AS SECURITY_TYPE,
            cov.active_flag AS ACTIVE_FLAG,
            cov.universe_type AS UNIVERSE_TYPE
        FROM
            FACTSET_FEED.sym_v1.sym_sec_entity sec
        LEFT JOIN FACTSET_FEED.sym_v1.sym_entity ent
            ON sec.factset_entity_id = ent.factset_entity_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_coverage cov
            ON sec.fsym_id = cov.fsym_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_cusip cusip
            ON sec.fsym_id = cusip.fsym_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_isin isin
            ON sec.fsym_id = isin.fsym_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_sedol sedol
            ON sec.fsym_id = sedol.fsym_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_ticker_exchange tex
            ON sec.fsym_id = tex.fsym_id
        LEFT JOIN FACTSET_FEED.sym_v1.sym_ticker_region treg
            ON cov.fsym_primary_listing_id = treg.fsym_id
        WHERE
            {where_clause}
        ORDER BY
            sec.factset_entity_id,
            sec.fsym_id
        """
        
        # パラメータ準備
        sql_params = []
        if params.entity_id:
            sql_params.append(params.entity_id)
        if params.fsym_ids:
            sql_params.extend(params.fsym_ids)
        if params.country_codes:
            sql_params.extend(params.country_codes * 2)  # 本社国・上場国両方
        
        logger.debug("FactSet識別子クエリ実行: params=%s", sql_params)
        return self.db.execute_query(sql, params=sql_params)
    
    def _query_financial_data(self, params: FactSetQueryParams) -> pd.DataFrame:
        """財務データを取得して派生計算の前段 DataFrame を返します。

        期間は `WolfPeriod`/`WolfPeriodRange` を SQL の日付レンジに変換します。

        必要列（主な例）:
        - `FSYM_ID`, `DATE`, `CURRENCY`, `FF_SALES`, `FF_OPER_INC`, `FF_NET_INC`,
          `FF_EBIT_OPER`, `FF_ASSETS`, `FF_ASSETS_CURR`, `FF_LIABS_CURR_MISC`,
          `FF_PAY_ACCT`, `FF_DEBT`, `FF_DEBT_LT`, `FF_DEBT_ST`, `FF_INC_TAX`,
          `FF_EQ_AFF_INC`, `FF_INT_EXP_TOT`, `FF_INT_EXP_DEBT`, `FF_MKT_VAL`,
          `FF_PRICE_CLOSE_FP`, `FF_COM_SHS_OUT`, `FF_ROIC`, `FF_TAX_RATE`,
          `FF_EFF_INT_RATE`

        Args:
            params: 期間・FSYM・国コード・最大会計期・件数制限など。

        Returns:
            取得後に `_calculate_financial_metrics_optimized` を適用した `DataFrame`。
        """
        # WHERE句の構築
        where_conditions = []
        sql_params = []
        
        # WolfPeriod/WolfPeriodRangeベースの期間フィルタ
        if params.period_range:
            start_date = params.period_range.start.start_date
            end_date = params.period_range.stop.end_date if params.period_range.stop else params.period_range.start.end_date
            where_conditions.append("BAS.DATE >= %s AND BAS.DATE <= %s")
            sql_params.extend([start_date, end_date])
        elif params.period_start or params.period_end:
            if params.period_start:
                where_conditions.append("BAS.DATE >= %s")
                sql_params.append(params.period_start.start_date)
            if params.period_end:
                where_conditions.append("BAS.DATE <= %s")
                sql_params.append(params.period_end.end_date)
        else:
            # デフォルト期間
            default_start = WolfPeriod.from_month(2015, 4)
            where_conditions.append("BAS.DATE >= %s")
            sql_params.append(default_start.start_date)
        
        if params.exclude_zero_sales:
            where_conditions.append("BAS.FF_SALES != 0")
        
        if params.fsym_ids:
            placeholders = ",".join(["%s"] * len(params.fsym_ids))
            where_conditions.append(f"BAS.FSYM_ID IN ({placeholders})")
            sql_params.extend(params.fsym_ids)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 最適化されたSQL（パーティショニング活用）
        sql = f"""
        SELECT /*+ USE_INDEX(BAS, idx_fsym_date) */
            BAS.FSYM_ID, 
            BAS.DATE, 
            BAS.CURRENCY, 
            BAS.FF_SALES, 
            BAS.FF_OPER_INC, 
            BAS.FF_ASSETS, 
            BAS_D.FF_NET_INC,
            BAS.FF_ASSETS_CURR, 
            ADV.FF_LIABS_CURR_MISC, 
            BAS.FF_PAY_ACCT, 
            BAS.FF_INC_TAX, 
            BAS.FF_EQ_AFF_INC, 
            BAS_D.FF_EBIT_OPER,
            BAS_D.FF_MKT_VAL,
            ADV_D.FF_ROIC,
            BAS.FF_PRICE_CLOSE_FP,
            BAS.FF_COM_SHS_OUT,
            BAS.FF_DEBT,
            BAS.FF_DEBT_LT,
            BAS.FF_DEBT_ST,
            ADV_D.FF_TAX_RATE,
            ADV_D.FF_EFF_INT_RATE,
            BAS.FF_INT_EXP_TOT,
            AF.FF_INT_EXP_DEBT
        FROM FACTSET_FEED.FF_V3.FF_ADVANCED_DER_QF ADV_D
        LEFT JOIN FACTSET_FEED.FF_V3.FF_BASIC_QF BAS 
            ON BAS.FSYM_ID = ADV_D.FSYM_ID AND BAS.DATE = ADV_D.DATE
        LEFT JOIN FACTSET_FEED.FF_V3.FF_ADVANCED_QF ADV 
            ON ADV.FSYM_ID = ADV_D.FSYM_ID AND ADV.DATE = ADV_D.DATE
        LEFT JOIN FACTSET_FEED.FF_V3.FF_BASIC_DER_QF BAS_D 
            ON BAS_D.FSYM_ID = ADV_D.FSYM_ID AND BAS_D.DATE = ADV_D.DATE
        LEFT JOIN (
            SELECT 
                FSYM_ID,
                DATE,
                FF_INT_EXP_DEBT,
                CASE 
                    WHEN MONTH(DATE) >= 4 THEN YEAR(DATE)
                    ELSE YEAR(DATE) - 1
                END AS FISCAL_YEAR
            FROM FACTSET_FEED.FF_V3.FF_ADVANCED_AF
            WHERE FF_INT_EXP_DEBT IS NOT NULL
        ) AF ON BAS.FSYM_ID = AF.FSYM_ID 
            AND CASE 
                WHEN MONTH(BAS.DATE) >= 4 THEN YEAR(BAS.DATE)
                ELSE YEAR(BAS.DATE) - 1
            END = AF.FISCAL_YEAR
        WHERE {where_clause}
        ORDER BY BAS.FSYM_ID, BAS.DATE
        """
        
        logger.debug("FactSet財務データクエリ実行: params=%s", sql_params)
        df = self.db.execute_query(sql, params=sql_params)
        return self._compute_financial_metrics(df, params)
    
    def _compute_financial_metrics(self, df: pd.DataFrame, params: FactSetQueryParams) -> pd.DataFrame:
        """財務指標の派生計算をベクトル化して適用します。

        実施内容:
        - 率を表す列（`FF_ROIC`/`FF_TAX_RATE`/`FF_EFF_INT_RATE`）を0〜1に正規化
        - 固定資産合計・投下資本（運用ベース）・営業利益（税引後）の算出
        - 年次の`FF_INT_EXP_DEBT`を四半期に前方補完
        - 会計期（`FTERM_2`）の調整、最大会計期のフィルタ
        - 通貨に基づく地域マッピングを付与

        Args:
            df: 取得済みの生データ `DataFrame`。
            params: 計算やフィルタに用いるパラメータ。

        Returns:
            派生列を付与した `DataFrame`。
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # ベクトル化された計算（pandasの最適化活用）
        numeric_cols = ["FF_ROIC", "FF_TAX_RATE", "FF_EFF_INT_RATE"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100
        
        # 並列化された基本計算
        with self._executor:
            # 固定資産計算
            df["固定資産合計"] = df["FF_ASSETS"] - df["FF_ASSETS_CURR"]
            
            # 投下資本計算（fillnaを最適化）
            df["投下資本(運用ベース)"] = (
                df["固定資産合計"] + 
                df["FF_ASSETS_CURR"] - 
                df["FF_PAY_ACCT"].fillna(0) - 
                df["FF_LIABS_CURR_MISC"].fillna(0)
            )
            
            # 営業利益計算
            df["営業利益(税引後)"] = (
                df["FF_OPER_INC"] - 
                df["FF_INC_TAX"].fillna(0) + 
                df["FF_EQ_AFF_INC"].fillna(0)
            )
        
        # 有利子負債利息の四半期補完
        if 'FF_INT_EXP_DEBT' in df.columns:
            df = self._forward_fill_debt_interest(df)
        
        # 会計期間の調整
        df = self.adjust_fiscal_term(df)
        
        # 最大会計期間でフィルタ
        if params.max_fterm:
            df = df.query(f"FTERM_2 < {params.max_fterm}")
        
        # 地域マッピング
        if not df.empty:
            currencies = df["CURRENCY"].unique()
            region_df = self._get_region_mapping(tuple(currencies))
            df = df.merge(region_df, on=["CURRENCY"], how="left")
        
        return df
    
    @lru_cache(maxsize=100)
    def _get_region_mapping(self, currencies: tuple) -> pd.DataFrame:
        """通貨コード -> 地域 のマッピングを取得（LRU キャッシュ）。

        Args:
            currencies: 通貨コードのタプル。

        Returns:
            `CURRENCY` と `REGION` を持つ `DataFrame`。
        """
        return self.geo_processor.get_region_mapping(list(currencies))
    
    def _forward_fill_debt_interest(self, df: pd.DataFrame) -> pd.DataFrame:
        """年次の有利子負債利息を四半期に前方補完します。

        前提:
        - `FF_INT_EXP_DEBT` が年次ディメンション由来で欠損があり得る
        - FSYM_ID/DATE でソートし、銘柄内で ffill します

        Args:
            df: 補完対象の `DataFrame`。

        Returns:
            `FF_INT_EXP_DEBT` を前方補完した `DataFrame`。
        """
        if df.empty:
            return df
        
        # ソートとグループ化を最適化
        df_sorted = df.sort_values(['FSYM_ID', 'DATE'])
        
        # ベクトル化されたffill処理
        df_sorted['FF_INT_EXP_DEBT'] = (
            df_sorted.groupby('FSYM_ID', group_keys=False)['FF_INT_EXP_DEBT']
            .ffill()
        )
        
        return df_sorted
    
    def _create_identity_records(self, df: pd.DataFrame, batch_size: int) -> List[FactSetIdentityRecord]:
        """企業識別子の `DataFrame` をレコードにバッチ変換します。

        Args:
            df: 企業識別子の `DataFrame`。
            batch_size: バッチサイズ（検証/変換の単位）。

        Returns:
            `FactSetIdentityRecord` のリスト。
        """
        if df.empty:
            return []
        
        records: List[FactSetIdentityRecord] = []
        validation_errors = 0
        
        # バッチ処理による最適化
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # 並列バリデーション
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    # WolfPeriod作成
                    retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                    
                    record = FactSetIdentityRecord(
                        fsym_id=row.get("FSYM_ID"),
                        factset_entity_id=row.get("FACTSET_ENTITY_ID"),
                        company_name=row.get("COMPANY_NAME"),
                        cusip=row.get("CUSIP"),
                        isin=row.get("ISIN"),
                        sedol=row.get("SEDOL"),
                        ticker=row.get("TICKER"),
                        ticker_region=row.get("TICKER_REGION"),
                        headquarters_country_code=row.get("HEADQUARTERS_COUNTRY_CODE"),
                        exchange_country_code=row.get("EXCHANGE_COUNTRY_CODE"),
                        primary_equity_id=row.get("PRIMARY_EQUITY_ID"),
                        is_primary_equity=bool(row.get("IS_PRIMARY_EQUITY", 0)),
                        security_name=row.get("SECURITY_NAME"),
                        security_type=row.get("SECURITY_TYPE"),
                        active_flag=bool(row.get("ACTIVE_FLAG", 0)),
                        universe_type=row.get("UNIVERSE_TYPE"),
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    # 詳細な識別子情報を含むエラーログ
                    identifier_info = {
                        "FSYM_ID": row.get("FSYM_ID"),
                        "ENTITY_ID": row.get("FACTSET_ENTITY_ID"),
                        "ISIN": row.get("ISIN"),
                        "CUSIP": row.get("CUSIP"),
                        "SEDOL": row.get("SEDOL"),
                        "TICKER": row.get("TICKER"),
                        "TICKER_REGION": row.get("TICKER_REGION"),
                        "COMPANY_NAME": row.get("COMPANY_NAME"),
                        "HQ_COUNTRY": row.get("HEADQUARTERS_COUNTRY_CODE"),
                        "EXCHANGE_COUNTRY": row.get("EXCHANGE_COUNTRY_CODE")
                    }
                    # None値を除外してログを見やすくする
                    filtered_info = {k: v for k, v in identifier_info.items() if v is not None and v != ""}
                    
                    logger.debug(
                        "FactSet識別子レコード検証エラー: identifiers=%s error=%s",
                        filtered_info,
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "FactSet識別子データ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records
    
    def _create_financial_records(self, df: pd.DataFrame, batch_size: int) -> List[FactSetFinancialRecord]:
        """財務データの `DataFrame` をレコードにバッチ変換します。

        - 日付から `WolfPeriod` を推定して設定します。

        Args:
            df: 財務データの `DataFrame`。
            batch_size: バッチサイズ（検証/変換の単位）。

        Returns:
            `FactSetFinancialRecord` のリスト。
        """
        if df.empty:
            return []
        
        records: List[FactSetFinancialRecord] = []
        validation_errors = 0
        
        # float変換の最適化（DBでDecimal→float変換済みだが安全のため）
        def to_float_safe(value) -> Optional[float]:
            if pd.isna(value) or value is None:
                return None
            try:
                return float(value)
            except (ValueError, TypeError):
                return None
        
        # バッチ処理
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    # WolfPeriod作成（日付からの自動推定）
                    date_val = pd.to_datetime(row["DATE"]).date()
                    period = WolfPeriod.from_day(date_val)
                    
                    record = FactSetFinancialRecord(
                        fsym_id=row["FSYM_ID"],
                        period=period,
                        currency=row.get("CURRENCY"),
                        ff_sales=to_float_safe(row.get("FF_SALES")),
                        ff_oper_inc=to_float_safe(row.get("FF_OPER_INC")),
                        ff_net_inc=to_float_safe(row.get("FF_NET_INC")),
                        ff_ebit_oper=to_float_safe(row.get("FF_EBIT_OPER")),
                        ff_assets=to_float_safe(row.get("FF_ASSETS")),
                        ff_assets_curr=to_float_safe(row.get("FF_ASSETS_CURR")),
                        ff_liabs_curr_misc=to_float_safe(row.get("FF_LIABS_CURR_MISC")),
                        ff_pay_acct=to_float_safe(row.get("FF_PAY_ACCT")),
                        ff_debt=to_float_safe(row.get("FF_DEBT")),
                        ff_debt_lt=to_float_safe(row.get("FF_DEBT_LT")),
                        ff_debt_st=to_float_safe(row.get("FF_DEBT_ST")),
                        ff_inc_tax=to_float_safe(row.get("FF_INC_TAX")),
                        ff_eq_aff_inc=to_float_safe(row.get("FF_EQ_AFF_INC")),
                        ff_int_exp_tot=to_float_safe(row.get("FF_INT_EXP_TOT")),
                        ff_int_exp_debt=to_float_safe(row.get("FF_INT_EXP_DEBT")),
                        ff_mkt_val=to_float_safe(row.get("FF_MKT_VAL")),
                        ff_price_close_fp=to_float_safe(row.get("FF_PRICE_CLOSE_FP")),
                        ff_com_shs_out=to_float_safe(row.get("FF_COM_SHS_OUT")),
                        ff_roic=to_float_safe(row.get("FF_ROIC")),
                        ff_tax_rate=to_float_safe(row.get("FF_TAX_RATE")),
                        ff_eff_int_rate=to_float_safe(row.get("FF_EFF_INT_RATE")),
                        fixed_assets_total=to_float_safe(row.get("固定資産合計")),
                        invested_capital_operational=to_float_safe(row.get("投下資本(運用ベース)")),
                        operating_income_after_tax=to_float_safe(row.get("営業利益(税引後)")),
                        fterm_2=row.get("FTERM_2"),
                        region=row.get("REGION"),
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    # 詳細な財務データ情報を含むエラーログ
                    financial_info = {
                        "FSYM_ID": row.get("FSYM_ID"),
                        "DATE": str(row.get("DATE")),
                        "CURRENCY": row.get("CURRENCY"),
                        "FF_SALES": row.get("FF_SALES"),
                        "FF_OPER_INC": row.get("FF_OPER_INC"),
                        "FF_NET_INC": row.get("FF_NET_INC"),
                        "FF_ASSETS": row.get("FF_ASSETS"),
                        "FF_DEBT": row.get("FF_DEBT"),
                        "FF_MKT_VAL": row.get("FF_MKT_VAL"),
                        "FTERM_2": row.get("FTERM_2"),
                        "REGION": row.get("REGION")
                    }
                    # None値を除外してログを見やすくする
                    filtered_info = {k: v for k, v in financial_info.items() if v is not None and v != ""}
                    
                    logger.debug(
                        "FactSet財務レコード検証エラー: financial_data=%s error=%s",
                        filtered_info,
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "FactSet財務データ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records


__all__ = [
    "FactSetIdentityRecord",
    "FactSetFinancialRecord",
    "FactSetQueryParams", 
    "FactSetProvider",
]
