"""
MSCIデータプロバイダー
=====================

目的
- MSCI構成銘柄データとインデックスデータの取得・管理
- グローバルベータ計算用のデータ提供を主目的とする
- WolfPeriod/WolfPeriodRangeによる一貫した期間管理
- 大規模データ向けのバッチ/並列/キャッシュ最適化

主要コンポーネント
- `MSCIProvider`: 取得・整形・最適化を担う高水準API
- `MSCIIndexRecord`: インデックス情報のPydanticモデル
- `MSCISecurityRecord`: 証券情報のPydanticモデル
- ベータ計算結果は `BetaRecord` として提供
- `MSCIQueryParams`: フィルタ、期間、性能チューニングを表すクエリモデル

使用例
    from data_providers.data_sources.msci.provider import MSCIProvider
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

"""

from __future__ import annotations

from datetime import datetime, timezone, date
from typing import List, Optional, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from wolf_period import WolfPeriod, WolfPeriodRange, Frequency
from gppm.core.config_manager import get_logger
from gppm.utils.country_code_manager import convert_to_alpha2, convert_to_alpha3, Alpha2Code, Alpha3Code
from data_providers.core.base_provider import BaseProvider
from gppm.utils.company_codes import CompanyCatalog
from gppm.analysis.risk_capital.beta_calculator import (
    BetaCalculator,
    ReturnFrequency,
    BetaType,
    BetaResult,
)

logger = get_logger(__name__)

from .types import (
    IndexType,
    DividendType,
    MSCIIndexRecord,
    MSCISecurityRecord,
)
from .query_params import MSCIQueryParams

# ベータ計算結果型定義
from gppm.analysis.risk_capital.beta_types import BetaRecord

class MSCIProvider(BaseProvider):
    """高速かつWolfPeriod対応のMSCI統合データプロバイダー。

    概要
    ----
    MSCIの構成銘柄データとインデックスデータを高速かつ安全に取得するプロバイダーです。
    期間フィルタ・地域マッピング・ベータ計算を統合し、大規模データセットに対応します。

    主要機能
    --------
    - インデックス構成銘柄データの高速取得
    - インデックス値データの取得（月次）
    - 証券価格データの取得（日次）
    - ベータ計算（グローバル・ローカル）
    - WolfPeriod/WolfPeriodRangeによる期間フィルタリング
    - バッチ処理・並列化・キャッシュによる性能最適化

    主要メソッド
    ------------
    - get_index_constituents(): インデックス構成銘柄取得
    - get_index_values(): インデックス値取得
    - get_security_prices(): 証券価格取得
    - get_beta(): ベータ計算
    """
    
    # 主要なMSCIインデックスの特殊ISO_COUNTRY_CODEマッピング
    MAJOR_INDICES = {
        'WORLD': {
            'country_codes': {'@DP': '配当なし', '@DT': '配当込み'},
            'default_code': '@DP',
            'name': 'MSCI World Index', 
            'description': '先進国株式インデックス'
        },
        'ACWI': {
            'country_codes': {'@AP': '配当なし', '@AT': '配当込み'},
            'default_code': '@AP',
            'name': 'MSCI ACWI',
            'description': '全世界株式インデックス（先進国＋新興国）'
        },
    }
    
    def __init__(self, max_workers: int = 4) -> None:
        """コンストラクタ。

        Args:
            max_workers: 並列計算に用いるスレッド数（計算/補助処理で使用）。
        """
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._validation_cache: Dict[str, Any] = {}
        
        # 企業コード辞書を初期ロード（失敗しても後続で遅延ロード）
        self.company_catalog = CompanyCatalog()
        try:
            self.company_catalog.load(primary_equity_only=True)
        except Exception as e:
            logger.warning(f"企業コードの初期ロードに失敗: {e}")
        
        # MAJOR_INDICESに含まれる全ての国コードを事前にセットとして抽出
        self._major_index_codes = {
            code
            for index_info in self.MAJOR_INDICES.values()
            for code in index_info['country_codes'].keys()
        }
        
        logger.debug("MSCIProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        """リソースクリーンアップ。

        スレッドプールをシャットダウンします（プロセス終了時のリーク回避）。
        """
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[MSCIQueryParams], kwargs: Dict[str, Any]) -> MSCIQueryParams:
        """ユーザフレンドリーなキーワードを`MSCIQueryParams`へ正規化。

        - `params`と`kwargs`の併用はエラー
        - 単一文字列はリストへ変換（`security_ids`）
        - 旧形式のパラメータを統一的な`period`パラメータに変換
        - `date`オブジェクトは`WolfPeriod`に変換
        """
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        # エイリアス吸収と型の昇格
        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        
        # list promotion for security_ids
        if "security_ids" in uf and isinstance(uf["security_ids"], str):
            uf["security_ids"] = [uf["security_ids"]]

        # 旧形式パラメータを統一的なperiodパラメータに変換
        # 優先順位: period > period_range > as_of_periods
        
        if "period" not in uf:
            # period_range を period に変換
            if "period_range" in uf:
                uf["period"] = uf.pop("period_range")
            
            # as_of_periods を period に変換
            elif "as_of_periods" in uf:
                uf["period"] = uf.pop("as_of_periods")

        return MSCIQueryParams.model_validate(uf)

    def _extract_period_info(self, params: MSCIQueryParams) -> Tuple[Optional[WolfPeriod], Optional[WolfPeriodRange], Optional[List[WolfPeriod]]]:
        """統一されたperiodパラメータから期間情報を抽出。
        
        Returns:
            (single_period, period_range, period_list)
        """
        period = params.period
        
        if period is None:
            return None, None, None
        
        if isinstance(period, WolfPeriod):
            return period, None, None
        elif isinstance(period, WolfPeriodRange):
            return None, period, None
        elif isinstance(period, list) and all(isinstance(p, WolfPeriod) for p in period):
            return None, None, period
        else:
            raise ValueError(f"サポートされていないperiod形式: {type(period)}")

    def get_index_constituents(
        self,
        params: Optional[MSCIQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[MSCISecurityRecord]:
        """インデックス構成銘柄レコードを高速取得。

        Args:
            params: 既存の `MSCIQueryParams` を直接指定。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - index_name: インデックス名（'WORLD', 'ACWI'）
                - country_code: 国コード（ローカルインデックス用）
                - dividend_flag: 配当込みインデックスフラグ
                - period: 期間指定（WolfPeriod）
                - batch_size: パフォーマンス制御

        Returns:
            取得・正規化済みの `MSCISecurityRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "MSCIインデックス構成銘柄データ取得開始: index_name=%s country_code=%s dividend_flag=%s batch_size=%d",
            params.index_name,
            params.country_code,
            params.dividend_flag,
            params.batch_size
        )
        
        df = self._query_constituents_data(params)
        records = self._create_security_records(df, params.batch_size)
        
        logger.info(
            "MSCIインデックス構成銘柄データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def get_index_values(
        self,
        params: Optional[MSCIQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[MSCIIndexRecord]:
        """インデックス値レコードを高速取得。

        Args:
            params: 既存の `MSCIQueryParams` を直接指定。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - index_name: インデックス名
                - country_code: 国コード
                - dividend_flag: 配当込みフラグ
                - period: 期間指定（WolfPeriod/WolfPeriodRange）
                - batch_size

        Returns:
            取得・整形・計算済みの `MSCIIndexRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "MSCIインデックス値データ取得開始: index_name=%s period=%s batch_size=%d",
            params.index_name,
            params.period,
            params.batch_size
        )
        
        df = self._query_index_values_data(params)
        records = self._create_index_records(df, params.batch_size)
        
        logger.info(
            "MSCIインデックス値データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def get_security_prices(
        self,
        params: Optional[MSCIQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[MSCISecurityRecord]:
        """証券価格レコードを高速取得。

        Args:
            params: 既存の `MSCIQueryParams` を直接指定。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - security_ids: 証券識別子リスト
                - identifier_type: 識別子タイプ
                - period: 期間指定（WolfPeriod/WolfPeriodRange）
                - batch_size

        Returns:
            取得・正規化済みの `MSCISecurityRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        if not params.security_ids:
            raise ValueError("security_ids は必須です")
        
        logger.info(
            "MSCI証券価格データ取得開始: security_ids=%d identifier_type=%s batch_size=%d",
            len(params.security_ids),
            params.identifier_type,
            params.batch_size
        )
        
        df = self._query_security_prices_data(params)
        records = self._create_security_records(df, params.batch_size)
        
        logger.info(
            "MSCI証券価格データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def get_beta(
        self,
        params: Optional[MSCIQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[BetaRecord]:
        """ベータ計算結果レコードを高速取得。

        Args:
            params: 既存の `MSCIQueryParams` を直接指定。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - security_ids: 証券識別子リスト
                - period: 期間指定（複数のWolfPeriodまたは単一のWolfPeriod）
                - lookback_periods: 遡及期間数
                - index_name: インデックス名
                - dividend_flag: 配当込みフラグ
                - frequency: 計算頻度
                - levered: レバードベータフラグ
                - batch_size

        Returns:
            計算・正規化済みの `MSCIBetaRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        if not params.security_ids:
            raise ValueError("security_ids は必須です")
        
        logger.info(
            "MSCIベータ計算開始: security_ids=%d lookback_periods=%d frequency=%s batch_size=%d",
            len(params.security_ids),
            params.lookback_periods,
            params.frequency,
            params.batch_size
        )
        
        records = self._calculate_beta_records(params)
        
        logger.info(
            "MSCIベータ計算完了: 計算件数=%d",
            len(records)
        )
        
        return records
    
    def _decide_index_code_and_name(
        self,
        index_name: Optional[str] = None,
        country_code: Optional[str] = None,
        dividend_flag: bool = False,
    ) -> Tuple[Optional[str], Optional[str]]:
        """インデックス指定からMSCIの国コードと名称を決定
        戻り値: (target_country_code, index_full_name)
        - グローバル: 特殊コード '@DP','@DT','@AP','@AT'
        - 国別: ISOアルファ alpha-3（例: 'JPN','USA'）に正規化
        """
        # 1) インデックス名が指定された場合は最優先で解決
        if index_name:
            index_key = index_name.strip().upper()
            index_info = self.MAJOR_INDICES.get(index_key)
            if index_info:
                country_codes = index_info['country_codes']
                keyword = '配当込み' if dividend_flag else '配当なし'
                target_country_code = next(
                    (code for code, desc in country_codes.items() if keyword in desc),
                    None,
                ) or index_info['default_code']
                return target_country_code, index_info.get('name', index_key)
            logger.error(f"未知のインデックス名: {index_name}")
            return None, index_name

        # 2) 国コード指定時は特殊コード（@）をそのまま受け、通常コードはalpha-3へ正規化
        if country_code:
            code_str = str(country_code).strip()
            if code_str.startswith('@'):
                return code_str, code_str
            alpha3 = convert_to_alpha3(code_str)
            if alpha3:
                return alpha3, alpha3
            logger.error(f"無効な国コード: {country_code}")
            return None, country_code

        # 3) どちらも無い場合はエラー
        raise ValueError("index_name または country_code を指定してください")

    def _country_filter_sql_template(self, target_country_code: Optional[str]) -> str:
        """国コードに応じた日次テーブル向けフィルタのSQLテンプレート。

        テーブル接頭辞（例: 'sec.' や 's2.'）を後から差し込めるよう、{alias} を含む
        書式文字列を返す。
        """
        if not target_country_code:
            return ""
        if target_country_code.startswith('@'):
            # グローバルインデックス
            if target_country_code in ('@DP', '@DT'):
                # WORLD: 先進国のみ
                # MSCI_D51_RAWテーブルからDM_FLAG=1の国を取得
                return """
                AND {alias}ISO_COUNTRY_SYMBOL IN (
                    SELECT DISTINCT ISO_COUNTRY_SYMBOL
                    FROM GIB_DB..MSCI_D51_RAW
                    WHERE DM_FLAG = 1
                    AND ISO_COUNTRY_SYMBOL NOT LIKE '@%'
                )
                """
            if target_country_code in ('@AP', '@AT'):
                # ACWI: 先進国＋新興国
                # MSCI_D51_RAWテーブルからDM_FLAG=1またはEM_FLAG=1の国を取得
                return """
                AND {alias}ISO_COUNTRY_SYMBOL IN (
                    SELECT DISTINCT ISO_COUNTRY_SYMBOL
                    FROM GIB_DB..MSCI_D51_RAW
                    WHERE (DM_FLAG = 1 OR EM_FLAG = 1)
                    AND ISO_COUNTRY_SYMBOL NOT LIKE '@%'
                )
                """
            return ""
        # 国別インデックス
        return f"AND {{alias}}ISO_COUNTRY_SYMBOL = '{target_country_code}'"

    def _build_country_filters_for_outer_and_subquery(self, target_country_code: Optional[str]) -> Tuple[str, str]:
        """外側クエリ（sec.）用とサブクエリ（s2.）用に展開したフィルタ文字列を返す。"""
        template = self._country_filter_sql_template(target_country_code)
        country_filter = template.format(alias='sec.') if template else ""
        country_filter_for_maxdate = template.format(alias='s2.') if template else ""
        return country_filter, country_filter_for_maxdate
    
    def _query_constituents_data(self, params: MSCIQueryParams) -> pd.DataFrame:
        """構成銘柄データを取得して DataFrame を返します。

        Args:
            params: フィルタ条件（インデックス名/国コード/配当フラグ/基準日など）。

        Returns:
            構成銘柄の `pandas.DataFrame`。
        """
        # インデックスの特定（共通化）
        target_country_code, index_full_name = self._decide_index_code_and_name(
            index_name=params.index_name,
            country_code=params.country_code,
            dividend_flag=params.dividend_flag,
        )
        logger.info(f"インデックス特定: {index_full_name} → {target_country_code}")
        
        try:
            # 国フィルタ（外側クエリ用とサブクエリ用を同時に生成）
            country_filter, country_filter_for_maxdate = self._build_country_filters_for_outer_and_subquery(target_country_code)

            # 期間情報の抽出
            single_period, period_range, period_list = self._extract_period_info(params)
            
            # 日付条件の設定
            if single_period:
                # 単一期間が指定された場合はその期間の最終日を使用
                target_date = single_period.end_date
                date_condition = f"AND sec.DATA_DATE = '{target_date}'"
            elif period_range:
                # 期間範囲が指定された場合は最終日を使用
                target_date = period_range.stop.end_date
                date_condition = f"AND sec.DATA_DATE = '{target_date}'"
            else:
                # 期間が指定されていない場合は最新日を取得
                date_condition = f"""
                AND sec.DATA_DATE = (
                    SELECT MAX(s2.DATA_DATE)
                    FROM GIB_DB..MSCI_DAILY_TRACKER_SECURITY s2
                    WHERE s2.FAMILY_STD_FLAG = 1
                    {country_filter_for_maxdate}
                )
                """
            
            sql = f"""
            SELECT DISTINCT
                sec.DATA_DATE,
                CAST(sec.MSCI_SECURITY_CODE AS VARCHAR(20)) AS MSCI_SECURITY_CODE,
                CAST(sec.MSCI_ISSUER_CODE AS VARCHAR(20)) AS MSCI_ISSUER_CODE,
                sec.CUSIP,
                sec.ISIN,
                sec.SEDOL,
                sec.ISO_COUNTRY_SYMBOL,
                sec.SECTOR,
                sec.INDUSTRY_GROUP,
                sec.INDUSTRY,
                sec.SUB_INDUSTRY,
                sec.ADJ_MARKET_CAP_USDOL as MARKET_CAP_USD,
                sec.PRICE,
                sec.PRICE_ISO_CURRENCY_SYMBOL as CURRENCY,
                sec.FOREIGN_INCLUSION_FACTOR,
                sec.DOMESTIC_INCLUSION_FACTOR
            FROM
                GIB_DB..MSCI_DAILY_TRACKER_SECURITY sec
            WHERE
                1=1
                {date_condition}
                AND sec.FAMILY_STD_FLAG = 1
                {country_filter}
            ORDER BY
                sec.ADJ_MARKET_CAP_USDOL DESC
            """
            
            df = self.execute_query(sql)
            
            # 企業コード情報を追加
            df = self._enrich_with_company_codes(df)
            logger.info(f"インデックス構成銘柄を取得: {len(df)}件 (country_code={target_country_code})")
            return df
            
        except Exception as e:
            logger.error(f"インデックス構成銘柄の取得エラー: {e}")
            raise
    
    def _query_index_values_data(self, params: MSCIQueryParams) -> pd.DataFrame:
        """インデックス値データを取得して DataFrame を返します。

        Args:
            params: フィルタ条件（インデックス名/国コード/配当フラグ/期間など）。

        Returns:
            インデックス値の `pandas.DataFrame`。
        """
        # インデックスの特定
        target_country_code, index_full_name = self._decide_index_code_and_name(
            index_name=params.index_name,
            country_code=params.country_code,
            dividend_flag=params.dividend_flag,
        )
        logger.info(f"インデックス特定: {index_full_name} → {target_country_code}")
        
        # 期間情報の抽出と条件構築
        single_period, period_range, period_list = self._extract_period_info(params)
        
        date_conditions = []
        if period_range:
            start_period = period_range.start
            end_period = period_range.stop if period_range.stop else period_range.start
            start_ym = start_period.y * 100 + (start_period.m or 1)
            end_ym = end_period.y * 100 + (end_period.m or 1)
            date_conditions.append(f"HISTORICAL_MONTH >= {start_ym}")
            date_conditions.append(f"HISTORICAL_MONTH <= {end_ym}")
        elif single_period:
            ym = single_period.y * 100 + (single_period.m or 1)
            date_conditions.append(f"HISTORICAL_MONTH = {ym}")
        elif period_list:
            # 複数期間の場合は範囲として扱う
            start_ym = min(p.y * 100 + (p.m or 1) for p in period_list)
            end_ym = max(p.y * 100 + (p.m or 1) for p in period_list)
            date_conditions.append(f"HISTORICAL_MONTH >= {start_ym}")
            date_conditions.append(f"HISTORICAL_MONTH <= {end_ym}")
        
        date_where = " AND " + " AND ".join(date_conditions) if date_conditions else ""
        
        try:
            sql = f"""
            SELECT
                CAST(CAST(HISTORICAL_MONTH AS VARCHAR(6)) + '01' AS DATE) AS DATA_DATE,
                ISO_COUNTRY_CODE,
                INDEX_DIVIDEND_FLG,
                INDEX_NAME,
                INDEX_VALUE,
                USD_RATE
            FROM GIB_DB..MSCI_MONTHLY_INDEX
            WHERE ISO_COUNTRY_CODE = '{target_country_code}'
            {date_where}
            ORDER BY DATA_DATE
            """

            df = self.execute_query(sql)

            if df.empty:
                logger.warning(f"月次インデックス値が空です (country_code={target_country_code})")
            else:
                # グローバルインデックスかどうかを判定
                is_major_index = target_country_code in self._major_index_codes
                
                # グローバルインデックスの場合、USD_RATEがNoneなら1.0で埋める
                if is_major_index:
                    if 'USD_RATE' in df.columns:
                        na_before = df['USD_RATE'].isna().sum()
                        if na_before > 0:
                            df['USD_RATE'] = df['USD_RATE'].fillna(1.0)
                            logger.debug(f"グローバルインデックス({target_country_code})のため、USD_RATEの欠損値{na_before}件を1.0で補完しました。")
                
                logger.info(f"月次インデックス値取得: {len(df)}件, 期間={df['DATA_DATE'].min()}~{df['DATA_DATE'].max()} ({target_country_code})")
                
                # 欠損値がある場合のみ警告
                na_index = df['INDEX_VALUE'].isna().sum()
                na_usd = df['USD_RATE'].isna().sum()
                if na_index > 0 or na_usd > 0:
                    logger.warning(f"インデックスデータに欠損値: INDEX_VALUE={na_index}, USD_RATE={na_usd}")
            
            return df

        except Exception as e:
            logger.error(f"インデックス値の取得エラー: {e}")
            raise
    
    def _query_security_prices_data(self, params: MSCIQueryParams) -> pd.DataFrame:
        """証券価格データを取得して DataFrame を返します。

        Args:
            params: フィルタ条件（証券識別子/期間など）。

        Returns:
            証券価格の `pandas.DataFrame`。
        """
        if not params.security_ids:
            raise ValueError("security_ids は必須です")
        
        id_column = str(params.identifier_type).strip().upper()
        if id_column not in {'ISIN', 'SEDOL', 'CUSIP'}:
            raise ValueError(f"無効な識別子タイプ: {params.identifier_type}")
        
        securities_str = "', '".join([str(s).replace("'", "''") for s in params.security_ids])
        
        # 期間情報の抽出と条件構築
        single_period, period_range, period_list = self._extract_period_info(params)
        
        date_conditions = []
        if period_range:
            start_date = period_range.start.start_date
            end_date = period_range.stop.end_date if period_range.stop else period_range.start.end_date
            date_conditions.append(f"DATA_DATE >= '{start_date}'")
            date_conditions.append(f"DATA_DATE <= '{end_date}'")
        elif single_period:
            start_date = single_period.start_date
            end_date = single_period.end_date
            date_conditions.append(f"DATA_DATE >= '{start_date}'")
            date_conditions.append(f"DATA_DATE <= '{end_date}'")
        elif period_list:
            # 複数期間の場合は範囲として扱う
            start_date = min(p.start_date for p in period_list)
            end_date = max(p.end_date for p in period_list)
            date_conditions.append(f"DATA_DATE >= '{start_date}'")
            date_conditions.append(f"DATA_DATE <= '{end_date}'")
        
        date_where = " AND " + " AND ".join(date_conditions) if date_conditions else ""
        
        try:
            sql = f"""
            SELECT
                DATA_DATE,
                CAST(MSCI_SECURITY_CODE AS VARCHAR(20)) AS MSCI_SECURITY_CODE,
                ISIN,
                SEDOL,
                CUSIP,
                ISO_COUNTRY_SYMBOL,
                PRICE,
                PRICE_ISO_CURRENCY_SYMBOL as CURRENCY,
                ADJ_MARKET_CAP_USDOL as MARKET_CAP_USD
            FROM
                GIB_DB..MSCI_DAILY_TRACKER_SECURITY
            WHERE
                {id_column} IN ('{securities_str}')
                {date_where}
            ORDER BY
                {id_column}, DATA_DATE
            """
            
            df = self.execute_query(sql)
            
            if not df.empty:
                logger.info(f"証券価格取得: {len(df)}件, 証券数={df[id_column].nunique()}/{len(params.security_ids)}, 期間={df['DATA_DATE'].min()}~{df['DATA_DATE'].max()}")
                # 詳細はdebugレベルで（最初の3銘柄のみ）
                for sec in params.security_ids[:3]:
                    sec_data = df[df[id_column] == sec]
                    if not sec_data.empty:
                        logger.debug(f"{sec}: データ数={len(sec_data)}")
            else:
                logger.warning(f"証券価格データが取得できませんでした: securities={len(params.security_ids)}件")
            
            return df
            
        except Exception as e:
            logger.error(f"証券価格取得エラー: {e}")
            raise
    
    def _calculate_beta_records(self, params: MSCIQueryParams) -> List[BetaRecord]:
        """ベータ計算結果レコードを生成します。

        Args:
            params: ベータ計算パラメータ。

        Returns:
            `BetaRecord` のリスト。
        """
        if not params.security_ids:
            raise ValueError("security_ids は必須です")
        
        # 入力のリスト化
        securities_list = params.security_ids
        
        # 期間情報の抽出
        single_period, period_range, period_list = self._extract_period_info(params)
        
        if period_list:
            periods_list = period_list
        elif single_period:
            periods_list = [single_period]
        elif period_range:
            # 期間範囲の場合は最終期間のみ使用（ベータ計算には特定の基準日が必要）
            periods_list = [period_range.stop]
        else:
            periods_list = []
        
        if not securities_list or not periods_list:
            logger.warning("証券IDリストまたは基準期間リストが空です。")
            return []

        # メインの計算処理を呼び出す
        df = self._get_betas(
            securities=securities_list,
            as_of_periods=periods_list,
            lookback_periods=params.lookback_periods,
            identifier_type=params.identifier_type,
            index_name=params.index_name,
            country_code=params.country_code,
            dividend_flag=params.dividend_flag,
            levered=params.levered,
            batch_size=params.batch_size,
        )

        if df.empty:
            return []
        
        # DataFrameをレコードに変換
        records = []
        for _, row in df.iterrows():
            try:
                # インデックスの特定
                target_country_code, index_full_name = self._decide_index_code_and_name(
                    index_name=params.index_name,
                    country_code=params.country_code,
                    dividend_flag=params.dividend_flag,
                )
                
                # ベータタイプの判定
                beta_type = "global" if str(target_country_code).startswith('@') else "local"
                
                # WolfPeriod作成
                retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                
                record = BetaRecord(
                    security_id=row['security_id'],
                    identifier_type=params.identifier_type,
                    index_code=target_country_code,
                    index_name=index_full_name,
                    as_of_date=periods_list[0].to_date() if periods_list else datetime.now(timezone.utc).date(),
                    lookback_periods=params.lookback_periods,
                    beta_value=row.get('beta_value'),
                    r_squared=row.get('r_squared'),
                    standard_error=row.get('standard_error'),
                    observations=row.get('observations'),
                    frequency=params.frequency,
                    levered=params.levered,
                    beta_type=beta_type,
                    error_message=row.get('error_message'),
                    retrieved_period=retrieved_period,
                )
                records.append(record)
            except Exception as e:
                logger.debug(f"ベータレコード作成エラー: {e}")
        
        return records
    
    def _get_betas(
        self,
        securities: List[str],
        as_of_periods: List[WolfPeriod],
        lookback_periods: int,
        identifier_type: str = 'isin',
        *,
        index_name: Optional[str] = None,
        country_code: Optional[str] = None,
        dividend_flag: bool = False,
        levered: bool = True,
        batch_size: int = 1000,
    ) -> pd.DataFrame:
        """複数の基準期間に対してベータを効率的に計算してDataFrameで返す
        - 必要な価格データとインデックスデータを一度のクエリでまとめて取得
        - 各証券・各基準期間の組み合わせでベータを計算
        """
        # 1. インデックスの特定
        target_country_code, index_full_name = self._decide_index_code_and_name(
            index_name=index_name,
            country_code=country_code,
            dividend_flag=dividend_flag,
        )

        # 2. データ取得のための全体期間を計算
        if not as_of_periods:
            return pd.DataFrame()
        
        # 最も新しい基準期間から全体期間を計算
        max_as_of_period = max(as_of_periods)
        start_period = max_as_of_period + (-lookback_periods)
        period_range = WolfPeriodRange(start_period, max_as_of_period)
        min_start_date = period_range.start.start_date
        period_end_date = period_range.stop.end_date if period_range.stop else period_range.start.end_date

        # 3. データの一括取得
        unique_securities = list(dict.fromkeys(securities))
        id_col = str(identifier_type).strip().upper()
        logger.info(
            f"ベータ計算開始: 銘柄数={len(unique_securities)}, "
            f"基準期間数={len(as_of_periods)}, 全体期間={min_start_date}~{period_end_date}"
        )

        prices_df = self._query_security_prices_data(
            MSCIQueryParams(
                security_ids=unique_securities,
                identifier_type=identifier_type,
                period=period_range,
                index_name=index_name,
                country_code=country_code,
                dividend_flag=dividend_flag,
                batch_size=batch_size
            )
        )
        
        if prices_df.empty:
            logger.warning("指定された全銘柄・全期間で価格データが取得できませんでした。")
            return pd.DataFrame()

        index_values = self._query_index_values_data(
            MSCIQueryParams(
                index_name=index_name,
                country_code=country_code,
                dividend_flag=dividend_flag,
                period=period_range,
                batch_size=batch_size
            )
        )

        if index_values.empty:
            logger.error(f"インデックス値が空のためベータ計算を中止します: {index_full_name}")
            return pd.DataFrame()

        # 4. 全体期間でのリターン計算
        # 4.1 インデックスリターン
        index_returns = (
            BetaCalculator.compute_index_returns_from_values(index_values, frequency='monthly')
            .rename('index_return')
        )

        # 4.2 各証券の月次リターン
        all_security_returns: Dict[str, pd.Series] = {}
        if not prices_df.empty:
            prices_df = prices_df.dropna(subset=['DATA_DATE', 'PRICE'])
            prices_df['DATA_DATE'] = pd.to_datetime(prices_df['DATA_DATE'])
            
            for security in unique_securities:
                grp = prices_df[prices_df[id_col] == security]
                if grp.empty:
                    continue
                
                series = (
                    grp.sort_values('DATA_DATE')
                    .drop_duplicates(subset=['DATA_DATE'])
                    .set_index('DATA_DATE')['PRICE']
                )
                
                daily = BetaCalculator.compute_daily_returns_from_prices(series)
                sec_monthly = (
                    BetaCalculator.compute_monthly_compounded_returns_from_daily(daily)
                    .rename('security_return')
                )
                sec_monthly.index = sec_monthly.index.to_period('M').to_timestamp('M')
                all_security_returns[security] = sec_monthly

        # 5. 各基準日・各証券の組み合わせでベータを計算
        all_results: List[Dict] = []
        beta_type = BetaType.GLOBAL if str(target_country_code).startswith('@') else BetaType.LOCAL
        
        total_calculations = len(as_of_periods) * len(unique_securities)
        completed_count = 0
        log_interval = max(1, total_calculations // 20)  # 5%ごと

        for as_of_period in sorted(as_of_periods, reverse=True):
            # 計算期間を取得
            start_period = as_of_period + (-lookback_periods)
            period_range = WolfPeriodRange(start_period, as_of_period)
            calc_start_date = period_range.start.start_date
            calc_end_date = period_range.stop.end_date if period_range.stop else period_range.start.end_date
            period_index_returns = index_returns.loc[calc_start_date:calc_end_date]

            for security in unique_securities:
                completed_count += 1
                if completed_count % log_interval == 0 or completed_count == total_calculations:
                    logger.info(f"計算進捗: {completed_count}/{total_calculations} ({completed_count*100//total_calculations}%)")

                if security not in all_security_returns:
                    continue

                period_sec_returns = all_security_returns[security].loc[calc_start_date:calc_end_date]
                merged = BetaCalculator.merge_returns(
                    period_sec_returns.to_frame(), period_index_returns.to_frame()
                )

                result = BetaCalculator.calculate_from_returns(
                    returns=merged,
                    security_id=security,
                    index_code=target_country_code,
                    index_name=index_full_name,
                    frequency=ReturnFrequency.MONTHLY,
                    beta_type=beta_type,
                    levered=levered,
                )

                if result:
                    res_dict = result.to_dict()
                    all_results.append(res_dict)
        
        logger.info(f"ベータ計算完了: {len(all_results)}件の結果を生成しました。")

        no_data_securities = [s for s in unique_securities if s not in all_security_returns]
        if no_data_securities:
            sample = no_data_securities[:10]
            suffix = f" 他{len(no_data_securities)-10}件" if len(no_data_securities) > 10 else ""
            logger.warning(f"価格データなし: {sample}{suffix}")
            
        calculated_count = len(all_results)
        total_attempts = len(as_of_periods) * (len(unique_securities) - len(no_data_securities))
        failed_count = total_attempts - calculated_count
        if failed_count > 0:
            logger.warning(f"計算試行 {total_attempts} 回のうち、{failed_count} 回が失敗しました（観測数不足など）。")

        if not all_results:
            return pd.DataFrame()
            
        final_df = pd.DataFrame(all_results)
        cols = ['security_id'] + [c for c in final_df.columns if c not in ['security_id']]
        return final_df[cols]
    
    def _create_security_records(self, df: pd.DataFrame, batch_size: int) -> List[MSCISecurityRecord]:
        """証券情報の `DataFrame` をレコードにバッチ変換します。

        Args:
            df: 証券情報の `DataFrame`。
            batch_size: バッチサイズ（検証/変換の単位）。

        Returns:
            `MSCISecurityRecord` のリスト。
        """
        if df.empty:
            return []
        
        records: List[MSCISecurityRecord] = []
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
                    
                    record = MSCISecurityRecord(
                        msci_security_code=row.get("MSCI_SECURITY_CODE"),
                        msci_issuer_code=row.get("MSCI_ISSUER_CODE"),
                        cusip=row.get("CUSIP"),
                        isin=row.get("ISIN"),
                        sedol=row.get("SEDOL"),
                        country_code=row.get("ISO_COUNTRY_SYMBOL"),
                        sector=row.get("SECTOR"),
                        industry_group=row.get("INDUSTRY_GROUP"),
                        industry=row.get("INDUSTRY"),
                        sub_industry=row.get("SUB_INDUSTRY"),
                        price=row.get("PRICE"),
                        currency=row.get("CURRENCY"),
                        market_cap_usd=row.get("MARKET_CAP_USD"),
                        foreign_inclusion_factor=row.get("FOREIGN_INCLUSION_FACTOR"),
                        domestic_inclusion_factor=row.get("DOMESTIC_INCLUSION_FACTOR"),
                        data_date=row.get("DATA_DATE"),
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    # 詳細な証券情報を含むエラーログ
                    security_info = {
                        "MSCI_SECURITY_CODE": row.get("MSCI_SECURITY_CODE"),
                        "ISIN": row.get("ISIN"),
                        "CUSIP": row.get("CUSIP"),
                        "SEDOL": row.get("SEDOL"),
                        "COUNTRY": row.get("ISO_COUNTRY_SYMBOL"),
                        "SECTOR": row.get("SECTOR"),
                        "PRICE": row.get("PRICE"),
                        "CURRENCY": row.get("CURRENCY"),
                        "DATA_DATE": str(row.get("DATA_DATE"))
                    }
                    # None値を除外してログを見やすくする
                    filtered_info = {k: v for k, v in security_info.items() if v is not None and v != ""}
                    
                    logger.debug(
                        "MSCI証券レコード検証エラー: security_data=%s error=%s",
                        filtered_info,
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "MSCI証券データ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records
    
    def _create_index_records(self, df: pd.DataFrame, batch_size: int) -> List[MSCIIndexRecord]:
        """インデックス情報の `DataFrame` をレコードにバッチ変換します。

        Args:
            df: インデックス情報の `DataFrame`。
            batch_size: バッチサイズ（検証/変換の単位）。

        Returns:
            `MSCIIndexRecord` のリスト。
        """
        if df.empty:
            return []
        
        records: List[MSCIIndexRecord] = []
        validation_errors = 0
        
        # バッチ処理による最適化
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # 並列バリデーション
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    # インデックスタイプの判定
                    index_code = row.get("ISO_COUNTRY_CODE")
                    is_global = index_code in self._major_index_codes
                    index_type = IndexType.GLOBAL if is_global else IndexType.LOCAL
                    
                    # 配当タイプの判定
                    dividend_flag = row.get("INDEX_DIVIDEND_FLG", 0)
                    dividend_type = DividendType.NET if dividend_flag else DividendType.GROSS
                    
                    # WolfPeriod作成
                    retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                    
                    record = MSCIIndexRecord(
                        index_code=index_code,
                        index_name=row.get("INDEX_NAME"),
                        index_type=index_type,
                        dividend_type=dividend_type,
                        country_code=row.get("ISO_COUNTRY_CODE") if not is_global else None,
                        index_value=row.get("INDEX_VALUE"),
                        usd_rate=row.get("USD_RATE"),
                        data_date=row.get("DATA_DATE"),
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    # 詳細なインデックス情報を含むエラーログ
                    index_info = {
                        "ISO_COUNTRY_CODE": row.get("ISO_COUNTRY_CODE"),
                        "INDEX_NAME": row.get("INDEX_NAME"),
                        "INDEX_DIVIDEND_FLG": row.get("INDEX_DIVIDEND_FLG"),
                        "INDEX_VALUE": row.get("INDEX_VALUE"),
                        "USD_RATE": row.get("USD_RATE"),
                        "DATA_DATE": str(row.get("DATA_DATE"))
                    }
                    # None値を除外してログを見やすくする
                    filtered_info = {k: v for k, v in index_info.items() if v is not None and v != ""}
                    
                    logger.debug(
                        "MSCIインデックスレコード検証エラー: index_data=%s error=%s",
                        filtered_info,
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "MSCIインデックスデータ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records
    
    def _enrich_with_company_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """企業コード情報でデータを充実させる
        
        ISINコードを基にFactSetやその他の企業コードを追加
        """
        if df.empty:
            raise ValueError("企業コード情報の付加対象データが空です")
            
        enriched_df = df.copy()
        
        # ISINベースでの企業情報追加
        if 'ISIN' in df.columns:
            company_info_list = []
            unique_isins = df['ISIN'].dropna().unique()
            
            for isin in unique_isins:
                if pd.notna(isin) and isin.strip():
                    try:
                        company = self.company_catalog.get(isin, 'isin')
                        if company:
                            company_info_list.append({
                                'ISIN': isin,
                                'FSYM_ID': company.fsym_id,
                                'FACTSET_ENTITY_ID': company.factset_entity_id,
                                'COMPANY_NAME': company.company_name,
                                'TICKER': company.ticker
                            })
                    except Exception as e:
                        logger.debug(f"企業情報取得エラー (ISIN={isin}): {e}")
            
            if company_info_list:
                company_df = pd.DataFrame(company_info_list)
                enriched_df = enriched_df.merge(company_df, on='ISIN', how='left')
                logger.info(f"企業コード情報を追加: {len(company_info_list)}件")
        
        # SEDOLベースでの企業情報追加（ISINが無い場合のフォールバック）
        if 'SEDOL' in df.columns and 'FSYM_ID' not in enriched_df.columns:
            company_info_list = []
            unique_sedols = df['SEDOL'].dropna().unique()
            
            for sedol in unique_sedols:
                if pd.notna(sedol) and sedol.strip():
                    try:
                        company = self.company_catalog.get(sedol, 'sedol')
                        if company:
                            company_info_list.append({
                                'SEDOL': sedol,
                                'FSYM_ID': company.fsym_id,
                                'FACTSET_ENTITY_ID': company.factset_entity_id,
                                'COMPANY_NAME': company.company_name,
                                'TICKER': company.ticker,
                                'ISIN': company.isin
                            })
                    except Exception as e:
                        logger.debug(f"企業情報取得エラー (SEDOL={sedol}): {e}")
            
            if company_info_list:
                company_df = pd.DataFrame(company_info_list)
                enriched_df = enriched_df.merge(company_df, on='SEDOL', how='left', suffixes=('', '_sedol'))
                logger.info(f"SEDOL経由で企業コード情報を追加: {len(company_info_list)}件")
        
        return enriched_df


__all__ = [
    "MSCIIndexRecord",
    "MSCISecurityRecord", 
    "MSCIQueryParams",
    "MSCIProvider",
    "IndexType",
    "DividendType",
] 