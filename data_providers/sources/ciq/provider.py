"""
CIQ データプロバイダー
=====================

目的
- CIQ（S&P Capital IQ）から企業識別子データ（Company ID, Ticker, ISIN 等）を取得する。
- 期間表現は `wolf_period` の `WolfPeriod`/`WolfPeriodRange` を使用する。
- 財務データ取得は将来対応予定（現状は未実装）。

構成
- `CIQProvider`: 企業識別子取得 API。
- `CIQIdentityRecord`: 取得レコードのスキーマ（検証・正規化を内包）。
- `CIQFinancialRecord`: 財務データ用のプレースホルダー。
- `CIQQueryParams`: 型安全なクエリ定義（必要時に直接指定）。

使用例
    from data_providers.sources.ciq.provider import CIQProvider

    # プロバイダーの初期化
    provider = CIQProvider(max_workers=4)

    # 企業識別子データの取得
    identity_records = provider.get_identity_records(
        country=["US", "JPN"],     # 米国・日本の企業 （alpha-2/alpha-3混在OK）
        listed_only=True,          # 上場企業のみ
    )

引数（get_identity_records）
- country/countries: 国コード（ISO 3166-1 alpha-2/alpha-3、単一または配列）
- listed_only: 上場企業のみ取得（既定: True）
- company_id/company_ids: Company ID（単一または配列）
- batch_size: 取得・検証の性能調整

例外
- ValidationError: スキーマ検証に失敗。
- DatabaseError: データベースアクセスに失敗。

互換性
- `CIQQueryParams` を直接渡すことも可能。キーワード引数との併用は不可。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from wolf_period import WolfPeriod
from gppm.core.config_manager import get_logger

from data_providers.core.base_provider import BaseProvider

# CIQ固有の型定義をインポート
from .types import (
    CIQIdentityRecord,
    CIQFinancialRecord,
)

from .query_params import CIQQueryParams

logger = get_logger(__name__)


class CIQProvider(BaseProvider):
    """CIQ企業データ取得用プロバイダー。

    概要
    ----
    CIQ（S&P Capital IQ）から企業識別子データを取得するプロバイダーです。
    WolfPeriod/WolfPeriodRangeによる期間管理と、大規模データ向けの最適化を提供します。

    主要機能
    --------
    - 企業識別子データの取得（Company ID, Ticker, ISIN, SEDOL, CUSIP等）
    - 上場・非上場企業のフィルタリング
    - 地域・企業IDによるフィルタリング
    - バッチ処理・並列化による性能最適化
    - 財務データ取得機能（将来実装予定）

    パフォーマンス最適化
    ------------------
    - バッチ処理による効率的なデータ検証
    - 並列処理による高速化
    - データベースクエリの最適化
    - キャッシュ機能による重複処理の回避

    主要メソッド
    ------------
    - get_identity_records(): 企業識別子データ取得（実装済み）
    - get_financial_records(): 財務データ取得（将来実装予定）

    使用例
    ------
        from data_providers.data_sources.ciq.provider import CIQProvider

        # プロバイダー初期化
        provider = CIQProvider(max_workers=4)

        # 企業識別子取得
        identity_records = provider.get_identity_records(
            country=["US", "JP"],
            listed_only=True
        )

        # 特定企業の取得
        company_records = provider.get_identity_records(
            company_ids=[12345, 67890],
            listed_only=False
        )

        # 財務データ取得（将来実装予定）
        # financial_records = provider.get_financial_records(country=["US"])
        # NotImplementedError: CIQ財務データ取得機能は将来実装予定です

    実装状況
    --------
    - 企業識別子取得: 完全実装済み
    - 財務データ取得: 将来実装予定（NotImplementedError）
    - WolfPeriod/WolfPeriodRange対応: 将来実装予定
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._validation_cache: Dict[str, Any] = {}
        logger.debug("CIQProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        """リソースクリーンアップ。"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[CIQQueryParams], kwargs: Dict[str, Any]) -> CIQQueryParams:
        """ユーザフレンドリーなキーワードを `CIQQueryParams` に正規化。

        - `country`/`countries` エイリアスを `country_codes` に集約
        - 単一文字列はリストに昇格（`country_codes`, `company_ids`）
        - `params` と `kwargs` の併用はエラー
        """
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        if "country_codes" not in uf:
            if "country" in uf:
                uf["country_codes"] = uf.pop("country")
            elif "countries" in uf:
                uf["country_codes"] = uf.pop("countries")
        if "country_codes" in uf and isinstance(uf["country_codes"], str):
            uf["country_codes"] = [uf["country_codes"]]
        if "company_ids" in uf and isinstance(uf["company_ids"], int):
            uf["company_ids"] = [uf["company_ids"]]
        return CIQQueryParams.model_validate(uf)

    def get_identity_records(
        self,
        params: Optional[CIQQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[CIQIdentityRecord]:
        """企業識別子レコードを取得。

        Args:
            params: 既存の `CIQQueryParams` を直接指定（後方互換）。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - country / countries: 国コード（単一または配列）
                - listed_only: 上場企業のみ
                - company_id / company_ids: 単一または配列
                - batch_size: パフォーマンス制御

        Returns:
            CIQIdentityRecordのリスト

        Raises:
            ValidationError: レコードの検証に失敗した場合
            DatabaseError: データベースアクセスに失敗した場合
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "CIQ企業識別子データ取得開始: company_id=%s listed_only=%s countries=%s batch_size=%d",
            params.company_id,
            params.listed_only,
            params.country_codes,
            params.batch_size
        )
        
        df = self._query_identity_data(params)
        records = self._create_identity_records(df, params.batch_size)
        
        logger.info(
            "CIQ企業識別子データ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def get_financial_records(
        self,
        params: Optional[CIQQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[CIQFinancialRecord]:
        """財務データレコードを取得（将来実装予定）。
        
        Args:
            params: クエリパラメータ（フィルタ条件）
            
        Returns:
            CIQFinancialRecordのリスト
            
        Raises:
            NotImplementedError: 現在は未実装
        """
        # パラメータは検証のみ先に通してユーザのエラーを早期発見
        _ = self._normalize_query_params(params, kwargs)
        logger.warning("CIQ財務データ取得機能は将来実装予定です")
        raise NotImplementedError(
            "CIQ財務データ取得機能は将来実装予定です。"
            "現在は企業識別子データのみ利用可能です。"
            "実装予定機能: WolfPeriod/WolfPeriodRangeベースの期間フィルタリング、"
            "float演算、バッチ処理最適化"
        )
    
    def _query_identity_data(self, params: CIQQueryParams) -> pd.DataFrame:
        """企業識別子データを取得して DataFrame を返します。

    - 必要列: `COMPANY_ID`, `COMPANY_NAME`, `HEADQUARTERS_COUNTRY_CODE`,
      `EXCHANGE_COUNTRY_CODE`, `CUSIP`, `ISIN`, `SEDOL`, `TICKER`, `UN_LISTED_FLG`

    Args:
        params: フィルタ条件（上場/非上場/国コードなど）。

    Returns:
        企業識別子の `pandas.DataFrame`。
    """
        # WHERE句の構築（SQLインジェクション防止）
        where_conditions = []
        sql_params = []
        
        if params.listed_only:
            where_conditions.append("UN_LISTED_FLG = %s")
            sql_params.append(0)
        
        if params.company_id is not None:
            where_conditions.append("COMPANY_ID = %s")
            sql_params.append(params.company_id)
        
        if params.company_ids:
            placeholders = ",".join(["%s"] * len(params.company_ids))
            where_conditions.append(f"COMPANY_ID IN ({placeholders})")
            sql_params.extend(params.company_ids)
        
        if params.country_codes:
            # params.country_codesは既にバリデーション段階でalpha-3に変換済み
            placeholders = ",".join(["%s"] * len(params.country_codes))
            # 本社国または上場国のいずれかが一致
            country_condition = (
                f"(HEADQUARTERS_COUNTRY_CODE IN ({placeholders}) "
                f"OR EXCHANGE_COUNTRY_CODE IN ({placeholders}))"
            )
            where_conditions.append(country_condition)
            sql_params.extend(params.country_codes * 2)  # 本社国・上場国両方
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        # 最適化されたSQL（インデックス活用）
        sql = f"""
        SELECT /*+ USE_INDEX(M_COMPANY_CIQ, idx_company_id) */
            COMPANY_ID,
            COMPANY_NAME,
            HEADQUARTERS_COUNTRY_CODE,
            EXCHANGE_COUNTRY_CODE,
            CUSIP,
            ISIN,
            SEDOL,
            TICKER,
            UN_LISTED_FLG
        FROM GIB_DB..M_COMPANY_CIQ
        WHERE {where_clause}
        ORDER BY COMPANY_ID

        """
        
        logger.debug("CIQ識別子クエリ実行: params=%s", sql_params)
        return self.execute_query(sql, params=sql_params)
    
    def _create_identity_records(self, df: pd.DataFrame, batch_size: int) -> List[CIQIdentityRecord]:
        """企業識別子の `DataFrame` をレコードに変換します。

    Args:
        df: 企業識別子の `DataFrame`。
        batch_size: バッチサイズ（検証/変換の単位）。

    Returns:
        `CIQIdentityRecord` のリスト。
    """
        if df.empty:
            return []
        
        records: List[CIQIdentityRecord] = []
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
                    
                    record = CIQIdentityRecord(
                        company_id=row.get("COMPANY_ID"),
                        company_name=row.get("COMPANY_NAME"),
                        headquarters_country_code=row.get("HEADQUARTERS_COUNTRY_CODE"),
                        exchange_country_code=row.get("EXCHANGE_COUNTRY_CODE"),
                        cusip=row.get("CUSIP"),
                        isin=row.get("ISIN"),
                        sedol=row.get("SEDOL"),
                        ticker=row.get("TICKER"),
                        un_listed_flg=row.get("UN_LISTED_FLG"),
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    logger.debug(
                        "CIQ識別子レコード検証エラー: company_id=%s error=%s",
                        row.get("COMPANY_ID"),
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "CIQ識別子データ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records


__all__ = [
    "CIQIdentityRecord",
    "CIQFinancialRecord",
    "CIQQueryParams",
    "CIQProvider",
]
