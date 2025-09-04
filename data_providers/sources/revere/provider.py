"""
REVERE統合データプロバイダー
==========================

目的
- FactSet REVEREデータベースからセグメントデータを取得・管理
- WolfPeriod/WolfPeriodRangeによる一貫した期間管理
- 大規模データ向けのバッチ/並列/キャッシュ最適化

主要コンポーネント
- `RevereDataProvider`: 取得・整形・最適化を担う高水準API
- `RevereRecord`: REVEREデータレコードのPydanticモデル
- `RevereQueryParams`: フィルタ、期間、性能チューニングを表すクエリモデル

使用例
    from data_providers.sources.revere.provider import RevereDataProvider
    from wolf_period import WolfPeriod, WolfPeriodRange

    # プロバイダーの初期化
    provider = RevereDataProvider(max_workers=4)

    # REVEREセグメントデータの取得
    revere_data = provider.get_revere_data(
        fsym_ids=["000C7F-E", "002D8G-F"],
        fiscal_year=2023,
        min_revenue_share=0.05
    )
"""

from __future__ import annotations

from datetime import datetime, timezone, date
from typing import List, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from wolf_period import WolfPeriod, WolfPeriodRange, Frequency
from gppm.core.config_manager import get_logger
from data_providers.core.base_provider import BaseProvider

# REVERE固有の型定義をインポート
from .types import RevereRecord
from .query_params import RevereQueryParams

logger = get_logger(__name__)


class RevereDataProvider(BaseProvider):
    """高速かつWolfPeriod対応のREVERE統合データプロバイダー。

    概要
    ----
    FactSet REVEREデータベースからセグメントデータを高速かつ安全に取得するプロバイダーです。
    期間フィルタ・地域マッピング・企業識別子統合を含み、大規模データセットに対応します。

    主要機能
    --------
    - REVEREセグメントデータの高速取得
    - セグメント名の正規化・処理
    - WolfPeriod/WolfPeriodRangeによる期間フィルタリング
    - 売上比率の計算・処理
    - バッチ処理・並列化・キャッシュによる性能最適化

    主要メソッド
    ------------
    - get_revere_data(): REVEREセグメントデータ取得
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.debug("RevereDataProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[RevereQueryParams], kwargs: Dict[str, Any]) -> RevereQueryParams:
        """ユーザフレンドリーなキーワードをRevereQueryParamsへ正規化。"""
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        
        # エイリアス処理とリスト昇格
        if "fsym_ids" not in uf and "fsym_id" in uf:
            uf["fsym_ids"] = uf.pop("fsym_id")
        if "fsym_ids" in uf and isinstance(uf["fsym_ids"], str):
            uf["fsym_ids"] = [uf["fsym_ids"]]

        return RevereQueryParams.model_validate(uf)

    def get_revere_data(
        self,
        params: Optional[RevereQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """REVEREセグメントデータを高速取得。"""
        params = self._normalize_query_params(params, kwargs)
        
        logger.info("REVEREセグメントデータ取得開始")
        
        df = self._query_revere_data(params)
        processed_df = self._process_revere_data(df)
        
        logger.info("REVEREセグメントデータ取得完了: 取得件数=%d", len(processed_df))
        return processed_df
    
    def _query_revere_data(self, params: RevereQueryParams) -> pd.DataFrame:
        """REVEREセグメントデータを取得。"""
        # WHERE句の構築
        where_conditions = ["A.PERIOD_END_DATE >= CONVERT(DATETIME, '2017-04-01 00:00:00')", "B.TYPE != 'N'"]
        sql_params = []
        
        if params.fsym_ids:
            placeholders = ",".join(["%s"] * len(params.fsym_ids))
            where_conditions.append(f"G.FS_ENTITY_ID IN ({placeholders})")
            sql_params.extend(params.fsym_ids)
        
        if params.fiscal_year:
            # 会計年度フィルタ（日本の会計年度: 4月始まり）
            start_date = f"{params.fiscal_year - 1}-04-01"
            end_date = f"{params.fiscal_year}-03-31"
            where_conditions.append("A.PERIOD_END_DATE >= CONVERT(DATETIME, %s)")
            where_conditions.append("A.PERIOD_END_DATE <= CONVERT(DATETIME, %s)")
            sql_params.extend([start_date, end_date])
        
        where_clause = " AND ".join(where_conditions)
        
        try:
            sql = f"""
            SELECT DISTINCT
                F.COMPANY_ID,
                G.FS_ENTITY_ID AS FACTSET_ENTITY_ID,
                D.NAME AS COMPANY_NAME,
                D.SEDOL, D.TICKER, D.ISIN, D.CUSIP,
                H.NAME AS REGION_NAME,
                H.DOME_REGION AS REGION_CODE,
                H.COUNTRY AS HQ_REGION_CODE,
                A.PERIOD_END_DATE,
                B.ID   AS SEGMENT_ID,
                B.NAME AS SEGMENT_NAME,
                B.TYPE,
                B.REVENUE_PERCENT AS SEG_SHARE,
                B.RBICS2_L6_ID AS REVENUE_L6_ID
            FROM FACTSET_REVERE.COMPANY_RBICS2_BUS_SEG_REPORT A
            INNER JOIN (
                SELECT
                    COMPANY_ID,
                    MAX(PERIOD_END_DATE) AS PERIOD_END_DATE,
                    MAX(STARTS) AS STARTS
                FROM FACTSET_REVERE.COMPANY_RBICS2_BUS_SEG_REPORT
                GROUP BY COMPANY_ID
            ) F ON A.COMPANY_ID = F.COMPANY_ID
            INNER JOIN FACTSET_REVERE.COMPANY_RBICS2_BUS_SEG_ITEM B
                ON A.ID = B.REPORT_ID
            INNER JOIN FACTSET_REVERE.COMPANY_FACTSET G
                ON A.COMPANY_ID = G.COMPANY_ID
            LEFT OUTER JOIN FACTSET_REVERE.COMPANY D
                ON A.COMPANY_ID = D.ID
            LEFT OUTER JOIN FACTSET_REVERE.REGION H
                ON D.HOME_REGION = H.ID
            WHERE {where_clause}
            ORDER BY G.FS_ENTITY_ID, A.PERIOD_END_DATE
            """
            
            df = self.execute_query(sql, params=sql_params)
            
            if not df.empty:
                logger.info(f"REVEREセグメント取得: {len(df)}件")
            else:
                logger.warning("REVEREセグメントデータが取得できませんでした")
            
            return df
            
        except Exception as e:
            logger.error(f"REVEREセグメント取得エラー: {e}")
            raise
    
    def _process_revere_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """REVEREデータの処理・正規化。"""
        if df.empty:
            return df
        df = df.copy()
        
        # セグメント名の正規化
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.strip()
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.replace(r"[—–]", "-", regex=True)
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.replace(r"(\w)\s", r"\1 ", regex=True)
        
        # 売上比率と会計年度の計算
        df["SALES_RATIO"] = df["SEG_SHARE"].astype(float) / 100
        
        # WolfPeriodを使用した会計年度計算
        from wolf_period import WolfPeriod, Frequency
        def get_fiscal_year_from_date(date):
            """日付から会計年度を取得"""
            date_obj = pd.to_datetime(date).date()
            period = WolfPeriod.from_day(date_obj, freq=Frequency.Y, fy_start_month=4)
            return period.y
        
        df["FISCAL_YEAR"] = df["PERIOD_END_DATE"].map(get_fiscal_year_from_date)
        
        # セグメントシェアの合計計算
        df["SEG_SHARE_SUM"] = (
            df.groupby(["FACTSET_ENTITY_ID", "FISCAL_YEAR"])["SEG_SHARE"].transform("sum")
        )
        
        df.sort_values(by=["FACTSET_ENTITY_ID", "FISCAL_YEAR", "SALES_RATIO"], inplace=True)
        
        # 冗長な行の除去とセグメント名の処理
        df = df.groupby(['FACTSET_ENTITY_ID', 'FISCAL_YEAR'], group_keys=False).apply(
            self._remove_redundant_rows
        )
        
        return df
    
    def _remove_redundant_rows(self, group) -> pd.DataFrame:
        """重複するセグメント行の除去。"""
        flag = group["SEG_SHARE"].sum() != 1
        indices_to_remove = []
        
        for idx, seg in group["SEGMENT_NAME"].items():
            search_str = seg + " "
            if not flag:
                if any(other_seg.startswith(search_str) for j, other_seg in group["SEGMENT_NAME"].items() if j != idx):
                    indices_to_remove.append(idx)
            
            for j, other_seg in group["SEGMENT_NAME"].items():
                if other_seg.startswith(search_str):
                    if not other_seg.startswith(seg + " - "):
                        updated_seg = seg + " - " + other_seg[len(seg):].lstrip()
                        group.at[j, "SEGMENT_NAME"] = updated_seg
        
        return group.drop(indices_to_remove)


