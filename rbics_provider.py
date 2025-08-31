"""
RBICSデータプロバイダー
=====================

目的
- FactSet REVEREデータベースからRBICS（セクター体系）データを取得・管理
- WolfPeriod/WolfPeriodRangeによる一貫した期間管理
- 大規模データ向けのバッチ/並列/キャッシュ最適化

主要コンポーネント
- `RBICSProvider`: 取得・整形・最適化を担う高水準API
- `RBICSStructureRecord`: RBICS構造マスタのPydanticモデル
- `RBICSCompanyRecord`: 企業RBICS情報のPydanticモデル
- `RBICSQueryParams`: フィルタ、期間、性能チューニングを表すクエリモデル

使用例
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
        company_ids=["123456789", "987654321"],
        min_revenue_share=0.05
    )
"""

from __future__ import annotations

from datetime import datetime, timezone, date
from typing import List, Optional, Dict, Any, Union, Tuple
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from pydantic import ValidationError

from wolf_period import WolfPeriod, WolfPeriodRange
from gppm.utils.config_manager import get_logger
from gppm.utils.data_processor import DataProcessor
from .rbics_types import (
    SegmentType,
    RBICSStructureRecord,
    RBICSCompanyRecord, 
    RBICSQueryParams,
)

logger = get_logger(__name__)


class FactSetRevereTable:
    """FactSet REVEREテーブル名定数。"""
    RBICS_STRUCTURE = 'FACTSET_REVERE..RBICS2_STRUCTURE_L6'
    COMPANY = 'FACTSET_REVERE..COMPANY'
    COMPANY_ADDRESS = 'FACTSET_REVERE..COMPANY_ADDRESS'
    COMPANY_FACTSET = 'FACTSET_REVERE..COMPANY_FACTSET'
    COMPANY_HQ = 'FACTSET_REVERE..COMPANY_HQ'
    BUS_SEG_REPORT = 'FACTSET_REVERE..COMPANY_RBICS2_BUS_SEG_REPORT'
    BUS_SEG_ITEM = 'FACTSET_REVERE..COMPANY_RBICS2_BUS_SEG_ITEM'
    FOCUS_L6 = 'FACTSET_REVERE..COMPANY_RBICS2_FOCUS_L6'
    REGION = 'FACTSET_REVERE..REGION'


class RBICSProvider(DataProcessor):
    """高速かつWolfPeriod対応のRBICS統合データプロバイダー。

    概要
    ----
    FactSet REVEREデータベースからRBICS分類データを高速かつ安全に取得するプロバイダーです。
    期間フィルタ・地域マッピング・企業識別子統合を含み、大規模データセットに対応します。

    主要機能
    --------
    - RBICS構造マスタデータの高速取得
    - 企業のRBICS売上セグメント情報の取得
    - 企業のRBICSフォーカス情報の取得
    - WolfPeriod/WolfPeriodRangeによる期間フィルタリング
    - 地域・証券識別子によるフィルタリング
    - バッチ処理・並列化・キャッシュによる性能最適化

    主要メソッド
    ------------
    - get_structure_records(): RBICS構造マスタ取得
    - get_company_records(): 企業RBICS情報取得
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._validation_cache: Dict[str, Any] = {}
        logger.debug("RBICSProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[RBICSQueryParams], kwargs: Dict[str, Any]) -> RBICSQueryParams:
        """ユーザフレンドリーなキーワードをRBICSQueryParamsへ正規化。"""
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        
        # エイリアス処理とリスト昇格
        if "company_ids" not in uf and "company_id" in uf:
            uf["company_ids"] = uf.pop("company_id")
        if "company_ids" in uf and isinstance(uf["company_ids"], str):
            uf["company_ids"] = [uf["company_ids"]]
        
        if "region_codes" not in uf:
            if "region_code" in uf:
                uf["region_codes"] = uf.pop("region_code")
            elif "region" in uf:
                uf["region_codes"] = uf.pop("region")
        if "region_codes" in uf and isinstance(uf["region_codes"], str):
            uf["region_codes"] = [uf["region_codes"]]
        
        if "security_ids" in uf and isinstance(uf["security_ids"], str):
            uf["security_ids"] = [uf["security_ids"]]
        
        if "segment_types" not in uf and "segment_type" in uf:
            uf["segment_types"] = uf.pop("segment_type")
        if "segment_types" in uf and isinstance(uf["segment_types"], (str, SegmentType)):
            uf["segment_types"] = [uf["segment_types"]]
        
        # 後方互換性: as_of_date -> period
        if "period" not in uf and "as_of_date" in uf and isinstance(uf["as_of_date"], date):
            d = uf.pop("as_of_date")
            uf["period"] = WolfPeriod.from_day(d)

        return RBICSQueryParams.model_validate(uf)

    def get_structure_records(
        self,
        params: Optional[RBICSQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[RBICSStructureRecord]:
        """RBICS構造マスタレコードを高速取得。"""
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "RBICS構造マスタデータ取得開始: period=%s batch_size=%d",
            params.period,
            params.batch_size
        )
        
        df = self._query_structure_data(params)
        records = self._create_structure_records(df, params.batch_size)
        
        logger.info("RBICS構造マスタデータ取得完了: 取得件数=%d", len(records))
        return records
    
    def get_company_records(
        self,
        params: Optional[RBICSQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[RBICSCompanyRecord]:
        """企業RBICSレコードを高速取得。"""
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "RBICS企業データ取得開始: segment_types=%s company_ids=%s batch_size=%d",
            params.segment_types,
            len(params.company_ids) if params.company_ids else 0,
            params.batch_size
        )
        
        all_records = []
        segment_types = params.segment_types or [SegmentType.REVENUE, SegmentType.FOCUS]
        
        for segment_type in segment_types:
            if segment_type == SegmentType.REVENUE and params.include_revenue_segments:
                df = self._query_revenue_segment_data(params)
                records = self._create_company_records(df, params.batch_size, SegmentType.REVENUE)
                all_records.extend(records)
            
            elif segment_type == SegmentType.FOCUS and params.include_focus_segments:
                df = self._query_focus_segment_data(params)
                records = self._create_company_records(df, params.batch_size, SegmentType.FOCUS)
                all_records.extend(records)
        
        logger.info("RBICS企業データ取得完了: 取得件数=%d", len(all_records))
        return all_records
    
    def _extract_period_info(self, params: RBICSQueryParams) -> Tuple[Optional[WolfPeriod], Optional[WolfPeriodRange]]:
        """統一されたperiodパラメータから期間情報を抽出。"""
        period = params.period
        
        if period is None:
            if params.as_of_date:
                single_period = WolfPeriod.from_day(params.as_of_date)
                return single_period, None
            if params.period_start or params.period_end:
                start = params.period_start
                end = params.period_end
                if start and end:
                    return None, WolfPeriodRange(start, end)
                elif start:
                    return start, None
                elif end:
                    return end, None
            return None, None
        
        if isinstance(period, WolfPeriod):
            return period, None
        elif isinstance(period, WolfPeriodRange):
            return None, period
        else:
            raise ValueError(f"サポートされていないperiod形式: {type(period)}")

    def _convert_period_to_sql_conditions(self, params: RBICSQueryParams) -> Tuple[str, str]:
        """期間パラメータをSQL条件文字列に変換。"""
        single_period, period_range = self._extract_period_info(params)
        
        if period_range:
            start_date = period_range.start.start_date
            end_date = period_range.stop.end_date if period_range.stop else period_range.start.end_date
        elif single_period:
            start_date = single_period.start_date
            end_date = single_period.end_date
        else:
            current_date = datetime.now(timezone.utc).date()
            start_date = end_date = current_date
        
        start_str = f"{start_date} 00:00:00"
        end_str = f"{end_date} 23:59:59"
        
        return start_str, end_str
    
    def _query_structure_data(self, params: RBICSQueryParams) -> pd.DataFrame:
        """RBICS構造マスタデータを取得。"""
        start_condition, end_condition = self._convert_period_to_sql_conditions(params)
        
        try:
            sql = f"""
            SELECT
                L1_ID, L2_ID, L3_ID, L4_ID, L5_ID, L6_ID,
                L1_NAME, L2_NAME, L3_NAME, L4_NAME, L5_NAME, L6_NAME,
                L6_DESCR AS SECTOR_DESC,
                START$ AS EFFECTIVE_START,
                END$ AS EFFECTIVE_END
            FROM {FactSetRevereTable.RBICS_STRUCTURE}
            WHERE
                START$ <= '{end_condition}' AND '{start_condition}' < END$
            ORDER BY L1_ID, L2_ID, L3_ID, L4_ID, L5_ID, L6_ID
            """
            
            logger.debug("RBICS構造マスタクエリ実行: start=%s end=%s", start_condition, end_condition)
            df = self.db.execute_query(sql)
            
            if not df.empty:
                string_columns = ['L1_NAME', 'L2_NAME', 'L3_NAME', 'L4_NAME', 'L5_NAME', 'L6_NAME', 'SECTOR_DESC']
                for col in string_columns:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.strip()
                
                logger.info(f"RBICS構造マスタ取得: {len(df)}件")
            else:
                logger.warning("RBICS構造マスタデータが取得できませんでした")
            
            return df
            
        except Exception as e:
            logger.error(f"RBICS構造マスタ取得エラー: {e}")
            raise
    
    def _query_revenue_segment_data(self, params: RBICSQueryParams) -> pd.DataFrame:
        """売上セグメントデータを取得。"""
        start_condition, end_condition = self._convert_period_to_sql_conditions(params)
        
        # WHERE句の構築
        where_conditions = []
        sql_params = []
        
        if params.company_ids:
            placeholders = ",".join(["%s"] * len(params.company_ids))
            where_conditions.append(f"A.COMPANY_ID IN ({placeholders})")
            sql_params.extend(params.company_ids)
        
        if params.min_revenue_share is not None:
            where_conditions.append("B.REVENUE_PERCENT >= %s")
            sql_params.append(params.min_revenue_share * 100)
        
        if params.exclude_zero_revenue:
            where_conditions.append("B.REVENUE_PERCENT > 0")
        
        additional_where = " AND " + " AND ".join(where_conditions) if where_conditions else ""
        
        try:
            sql = f"""
            SELECT DISTINCT
                D.ID AS COMPANY_ID,               -- FactSet定義の企業ID
                G.FS_ENTITY_ID AS FACTSET_ENTITY_ID, -- 企業名 (en)
                D.NAME AS COMPANY_NAME,           -- その他ID
                D.SEDOL, D.TICKER, D.ISIN, D.CUSIP,  -- 会社の法的所在地
                E.NAME AS REGION_NAME,            -- HQまたは法的所在地名
                D.HOME_REGION AS REGION_CODE,     -- HQ所在地コード
                C.COUNTRY AS HQ_REGION_CODE,      -- HQ所在地コード
                A.PERIOD_END_DATE,                -- レポートの会計終了日付
                B.ID AS SEGMENT_ID,               -- 開示文書由来のFactSet定義のセグメントID
                B.NAME AS SEGMENT_NAME,           -- 開示文書由来のFactSet定義のセグメント名
                B.REVENUE_PERCENT AS SEG_SHARE,   -- セグメントごとのセールスシェア
                B.RBICS2_L6_ID AS REVENUE_L6_ID   -- RBICS(ID) 第6レベルのセクター
            FROM {FactSetRevereTable.BUS_SEG_REPORT} A
            INNER JOIN (
                SELECT
                    COMPANY_ID,
                    MAX(PERIOD_END_DATE) AS PERIOD_END_DATE,
                    MAX(START$) AS START$
                FROM {FactSetRevereTable.BUS_SEG_REPORT}
                WHERE START$ < CONVERT(DATETIME, '{end_condition}')
                  AND END$ > CONVERT(DATETIME, '{start_condition}')
                  AND START$ < CONVERT(DATETIME, '{start_condition}') - 550
                GROUP BY COMPANY_ID
            ) F ON A.COMPANY_ID = F.COMPANY_ID
               AND A.PERIOD_END_DATE = F.PERIOD_END_DATE
               AND A.START$ = F.START$
            INNER JOIN {FactSetRevereTable.BUS_SEG_ITEM} B
                ON A.ID = B.REPORT_ID
               AND B.START$ < CONVERT(DATETIME, '{end_condition}')
               AND B.END$ > CONVERT(DATETIME, '{start_condition}')
            INNER JOIN {FactSetRevereTable.COMPANY} D
                ON A.COMPANY_ID = D.ID
               AND D.START$ < CONVERT(DATETIME, '{end_condition}')
               AND D.END$ > CONVERT(DATETIME, '{start_condition}')
               AND D.COVERED = 'Y'
            LEFT OUTER JOIN {FactSetRevereTable.COMPANY_ADDRESS} C
                ON D.ID = C.COMPANY_ID
               AND C.HQ = 'Y'
               AND C.START$ < CONVERT(DATETIME, '{end_condition}')
               AND C.END$ > CONVERT(DATETIME, '{start_condition}')
            INNER JOIN {FactSetRevereTable.REGION} E -- #TODO: COUNTRYとの違いを調べておく
                ON D.HOME_REGION = E.ID
               AND E.START$ < CONVERT(DATETIME, '{end_condition}')
               AND E.END$ > CONVERT(DATETIME, '{start_condition}')
            INNER JOIN {FactSetRevereTable.COMPANY_FACTSET} G
                ON D.ID = G.COMPANY_ID
               AND G.START$ < CONVERT(DATETIME, '{end_condition}')
               AND G.END$ > CONVERT(DATETIME, '{start_condition}')
            WHERE 1=1{additional_where}
            ORDER BY D.ID, B.REVENUE_PERCENT DESC
            """
            
            logger.debug("RBICS売上セグメントクエリ実行")
            df = self.db.execute_query(sql, params=sql_params)
            
            if not df.empty:
                if 'SEG_SHARE' in df.columns:
                    df['SEG_SHARE'] = pd.to_numeric(df['SEG_SHARE'], errors='coerce') / 100
                
                logger.info(f"RBICS売上セグメント取得: {len(df)}件")
            else:
                logger.warning("RBICS売上セグメントデータが取得できませんでした")
            
            return df
            
        except Exception as e:
            logger.error(f"RBICS売上セグメント取得エラー: {e}")
            raise
    
    def _query_focus_segment_data(self, params: RBICSQueryParams) -> pd.DataFrame:
        """フォーカスセグメントデータを取得。"""
        start_condition, end_condition = self._convert_period_to_sql_conditions(params)
        
        where_conditions = []
        sql_params = []
        
        if params.company_ids:
            placeholders = ",".join(["%s"] * len(params.company_ids))
            where_conditions.append(f"A.COMPANY_ID IN ({placeholders})")
            sql_params.extend(params.company_ids)
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        try:
            sql = f"""
            SELECT
                A.COMPANY_ID,
                B.NAME AS COMPANY_NAME,
                B.HOME_REGION AS REGION_CODE,
                A.RBICS2_L6_ID AS RBICS_L6_ID,
                C.COUNTRY AS HQ_REGION_CODE,
                D.FS_ENTITY_ID AS FACTSET_ENTITY_ID,
                B.SEDOL, B.TICKER, B.ISIN, B.CUSIP
            FROM {FactSetRevereTable.FOCUS_L6} AS A
            LEFT OUTER JOIN {FactSetRevereTable.COMPANY} AS B 
                ON A.COMPANY_ID = B.ID
                AND B.START$ < CONVERT(DATETIME, '{end_condition}')
                AND B.END$ > CONVERT(DATETIME, '{start_condition}')
            LEFT OUTER JOIN {FactSetRevereTable.COMPANY_ADDRESS} AS C 
                ON B.ID = C.COMPANY_ID 
                AND C.HQ = 'Y'
                AND C.START$ < CONVERT(DATETIME, '{end_condition}')
                AND C.END$ > CONVERT(DATETIME, '{start_condition}')
            LEFT OUTER JOIN {FactSetRevereTable.COMPANY_FACTSET} AS D
                ON A.COMPANY_ID = D.COMPANY_ID
                AND D.START$ < CONVERT(DATETIME, '{end_condition}')
                AND D.END$ > CONVERT(DATETIME, '{start_condition}')
            WHERE 
                A.START$ < CONVERT(DATETIME, '{end_condition}')
                AND A.END$ > CONVERT(DATETIME, '{start_condition}')
                AND {where_clause}
            ORDER BY A.COMPANY_ID
            """
            
            logger.debug("RBICSフォーカスセグメントクエリ実行")
            df = self.db.execute_query(sql, params=sql_params)
            
            if not df.empty:
                logger.info(f"RBICSフォーカスセグメント取得: {len(df)}件")
            else:
                logger.warning("RBICSフォーカスセグメントデータが取得できませんでした")
            
            return df
            
        except Exception as e:
            logger.error(f"RBICSフォーカスセグメント取得エラー: {e}")
            raise
    
    def _create_structure_records(self, df: pd.DataFrame, batch_size: int) -> List[RBICSStructureRecord]:
        """RBICS構造マスタのDataFrameをレコードにバッチ変換。"""
        if df.empty:
            return []
        
        records: List[RBICSStructureRecord] = []
        validation_errors = 0
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                    
                    record = RBICSStructureRecord(
                        l1_id=row.get("L1_ID"),
                        l2_id=row.get("L2_ID"),
                        l3_id=row.get("L3_ID"),
                        l4_id=row.get("L4_ID"),
                        l5_id=row.get("L5_ID"),
                        l6_id=row.get("L6_ID"),
                        l1_name=row.get("L1_NAME"),
                        l2_name=row.get("L2_NAME"),
                        l3_name=row.get("L3_NAME"),
                        l4_name=row.get("L4_NAME"),
                        l5_name=row.get("L5_NAME"),
                        l6_name=row.get("L6_NAME"),
                        sector_description=row.get("SECTOR_DESC"),
                        effective_start=row.get("EFFECTIVE_START").date() if row.get("EFFECTIVE_START") is not None else None,
                        effective_end=row.get("EFFECTIVE_END").date() if row.get("EFFECTIVE_END") is not None else None,
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    logger.debug(
                        "RBICS構造マスタレコード検証エラー: l6_id=%s error=%s",
                        row.get("L6_ID"),
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "RBICS構造マスタデータ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records
    
    def _create_company_records(self, df: pd.DataFrame, batch_size: int, segment_type: SegmentType) -> List[RBICSCompanyRecord]:
        """企業RBICSのDataFrameをレコードにバッチ変換。"""
        if df.empty:
            return []
        
        records: List[RBICSCompanyRecord] = []
        validation_errors = 0
        
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                    
                    revenue_share = None
                    if segment_type == SegmentType.REVENUE and 'SEG_SHARE' in row:
                        revenue_share = row.get("SEG_SHARE")
                        if pd.notna(revenue_share):
                            revenue_share = float(revenue_share)
                    
                    # RBICSのL6 IDを適切に取得
                    rbics_l6_id = None
                    if segment_type == SegmentType.REVENUE:
                        rbics_l6_id = row.get("REVENUE_L6_ID")
                    else:
                        rbics_l6_id = row.get("RBICS_L6_ID")
                    
                    record = RBICSCompanyRecord(
                        company_id=row.get("COMPANY_ID"),
                        factset_entity_id=row.get("FACTSET_ENTITY_ID"),
                        company_name=row.get("COMPANY_NAME"),
                        cusip=row.get("CUSIP"),
                        isin=row.get("ISIN"),
                        sedol=row.get("SEDOL"),
                        ticker=row.get("TICKER"),
                        region_name=row.get("REGION_NAME"),
                        region_code=row.get("REGION_CODE"),
                        hq_region_code=row.get("HQ_REGION_CODE"),
                        segment_type=segment_type,
                        rbics_l6_id=rbics_l6_id,
                        segment_id=row.get("SEGMENT_ID") if segment_type == SegmentType.REVENUE else None,
                        segment_name=row.get("SEGMENT_NAME") if segment_type == SegmentType.REVENUE else None,
                        revenue_share=revenue_share,
                        period_end_date=row.get("PERIOD_END_DATE").date() if row.get("PERIOD_END_DATE") is not None else None,
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    logger.debug(
                        "RBICS企業レコード検証エラー: company_id=%s error=%s",
                        row.get("COMPANY_ID"),
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "RBICS企業データ変換完了: 有効レコード=%d 検証エラー=%d segment_type=%s",
                len(records),
                validation_errors,
                segment_type.value
            )
        
        return records


__all__ = [
    "FactSetRevereTable",
    "RBICSProvider",
]