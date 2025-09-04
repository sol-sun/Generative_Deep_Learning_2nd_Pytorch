"""
REVEREデータプロバイダー
=====================

目的
- FactSet REVEREデータベースから企業セグメントデータを取得・管理
- WolfPeriodによる一貫した期間管理（会計年度ベース）
- セグメント名の正規化とデータ品質向上

主要コンポーネント
- `RevereDataProvider`: セグメントデータ取得・整形を担う高水準API
- `RevereRecord`: セグメントデータレコードのPydanticモデル
- `RevereQueryParams`: フィルタ、期間を表すクエリモデル

主要機能
- 企業セグメント情報の高速取得
- セグメント名の正規化と重複行の除去
- WolfPeriodによる会計年度計算
- セグメントシェアの集計・検証

使用例
    from data_providers.sources.revere.provider import RevereDataProvider
    from data_providers.sources.revere.query_params import RevereQueryParams
    from wolf_period import WolfPeriod

    # プロバイダーの初期化
    provider = RevereDataProvider()

    # セグメントデータの取得
    segment_data = provider.get_revere_data()

    # 特定の会計年度のデータ取得
    params = RevereQueryParams(fiscal_year=2023)
    filtered_data = provider.get_filtered_data(params)
"""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import date

import pandas as pd
from wolf_period import WolfPeriod, Frequency

from data_providers.core.base_provider import BaseProvider
from .types import RevereRecord
from .query_params import RevereQueryParams


class RevereDataProvider(BaseProvider):
    """FactSet REVEREデータベースからセグメントデータを高速取得するプロバイダー。

    概要
    ----
    FactSet REVEREデータベースから企業のセグメント情報を取得し、
    データ品質を向上させるための正規化・検証処理を行います。

    主要機能
    --------
    - 企業セグメント情報の高速取得
    - セグメント名の正規化処理
    - 重複行の除去とデータクリーニング
    - WolfPeriodによる会計年度計算
    - セグメントシェアの集計・検証

    主要メソッド
    ------------
    - get_revere_data(): 全セグメントデータ取得
    - get_filtered_data(): フィルタ条件付きデータ取得
    """
    
    def get_revere_data(self) -> pd.DataFrame:
        """REVEREセグメントデータの高速取得。

        FactSet REVEREデータベースから最新の企業セグメント情報を取得し、
        データ品質向上のための正規化処理を実行します。

        戻り値
        -------
        pd.DataFrame
            処理済みセグメントデータ。以下のカラムを含む：
            - FACTSET_ENTITY_ID: FactSet Entity ID
            - COMPANY_NAME: 企業名
            - SEGMENT_NAME: 正規化済みセグメント名
            - SALES_RATIO: 売上比率（0.0-1.0）
            - FISCAL_YEAR: 会計年度
            - REVENUE_L6_ID: RBICS L6分類ID

        注意事項
        -------
        - 2017年4月以降のデータのみ取得
        - セグメント名の正規化処理を自動実行
        - 重複行は自動除去
        """
        sql = """
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
        WHERE A.PERIOD_END_DATE >= CONVERT(DATETIME, '2017-04-01 00:00:00')
          AND B.TYPE != 'N'
        """
        
        df = self.execute_query(sql)
        return self._process_revere_data(df)
    
    def _process_revere_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """REVEREデータの正規化・品質向上処理。

        取得した生データに対して以下の処理を実行：
        1. セグメント名の正規化（文字クリーニング、空白調整）
        2. 売上比率の変換（パーセント → 小数）
        3. WolfPeriodによる会計年度計算
        4. セグメントシェア合計の計算
        5. 重複行の除去

        パラメータ
        ----------
        df : pd.DataFrame
            生のREVEREデータ

        戻り値
        -------
        pd.DataFrame
            正規化・クリーニング済みデータ
        """
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
    
    def _remove_redundant_rows(self, group):
        """重複するセグメント行の除去とセグメント名の階層化。

        同一企業・会計年度内で重複するセグメント行を検出し：
        1. より具体的なセグメント名を持つ行を優先
        2. 階層的なセグメント名を「親 - 子」形式に統一
        3. 冗長な行を除去

        パラメータ
        ----------
        group : pd.DataFrame
            企業・会計年度でグループ化されたデータ

        戻り値
        -------
        pd.DataFrame
            重複除去・階層化済みデータ
        """
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
    
    def get_filtered_data(
        self, 
        params: Optional[RevereQueryParams] = None,
        /,
        **kwargs: Any
    ) -> List[RevereRecord]:
        """フィルタ条件付きセグメントデータの取得。

        指定されたクエリパラメータに基づいてセグメントデータを取得し、
        Pydanticモデルのリストとして返します。

        パラメータ
        ----------
        params : RevereQueryParams, optional
            クエリパラメータ。指定しない場合はキーワード引数を使用
        **kwargs : Any
            キーワード引数でクエリパラメータを指定

        戻り値
        -------
        List[RevereRecord]
            フィルタ条件に一致するセグメントレコードのリスト

        例
        ----
        >>> provider = RevereDataProvider()
        >>> records = provider.get_filtered_data(
        ...     fiscal_year=2023,
        ...     fsym_ids=["ABC123-S", "DEF456-S"]
        ... )
        """
        # パラメータの正規化
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        
        if params is None:
            params = RevereQueryParams.model_validate(kwargs)
        
        # 基本データの取得
        df = self.get_revere_data()
        
        # フィルタの適用
        if params.fiscal_year is not None:
            df = df[df["FISCAL_YEAR"] == params.fiscal_year]
        
        if params.fsym_ids is not None:
            df = df[df["FACTSET_ENTITY_ID"].isin(params.fsym_ids)]
        
        # Pydanticレコードに変換
        records = []
        for _, row in df.iterrows():
            try:
                record = RevereRecord(
                    fsym_id=row.get("FACTSET_ENTITY_ID"),
                    segment_name=row.get("SEGMENT_NAME"),
                    revenue_share=row.get("SALES_RATIO"),
                    fiscal_year=row.get("FISCAL_YEAR")
                )
                records.append(record)
            except Exception as e:
                # バリデーションエラーをログに記録（実装に応じて）
                continue
        
        return records


