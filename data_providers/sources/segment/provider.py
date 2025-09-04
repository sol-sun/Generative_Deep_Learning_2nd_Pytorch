"""
セグメントデータプロバイダー

FactSetからセグメントデータを取得し、各種比率を計算
"""

import pandas as pd
import numpy as np
from data_providers.core.base_provider import BaseProvider


class SegmentDataProvider(BaseProvider):
    """セグメントデータの処理"""
    
    def get_segment_data(self) -> pd.DataFrame:
        """セグメントデータの取得"""
        sql = """
        SELECT FSYM_ID, DATE, FF_SEGMENT_NUM, CURRENCY, LABEL, SALES AS SALES_ER, OPINC, ASSETS, INTERSEG_REV AS SALES_IR
        FROM FACTSET_FEED.FF_V3.FF_SEGBUS_AF SEG
        WHERE SEG.DATE >= '2015-4-28'
        """
        
        df = self.execute_query(sql)
        return self._process_segment_data(df)
    
    def _process_segment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """セグメントデータの処理"""
        df = df.copy()
        df["LABEL_2"] = df["LABEL"].apply(self.normalize_text)
        
        # 会計期間の調整
        df = self.adjust_fiscal_term(df)
        
        # 売上の計算
        df["SALES"] = df["SALES_ER"] + df["SALES_IR"].fillna(0)
        
        # 売上高が正の企業のみ
        df = df.query("SALES > 0")
        
        # Reconciling Itemsの処理
        recon_df = df.query("LABEL == 'Reconciling Items'").reset_index(drop=True)
        recon_df = recon_df[["FSYM_ID", "FISCAL_YEAR", "SALES"]].drop_duplicates()
        recon_df = recon_df.rename(columns={"SALES": "RECON_SALES"})
        
        df = df.query("LABEL != 'Reconciling Items'").reset_index(drop=True)
        
        # 売上高・営業利益・資産の合計計算
        sum_cols = ["SALES", "SALES_ER", "OPINC", "ASSETS"]
        sum_df = (df.groupby(["FSYM_ID", "FTERM_2"])[sum_cols].sum()
                   .reset_index()
                   .rename(columns={col: f"{col}_SUM" for col in sum_cols}))
        
        df = df.merge(sum_df, on=["FSYM_ID", "FTERM_2"])
        df = df.merge(recon_df, how="left", on=["FSYM_ID", "FISCAL_YEAR"])
        
        # 比率の計算
        df = self._calculate_ratios(df)
        
        return df
    
    def _calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """各種比率の計算"""
        # ゼロを NaN に置換
        for col in ["SALES_SUM", "SALES_ER_SUM", "OPINC_SUM", "ASSETS_SUM"]:
            df[col] = df[col].replace(0, np.nan)
        
        # 比率計算
        df["SALES_RATIO"] = df["SALES"] / (df["SALES_ER_SUM"] + df["RECON_SALES"].fillna(0))
        df["SALES_ER_RATIO"] = df["SALES_ER"] / df["SALES_ER_SUM"]
        df["OPINC_RATIO"] = df["OPINC"] / df["OPINC_SUM"]
        df["ASSETS_RATIO"] = df["ASSETS"] / df["ASSETS_SUM"]
        
        # 必要な列のみ選択（entity情報はここではまだ含まれていない）
        columns = [
            "FSYM_ID", "LABEL", "LABEL_2", "DATE", "FTERM_2", "FISCAL_YEAR",
            "SALES", "SALES_ER", "SALES_IR", "OPINC", "ASSETS",
            "OPINC_RATIO", "ASSETS_RATIO", "SALES_RATIO", "SALES_ER_RATIO", "RECON_SALES"
        ]
        
        return df[columns]


