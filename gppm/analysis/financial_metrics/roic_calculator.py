"""
ROIC計算クラス

連結・セグメント別ROIC（投下資本利益率）の計算機能を提供
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import warnings


class ROICCalculator:
    """ROIC（投下資本利益率）計算クラス"""
    
    def __init__(self):
        self.invest_cap_col = "投下資本(運用ベース)"
        self.seg_invest_cap_col = "SEG_投下資本(運用ベース)"
        self.oper_inc_col = "営業利益(税引後)"
        self.seg_oper_inc_col = "SEG_営業利益(税引後)"
    
    def calculate_consol_roic(self, financial_pivot: pd.DataFrame) -> pd.DataFrame:
        """
        連結ROIC計算
        
        Parameters:
        -----------
        financial_pivot : pd.DataFrame
            ピボットテーブル形式の財務データ（FTERM_2がインデックス、企業IDがカラム）
            
        Returns:
        --------
        pd.DataFrame
            連結ROIC計算結果（企業ID×時期のピボットテーブル）
        """
        # 4四半期合計営業利益
        oper_inc_4q_sum = financial_pivot[self.oper_inc_col].rolling(window=4).sum()
        
        # 4四半期平均投下資本
        invest_4q_mean = financial_pivot[self.invest_cap_col].rolling(window=4, min_periods=1).mean()
        invest_4q_mean = invest_4q_mean.apply(lambda col: col.map(lambda x: np.nan if x < 0 else x))
        
        # ROIC計算
        common_columns = list(set(oper_inc_4q_sum.columns) & set(invest_4q_mean.columns))
        roic_4q_mean = (
            oper_inc_4q_sum[common_columns] / invest_4q_mean[common_columns]
        ).T
        
        # アウトライヤー除去
        roic_4q_mean = self._remove_outliers(roic_4q_mean)
        roic_4q_mean = roic_4q_mean.dropna(how='all', axis=0)
        
        return roic_4q_mean
    
    def calculate_segment_roic(self, segment_pivot: pd.DataFrame) -> pd.DataFrame:
        """
        セグメント別ROIC計算
        
        Parameters:
        -----------
        segment_pivot : pd.DataFrame
            セグメント別財務データのピボットテーブル
            
        Returns:
        --------
        pd.DataFrame
            セグメント別ROIC計算結果
        """
        # 4四半期合計セグメント営業利益
        seg_oper_inc_4q_sum = segment_pivot[self.seg_oper_inc_col].rolling(window=4).sum()
        
        # 4四半期平均セグメント投下資本
        seg_invest_4q_mean = segment_pivot[self.seg_invest_cap_col].rolling(window=4, min_periods=1).mean()
        seg_invest_4q_mean = seg_invest_4q_mean.apply(lambda col: col.map(lambda x: np.nan if x < 0 else x))
        
        # セグメントROIC計算
        common_columns = list(set(seg_oper_inc_4q_sum.columns) & set(seg_invest_4q_mean.columns))
        seg_roic_4q_mean = (
            seg_oper_inc_4q_mean[common_columns] / seg_invest_4q_mean[common_columns]
        ).T
        
        # アウトライヤー除去
        seg_roic_4q_mean = self._remove_outliers(seg_roic_4q_mean)
        seg_roic_4q_mean = seg_roic_4q_mean.dropna(how='all', axis=0)
        
        return seg_roic_4q_mean
    
    def _remove_outliers(self, df: pd.DataFrame, lower_quantile: float = 0.01, 
                        upper_quantile: float = 0.99) -> pd.DataFrame:
        """
        アウトライヤー除去（各時期ごとに上位・下位パーセンタイルの値をNaNに置換）
        
        Parameters:
        -----------
        df : pd.DataFrame
            対象データフレーム
        lower_quantile : float
            下位パーセンタイル閾値
        upper_quantile : float
            上位パーセンタイル閾値
            
        Returns:
        --------
        pd.DataFrame
            アウトライヤー除去後のデータフレーム
        """
        df_cleaned = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            # 上位と下位の閾値を計算
            upper_threshold = df[col].quantile(upper_quantile)
            lower_threshold = df[col].quantile(lower_quantile)
            
            # 条件に該当する値をNaNに置き換える
            df_cleaned[col] = df_cleaned[col].mask(
                (df_cleaned[col] > upper_threshold) | (df_cleaned[col] < lower_threshold)
            )
        
        return df_cleaned


