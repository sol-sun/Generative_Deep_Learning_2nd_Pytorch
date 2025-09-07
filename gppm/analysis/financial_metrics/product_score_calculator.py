"""
製品スコア計算クラス

セグメントスコアの計算、ピボットテーブルの作成、地理的情報の除去などを担当
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
import gc
from gppm.utils.geographic_processor import GeographicProcessor


class ProductScoreCalculator:
    """製品スコア計算クラス"""
    
    def __init__(self, geo_processor: GeographicProcessor):
        self.geo_processor = geo_processor
    
    def calculate_segment_scores(self, financial_data: pd.DataFrame, segment_mapping: pd.DataFrame, 
                               rbics_master: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
        """セグメント別スコアの計算"""
        
        # セグメント・財務データの統合
        merged_data = financial_data.merge(
            segment_mapping[["FACTSET_ENTITY_ID", "LABEL", "FISCAL_YEAR", 
                           "SALES", "SALES_ER", "OPINC", "ASSETS",
                           "OPINC_RATIO", "ASSETS_RATIO", "SALES_RATIO", "REVENUE_L6_ID"]],
            on=["FACTSET_ENTITY_ID", "FISCAL_YEAR"],
            how="inner"
        )
        
        # 売上比率の平均計算
        merged_data["SALES_RATIO"] = (
            merged_data.groupby(['FSYM_ID', 'FTERM_2', 'LABEL'])["SALES_RATIO"]
            .transform("mean")
        )
        merged_data.drop_duplicates(inplace=True)
        
        # セグメント別財務指標の計算
        merged_data["SEG_営業利益(税引後)"] = merged_data["営業利益(税引後)"] * merged_data["OPINC_RATIO"]
        merged_data["SEG_投下資本(運用ベース)"] = merged_data["投下資本(運用ベース)"] * merged_data["ASSETS_RATIO"]
        
        # 必要な列の選択
        result_columns = [
            "FSYM_ID", "FACTSET_ENTITY_ID", "FF_CO_NAME", "CURRENCY",
            "FTERM_2", "FISCAL_YEAR", "LABEL",
            "固定資産合計", "投下資本(運用ベース)",
            "SEG_営業利益(税引後)", "SEG_投下資本(運用ベース)",
            "SALES_ER", "SALES_RATIO", "REVENUE_L6_ID"
        ]
        
        merged_data = merged_data[result_columns]
        
        # REVENUE_L6_IDの展開
        merged_data["REVENUE_L6_ID"] = (
            merged_data["REVENUE_L6_ID"]
            .str.split(',')
            .apply(lambda ids: [s.strip() for s in ids] if isinstance(ids, list) else [])
        )
        merged_data = merged_data.explode("REVENUE_L6_ID")
        
        # マッピングデータの適用（必須）
        identity_mapping = merged_data[["FACTSET_ENTITY_ID", "REVENUE_L6_ID", "REVENUE_L6_ID"]].drop_duplicates()
        identity_mapping.columns = ["FACTSET_ENTITY_ID", "PRODUCT_L6_ID", "RELABEL_L6_ID"]
        
        combined_mapping = pd.concat([mapping_df, identity_mapping], axis=0).drop_duplicates()
        combined_mapping.reset_index(drop=True, inplace=True)
        
        # PRODUCT_L6_NAMEを作成するためのマージ
        combined_mapping = combined_mapping.merge(
            rbics_master[["L6_ID", "L6_NAME"]].rename(
                columns={"L6_ID": "PRODUCT_L6_ID", "L6_NAME": "PRODUCT_L6_NAME"}
            ),
            how="left",
            on="PRODUCT_L6_ID"
        )
        
        merged_data = merged_data.merge(
            combined_mapping.rename(columns={"RELABEL_L6_ID": "REVENUE_L6_ID"}),
            left_on=["FACTSET_ENTITY_ID", "REVENUE_L6_ID"],
            right_on=["FACTSET_ENTITY_ID", "REVENUE_L6_ID"],
            how="left"
        )
        
        # 依存度の計算
        merged_data = self._calculate_dependencies(merged_data)
        
        return merged_data
    
    def _calculate_dependencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """各種依存度の計算"""
        df = df.copy()
        
        # 売上比率依存度の計算
        df["SALES_RATIO_DEPENDENCY"] = (
            df.groupby(["FACTSET_ENTITY_ID", "FTERM_2", "LABEL"])["SALES_RATIO"]
            .transform(lambda x: x / x.sum())
        )
        
        # セグメント調整コードの作成
        df["SEGMENT_ADJ"] = df["LABEL"].str.strip().str.replace(" ", "_")
        df["CODE_SEGMENT_ADJ"] = df["FACTSET_ENTITY_ID"] + "_" + df["SEGMENT_ADJ"]
        
        # セグメントシェアの計算
        df["SEG_SHARE"] = df["SALES_RATIO_DEPENDENCY"] * df["SALES_RATIO"]
        
        # セグメント正規化依存度の計算
        df["セグメント正規化依存度"] = (
            df.groupby(["FACTSET_ENTITY_ID", "SEGMENT_ADJ", "FTERM_2"])["SEG_SHARE"]
            .transform(lambda x: x / x.sum())
        )
        
        # 企業全体の正規化依存度の計算
        df["正規化依存度"] = (
            df.groupby(["FACTSET_ENTITY_ID", "FTERM_2"])["SALES_RATIO"]
            .transform(lambda x: x / x.sum())
        )
        
        return df
    
    def create_pivot_tables(self, df: pd.DataFrame, rbics_master: pd.DataFrame, 
                          currency_filter: Optional[str] = 'USD') -> Dict[str, pd.DataFrame]:
        """ピボットテーブルの作成"""
        
        # 通貨フィルタ（オプショナル）
        if currency_filter is not None:
            filtered_df = df.query(f"CURRENCY == '{currency_filter}'")
        else:
            filtered_df = df
        
        # L6レベルのピボットテーブル（必ずPRODUCT_L6_NAMEを使用）
        product_name_col = 'PRODUCT_L6_NAME'
        
        # セグメント別製品スコア
        segment_product_score = pd.pivot_table(
            filtered_df, 
            index=['CODE_SEGMENT_ADJ', 'FTERM_2'], 
            values='セグメント正規化依存度',
            columns=[product_name_col], 
            fill_value=0
        )
        
        # 連結企業製品スコア
        consol_product_score = pd.pivot_table(
            filtered_df, 
            index=['FACTSET_ENTITY_ID', 'FTERM_2'], 
            values='正規化依存度',
            columns=[product_name_col], 
            fill_value=0, 
            aggfunc='sum'
        )
        
        # 階層レベルでの集約
        level_mappings = self._create_level_mappings(rbics_master, product_name_col)
        
        result = {
            'segment_l6': segment_product_score,
            'consol_l6': consol_product_score
        }
        
        # L5, L4レベルの集約
        for level in ['L5', 'L4']:
            if level in level_mappings:
                mapping = level_mappings[level]
                
                result[f'segment_{level.lower()}'] = (
                    segment_product_score.rename(columns=mapping)
                    .groupby(level=0, axis=1).sum()
                )
                
                result[f'consol_{level.lower()}'] = (
                    consol_product_score.rename(columns=mapping)
                    .groupby(level=0, axis=1).sum()
                )
        
        return result
    
    def _create_level_mappings(self, rbics_master: pd.DataFrame, 
                             product_name_col: str) -> Dict[str, Dict]:
        """階層レベルマッピングの作成"""
        mappings = {}
        
        if product_name_col == 'PRODUCT_L6_NAME':
            # L6からL5へのマッピング
            mappings['L5'] = (
                rbics_master.drop_duplicates(subset=["L6_NAME"]).set_index('L6_NAME')["L5_NAME"].to_dict()
            )
            # L6からL4へのマッピング
            mappings['L4'] = (
                rbics_master.drop_duplicates(subset=["L6_NAME"]).set_index('L6_NAME')["L4_NAME"].to_dict()
            )
        
        return mappings
    
    def remove_geographic_info(self, pivot_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """地理的情報の除去"""
        processed_tables = {}
        
        for name, df in pivot_tables.items():
            print(f"{name}: {len(df.columns)} 列")
            
            # 地理的情報の除去
            column_mapping = {col: self.geo_processor.remove_geographic_info(col) for col in df.columns}
            processed_df = df.rename(columns=column_mapping).groupby(level=0, axis=1).sum()
            
            print(f"{name} (処理後): {len(processed_df.columns)} 列")
            processed_tables[name] = processed_df
        
        return processed_tables
    
    def save_pivot_tables(self, pivot_tables: Dict[str, pd.DataFrame], 
                         output_dir: str = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202505/"):
        """ピボットテーブルの保存"""
        for name, df in pivot_tables.items():
            # ファイル保存
            filename = f"{output_dir}{name}_product_share.pkl"
            df.to_pickle(filename)
            print(f"保存完了: {filename}")
            
            # メモリ解放
            del df
            gc.collect()


