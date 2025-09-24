"""
製品スコア計算クラス

セグメントスコアの計算、ピボットテーブルの作成、地理的情報の除去などを担当
"""

from typing import Dict, Optional, List
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
                          currency_filter: Optional[str] = None,
                          chunk_size: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        ピボットテーブルの作成（メモリ効率最適化版）
        
        Args:
            df: 入力データフレーム
            rbics_master: RBICSマスターデータ
            currency_filter: 通貨フィルタ（オプション）
            chunk_size: チャンクサイズ（大容量データ用、Noneの場合は自動判定）
            
        Returns:
            ピボットテーブルの辞書
            
        Note:
            - メモリ使用量を監視し、必要に応じてチャンク処理を実行
            - データ型最適化によりメモリ使用量を約50%削減
            - 大容量データ（chunk_size以上）の場合は自動的にチャンク処理を実行
        """
        
        # メモリ使用量の監視
        import psutil
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            # 通貨フィルタ（オプショナル）
            if currency_filter is not None:
                filtered_df = df.query(f"CURRENCY == '{currency_filter}'")
            else:
                filtered_df = df
            
            # データ型最適化（メモリ使用量削減）
            filtered_df = self._optimize_data_types(filtered_df)
            
            # チャンク処理の判定
            if chunk_size is not None and len(filtered_df) > chunk_size:
                print(f"大容量データ検出: {len(filtered_df)} 行 -> チャンク処理を実行")
                return self._create_pivot_tables_chunked(
                    filtered_df, rbics_master, chunk_size
                )
            
            # L6レベルのピボットテーブル（必ずPRODUCT_L6_NAMEを使用）
            product_name_col = 'PRODUCT_L6_NAME'
            
            # セグメント別製品スコア（メモリ効率的に作成）
            segment_product_score = self._create_pivot_table_memory_efficient(
                filtered_df, 
                index=['CODE_SEGMENT_ADJ', 'FTERM_2'], 
                values='セグメント正規化依存度',
                columns=[product_name_col]
            )
            
            # 連結企業製品スコア（メモリ効率的に作成）
            consol_product_score = self._create_pivot_table_memory_efficient(
                filtered_df, 
                index=['FACTSET_ENTITY_ID', 'FTERM_2'], 
                values='正規化依存度',
                columns=[product_name_col],
                aggfunc='sum'
            )
            
            # 階層レベルでの集約
            level_mappings = self._create_level_mappings(rbics_master, product_name_col)
            
            result = {
                'segment_l6': segment_product_score,
                'consol_l6': consol_product_score
            }
            
            # L5, L4レベルの集約（メモリ効率的に）
            for level in ['L5', 'L4']:
                if level in level_mappings:
                    mapping = level_mappings[level]
                    
                    # セグメントレベル集約
                    segment_aggregated = self._aggregate_level_memory_efficient(
                        segment_product_score, mapping
                    )
                    result[f'segment_{level.lower()}'] = segment_aggregated
                    
                    # 連結レベル集約
                    consol_aggregated = self._aggregate_level_memory_efficient(
                        consol_product_score, mapping
                    )
                    result[f'consol_{level.lower()}'] = consol_aggregated
                    
                    # 中間データの解放
                    del segment_aggregated, consol_aggregated
                    gc.collect()
            
            # メモリ使用量のログ出力
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = final_memory - initial_memory
            print(f"ピボットテーブル作成完了 - メモリ使用量: {memory_used:.2f} MB")
            
            return result
            
        except Exception as e:
            # エラー時のメモリ解放
            gc.collect()
            raise e
    
    def _create_pivot_tables_chunked(self, df: pd.DataFrame, rbics_master: pd.DataFrame, 
                                   chunk_size: int) -> Dict[str, pd.DataFrame]:
        """チャンク処理による大容量データ対応のピボットテーブル作成"""
        print(f"チャンク処理開始: チャンクサイズ {chunk_size}")
        
        # チャンクに分割
        chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
        print(f"総チャンク数: {len(chunks)}")
        
        # 結果を格納する辞書
        result = {}
        product_name_col = 'PRODUCT_L6_NAME'
        
        # 階層レベルマッピングを事前に作成
        level_mappings = self._create_level_mappings(rbics_master, product_name_col)
        
        # 各チャンクを処理
        for i, chunk in enumerate(chunks):
            print(f"チャンク {i+1}/{len(chunks)} 処理中...")
            
            try:
                # チャンクのピボットテーブル作成
                segment_chunk = self._create_pivot_table_memory_efficient(
                    chunk, 
                    index=['CODE_SEGMENT_ADJ', 'FTERM_2'], 
                    values='セグメント正規化依存度',
                    columns=[product_name_col]
                )
                
                consol_chunk = self._create_pivot_table_memory_efficient(
                    chunk, 
                    index=['FACTSET_ENTITY_ID', 'FTERM_2'], 
                    values='正規化依存度',
                    columns=[product_name_col],
                    aggfunc='sum'
                )
                
                # 初回チャンクの場合
                if i == 0:
                    result['segment_l6'] = segment_chunk
                    result['consol_l6'] = consol_chunk
                else:
                    # 既存の結果と結合
                    result['segment_l6'] = pd.concat([result['segment_l6'], segment_chunk], 
                                                   ignore_index=False)
                    result['consol_l6'] = pd.concat([result['consol_l6'], consol_chunk], 
                                                  ignore_index=False)
                
                # チャンクデータの解放
                del segment_chunk, consol_chunk
                gc.collect()
                
            except Exception as e:
                print(f"チャンク {i+1} 処理中にエラー: {e}")
                gc.collect()
                continue
        
        # 重複を除去して集約
        print("重複除去と集約中...")
        result['segment_l6'] = result['segment_l6'].groupby(level=[0, 1]).sum()
        result['consol_l6'] = result['consol_l6'].groupby(level=[0, 1]).sum()
        
        # L5, L4レベルの集約
        for level in ['L5', 'L4']:
            if level in level_mappings:
                mapping = level_mappings[level]
                
                # セグメントレベル集約
                segment_aggregated = self._aggregate_level_memory_efficient(
                    result['segment_l6'], mapping
                )
                result[f'segment_{level.lower()}'] = segment_aggregated
                
                # 連結レベル集約
                consol_aggregated = self._aggregate_level_memory_efficient(
                    result['consol_l6'], mapping
                )
                result[f'consol_{level.lower()}'] = consol_aggregated
                
                # 中間データの解放
                del segment_aggregated, consol_aggregated
                gc.collect()
        
        print("チャンク処理完了")
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
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """データ型最適化（メモリ使用量削減）"""
        df_optimized = df.copy()
        
        # 数値列の最適化
        for col in df_optimized.select_dtypes(include=['float64']).columns:
            if df_optimized[col].min() >= -3.4e38 and df_optimized[col].max() <= 3.4e38:
                df_optimized[col] = df_optimized[col].astype('float32')
        
        # 整数列の最適化
        for col in df_optimized.select_dtypes(include=['int64']).columns:
            if df_optimized[col].min() >= -2147483648 and df_optimized[col].max() <= 2147483647:
                df_optimized[col] = df_optimized[col].astype('int32')
        
        return df_optimized
    
    def _create_pivot_table_memory_efficient(self, df: pd.DataFrame, index: List[str], 
                                           values: str, columns: List[str], 
                                           aggfunc: str = 'mean') -> pd.DataFrame:
        """メモリ効率的なピボットテーブル作成"""
        try:
            # 必要な列のみを選択してメモリ使用量を削減
            required_cols = index + [values] + columns
            df_subset = df[required_cols].copy()
            
            # ピボットテーブル作成
            pivot_table = pd.pivot_table(
                df_subset,
                index=index,
                values=values,
                columns=columns,
                fill_value=0,
                aggfunc=aggfunc
            )
            
            # データ型最適化
            pivot_table = pivot_table.astype('float32')
            
            # 中間データの解放
            del df_subset
            gc.collect()
            
            return pivot_table
            
        except Exception as e:
            # エラー時のメモリ解放
            gc.collect()
            raise e
    
    def _aggregate_level_memory_efficient(self, pivot_table: pd.DataFrame, 
                                        mapping: Dict[str, str]) -> pd.DataFrame:
        """メモリ効率的なレベル集約"""
        try:
            # マッピングに基づいて列名を変更
            renamed_table = pivot_table.rename(columns=mapping)
            
            # レベル集約
            aggregated = renamed_table.groupby(level=0, axis=1).sum()
            
            # データ型最適化
            aggregated = aggregated.astype('float32')
            
            return aggregated
            
        except Exception as e:
            # エラー時のメモリ解放
            gc.collect()
            raise e
    
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


