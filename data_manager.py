"""
FactSetデータマネージャー

各種データプロバイダーを統合し、セグメントマッピング処理を実行
"""

from typing import Dict
import pandas as pd
import numpy as np
import warnings
from gppm.providers.factset_provider import FactSetProvider
from gppm.providers.segment_data_provider import SegmentDataProvider
from gppm.providers.revere_data_provider import RevereDataProvider
from gppm.providers.rbics_provider import RBICSProvider
from gppm.finance.geographic_processor import GeographicProcessor
from gppm.utils.data_processor import DataProcessor


class FactSetDataManager:
    """
    FactSetデータの統合処理クラス
    
    各種データプロバイダーを統合し、セグメントマッピング処理を実行します。
    """
    
    def __init__(self):
        self.financial_provider = FactSetProvider()
        self.segment_provider = SegmentDataProvider()
        self.revere_provider = RevereDataProvider()
        self.rbics_provider = RBICSProvider()
        self.geo_processor = GeographicProcessor()
    
    def initialize_data(self) -> Dict[str, pd.DataFrame]:
        """
        初期データの取得・処理
        
        各種データプロバイダーから基本データを取得し、統合処理を実行します。
        
        Returns:
            統合・処理されたデータの辞書
                - entity: 企業エンティティデータ
                - financial: 財務データ（重複除去済み）
                - segment: セグメントデータ（重複除去済み）
                - revere: REVEREデータ
                - rbics_master: RBICSマスターデータ
        """
        print("データの取得を開始...")
        
        # 基本データの取得
        entity_data = self._get_entity_data()
        financial_data = self._get_financial_data()
        segment_data = self.segment_provider.get_segment_data()
        revere_data = self.revere_provider.get_revere_data()
        rbics_master = self.rbics_provider.get_master_table()
        
        print(f"企業データ: {len(entity_data)} 件")
        print(f"財務データ: {len(financial_data)} 件")
        print(f"セグメントデータ: {len(segment_data)} 件")
        print(f"REVEREデータ: {len(revere_data)} 件")
        
        # プライマリーエクイティのフィルタリング（entity_data取得直後）
        print("プライマリーエクイティのフィルタリングを実行中...")
        entity_data = entity_data.groupby('FACTSET_ENTITY_ID', group_keys=False).apply(
            DataProcessor.filter_func
        )
        print(f"フィルタリング後の企業データ: {len(entity_data)} 件")
        
        # 企業データのマージ
        financial_data = financial_data.merge(
            entity_data[["FSYM_ID", "FACTSET_ENTITY_ID", "FF_CO_NAME", "PRIMARY_EQUITY_FLAG"]],
            on="FSYM_ID"
        )
        
        segment_data = segment_data.merge(
            entity_data[["FSYM_ID", "FACTSET_ENTITY_ID", "FF_CO_NAME", "PRIMARY_EQUITY_FLAG"]],
            on="FSYM_ID"
        )
        
        # 重複する会計期間データの除去
        print("重複する会計期間データをチェック中...")
        financial_data = DataProcessor.remove_duplicate_periods(
            financial_data, 
            group_cols=["FSYM_ID", "FACTSET_ENTITY_ID"]
        )
        
        segment_data = DataProcessor.remove_duplicate_periods(
            segment_data,
            group_cols=["FSYM_ID", "FACTSET_ENTITY_ID", "LABEL"]
        )
        
        # seg_af2とseg_af3の作成（オリジナルコードと同等）
        segment_data = self._create_segment_variations(segment_data)
        
        return {
            "entity": entity_data,
            "financial": financial_data,
            "segment": segment_data,
            "revere": revere_data,
            "rbics_master": rbics_master
        }
    
    def get_entity_info(self) -> pd.DataFrame:
        """
        Entity情報のみを取得する軽量メソッド
        
        Returns:
            企業エンティティデータ（プライマリーエクイティフィルタリング済み）
        """
        
        # エンティティデータの取得
        entity_data = self._get_entity_data()        
        # プライマリーエクイティのフィルタリング
        entity_data = entity_data.groupby('FACTSET_ENTITY_ID', group_keys=False).apply(
            DataProcessor.filter_func
        )
        
        return entity_data
    
    def _create_segment_variations(self, segment_data: pd.DataFrame) -> pd.DataFrame:
        """seg_af2とseg_af3を作成（オリジナルコードのseg_af2, seg_af3と同等）"""
        columns_base = [
            "FSYM_ID", "FACTSET_ENTITY_ID", "LABEL", "LABEL_2", "FTERM_2", "FISCAL_YEAR",
            "SALES", "SALES_ER", "SALES_IR", "OPINC", "ASSETS",
            "OPINC_RATIO", "ASSETS_RATIO", "RECON_SALES", "PRIMARY_EQUITY_FLAG"
        ]
        
        # seg_af2相当（SALES_RATIOを使用）
        seg_af2 = segment_data[columns_base + ["SALES_RATIO"]].copy()
        
        # seg_af3相当（SALES_ER_RATIOをSALS_RATIOとしてリネーム）
        seg_af3 = segment_data[columns_base + ["SALES_ER_RATIO"]].copy()
        seg_af3 = seg_af3.rename(columns={"SALES_ER_RATIO": "SALES_RATIO"})
        
        # 連結して重複除去
        result = pd.concat([seg_af2, seg_af3], axis=0).reset_index(drop=True)
        result.drop_duplicates(inplace=True)
        
        return result
    
    def merge_segment_mapping(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """セグメントマッピングの統合"""
        revere_data = data["revere"]
        segment_data = data["segment"]
        entity_data = data["entity"]
        
        # セグメント名の正規化
        revere_processed = self._process_revere_segments(revere_data)
        
        # マッピングの実行
        merged = self._perform_segment_mapping(revere_processed, segment_data)
        
        # 企業情報の追加
        merged = merged.merge(
            entity_data[['FACTSET_ENTITY_ID', 'ISO_COUNTRY_FACT', 'FF_CO_NAME']].drop_duplicates(),
            on=['FACTSET_ENTITY_ID']
        )
        
        # ** 重要：セグメントデータの財務指標をマージ **
        # オリジナルのseg_af_mergedと同等の処理
        # pivot処理でprocess_entity関数と同等の処理を実行
        def process_entity(group):
            group_filled = group.fillna("-9999")
            pivot = group_filled.pivot_table(
                index=["FACTSET_ENTITY_ID", "FISCAL_YEAR"], 
                columns="SEGMENT_NAME_3",
                values=["LABEL", "REVENUE_L6_ID"], 
                aggfunc="first"
            )
            pivot_filled = pivot.copy()
            for col in pivot.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=FutureWarning)
                    orig_nan = pivot[col].isna()
                    tmp = pivot[col].replace("-9999", np.nan)
                    filled = tmp.bfill().ffill()
                    filled[orig_nan] = np.nan
                    pivot_filled[col] = filled
            stacked = pivot_filled.stack(level=1, future_stack=True).reset_index()
            stacked["FACTSET_ENTITY_ID"] = group["FACTSET_ENTITY_ID"].iloc[0]
            return stacked
        
        result = merged.groupby("FACTSET_ENTITY_ID", group_keys=False).apply(process_entity)
        
        # セグメントデータとマージして財務指標を追加
        seg_af_merged = segment_data.merge(
            result.query("LABEL.notna()"),
            left_on=["FACTSET_ENTITY_ID", "FISCAL_YEAR", "LABEL"],
            right_on=["FACTSET_ENTITY_ID", "FISCAL_YEAR", "LABEL"],
            how="inner"
        )
        
        return seg_af_merged
    
    def _process_revere_segments(self, revere_data: pd.DataFrame) -> pd.DataFrame:
        """REVEREセグメントの正規化処理"""
        df = revere_data.copy()
        
        # セグメント名の正規化処理
        df['SEGMENT_NAME_2'] = df['SEGMENT_NAME']
        df['SEGMENT_NAME_3'] = df['SEGMENT_NAME_2'].str.split(' - ').str[0].str.strip()
        
        # 不要なキーワードの除去
        target_blocks = ['Business', 'Sector']
        pattern = r'\s*(?:' + '|'.join(target_blocks) + r')$'
        df['SEGMENT_NAME_3'] = df['SEGMENT_NAME_3'].str.replace(pattern, '', regex=True).str.strip()
        df['SEGMENT_NAME_3'] = df['SEGMENT_NAME_3'].apply(DataProcessor.normalize_text)
        
        return df
    
    def _perform_segment_mapping(self, revere_data: pd.DataFrame, segment_data: pd.DataFrame) -> pd.DataFrame:
        """セグメントマッピングの実行"""
        # merge_asofを使用したマッピング
        merged = pd.merge_asof(
            revere_data[['FACTSET_ENTITY_ID', 'SEGMENT_NAME', 'FISCAL_YEAR', 'REVENUE_L6_ID', 'SALES_RATIO', 'SEGMENT_NAME_3']]
                .drop_duplicates()
                .sort_values(by=['SALES_RATIO']),
            segment_data.dropna(subset=['SALES_RATIO'])
                .drop_duplicates(subset=['FACTSET_ENTITY_ID', 'FTERM_2', 'LABEL', 'SALES_RATIO'])
                [['FACTSET_ENTITY_ID', 'LABEL', 'FISCAL_YEAR', 'SALES_RATIO']]
                .sort_values(by=['SALES_RATIO']),
            by=['FACTSET_ENTITY_ID', 'FISCAL_YEAR'],
            on='SALES_RATIO',
            direction='nearest',
            tolerance=0.001
        ).drop_duplicates()
        
        # 文字列一致による追加マッピング
        merged2 = merged.merge(
            segment_data[['FACTSET_ENTITY_ID', 'FISCAL_YEAR', 'LABEL', 'LABEL_2']],
            left_on=['FACTSET_ENTITY_ID', 'FISCAL_YEAR', 'SEGMENT_NAME_3'],
            right_on=['FACTSET_ENTITY_ID', 'FISCAL_YEAR', 'LABEL_2'],
            how='left',
            suffixes=('', '_new')
        )
        
        merged2['LABEL'] = merged2['LABEL'].fillna(merged2['LABEL_new'])
        merged2.drop(columns=['LABEL_new', 'LABEL_2'], inplace=True)
        merged2.drop_duplicates(inplace=True)
        
        return merged2
    
    def create_pivot_tables(self, financial_data: pd.DataFrame, 
                          segment_scores: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        ROIC計算用のピボットテーブルを作成
        
        Args:
            financial_data: 財務データ
            segment_scores: セグメントスコアデータ
            
        Returns:
            作成されたピボットテーブルの辞書
        """
        print("ピボットテーブルを作成中...")
        
        # 連結財務データのピボットテーブル
        financial_pivot = pd.pivot_table(
            financial_data,
            index=["FTERM_2"],
            columns=["FACTSET_ENTITY_ID"], 
            values=["営業利益(税引後)", "投下資本(運用ベース)"]
        ).sort_index(axis=1)
        
        # セグメントスコアのピボットテーブル
        segment_pivot = pd.pivot_table(
            segment_scores,
            index=["FTERM_2"],
            columns=["CODE_SEGMENT_ADJ"],
            values=["SEG_営業利益(税引後)", "SEG_投下資本(運用ベース)"]
        ).sort_index(axis=1)
        
        pivot_tables = {
            'financial': financial_pivot,
            'segment': segment_pivot
        }
        
        print(f"ピボットテーブル作成完了:")
        print(f"- 連結データ: {financial_pivot.shape}")
        print(f"- セグメントデータ: {segment_pivot.shape}")
        
        return pivot_tables
    
    def _get_entity_data(self) -> pd.DataFrame:
        """FactSetProviderから企業データを取得する統合メソッド"""
        # FactSetProviderから企業識別子データを取得し、従来形式に変換
        identity_records = self.financial_provider.get_identity_records(primary_equity_only=True)
        
        # レコードをDataFrameに変換
        entity_data = []
        for record in identity_records:
            entity_data.append({
                'FACTSET_ENTITY_ID': record.factset_entity_id,
                'FSYM_ID': record.fsym_id,
                'FF_CO_NAME': record.company_name,
                'ISO_COUNTRY_FACT': record.headquarters_country_code,
                'PRIMARY_EQUITY_FLAG': 1 if record.is_primary_equity else 0,
                'FREF_SECURITY_TYPE': getattr(record, 'security_type', None),
                'ACTIVE_FLAG': 1 if record.active_flag else 0
            })
        
        return pd.DataFrame(entity_data)
    
    def _get_financial_data(self) -> pd.DataFrame:
        """FactSetProviderから財務データを取得する統合メソッド"""
        # FactSetProviderから財務データを取得し、従来形式に変換
        financial_records = self.financial_provider.get_financial_records(primary_equity_only=True)
        
        # レコードをDataFrameに変換
        financial_data = []
        for record in financial_records:
            financial_data.append({
                'FSYM_ID': record.fsym_id,
                'DATE': record.period.end_date if hasattr(record.period, 'end_date') else record.period.to_date(),
                'CURRENCY': record.currency,
                # 売上高
                'FF_SALES': record.ff_sales,
                # 営業利益
                'FF_OPER_INC': record.ff_oper_inc,
                # 純利益
                'FF_NET_INC': record.ff_net_inc,
                # EBIT
                'FF_EBIT_OPER': record.ff_ebit_oper,
                # 資産
                'FF_ASSETS': record.ff_assets,
                'FF_ASSETS_CURR': record.ff_assets_curr,
                # 負債
                'FF_DEBT_ST': record.ff_debt_st,
                'FF_DEBT_LT': record.ff_debt_lt,
                #'FF_TOT_DEBT': record.ff_debt,
                'FF_PAY_ACCT': record.ff_pay_acct,  # 買掛金
                'FF_LIABS_CURR_MISC': record.ff_liabs_curr_misc,  # その他流動負債
                # 株主資本
                #'FF_SH_EQ': record.ff_sh_eq,
                # 税務・利息項目
                'FF_TAX_RATE': record.ff_tax_rate,
                'FF_INC_TAX': record.ff_inc_tax,  # 法人税
                'FF_EQ_AFF_INC': record.ff_eq_aff_inc,  # 持分法投資損益
                'FF_INT_EXP_DEBT': record.ff_int_exp_debt,  # 有利子負債利息
                # ROIC関連
                'FF_ROIC': getattr(record, 'ff_roic', None),
                'FF_EFF_INT_RATE': getattr(record, 'ff_eff_int_rate', None),
                # キャッシュフロー
                #'FF_CF_OPER': record.ff_cf_oper,
                #'FF_CAPEX': record.ff_capex,
                #'FF_CF_INV': record.ff_cf_inv,
                #'FF_CF_FIN': record.ff_cf_fin,
                # 時価総額関連
                'FF_MKT_VAL': record.ff_mkt_val,
                #'FF_ENTERPRISE_VAL': record.ff_enterprise_val
            })
        
        df = pd.DataFrame(financial_data)
        
        # 会計期間の調整
        df = DataProcessor.adjust_fiscal_term(df)
        
        # 必要な計算済み列を追加（従来のコードと互換性を保つため）
        self._add_calculated_financial_columns(df)
        
        # 地域マッピング
        currencies = df["CURRENCY"].unique()
        region_df = self.geo_processor.get_region_mapping(currencies)
        df = df.merge(region_df, on=["CURRENCY"], how="left")
        
        return df
    
    def _add_calculated_financial_columns(self, df: pd.DataFrame) -> None:
        """計算済み財務列を追加（元のコードと同じ計算ロジック）"""
        # パーセント値を小数点形式に変換
        if 'FF_ROIC' in df.columns:
            df["FF_ROIC"] /= 100
        df["FF_TAX_RATE"] /= 100
        if 'FF_EFF_INT_RATE' in df.columns:
            df["FF_EFF_INT_RATE"] /= 100

        # 基本計算（元のコードと同じロジック）
        df["固定資産合計"] = df["FF_ASSETS"] - df["FF_ASSETS_CURR"]
        df["投下資本(運用ベース)"] = (
            df["固定資産合計"] +
            df["FF_ASSETS_CURR"] -
            df["FF_PAY_ACCT"].fillna(0) -
            df["FF_LIABS_CURR_MISC"].fillna(0)
        )
        df["営業利益(税引後)"] = (
            df["FF_OPER_INC"] -
            df["FF_INC_TAX"].fillna(0) +
            df["FF_EQ_AFF_INC"].fillna(0)
        )

        # 有利子負債利息の四半期換算(ffill処理)
        if 'FF_INT_EXP_DEBT' in df.columns:
            df = self._fill_debt_interest_quarterly(df)

        # 平均有利子負債
        df['平均有利子負債'] = df['FF_TOT_DEBT']
        
        # 時価総額
        df['時価総額'] = df['FF_MKT_VAL']
        
        # 企業価値
        df['企業価値'] = df['FF_ENTERPRISE_VAL']
        
    def _fill_debt_interest_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """有利子負債利息の四半期換算(ffill処理)"""
        # グループ化して前方補完
        df['FF_INT_EXP_DEBT'] = (
            df.groupby('FSYM_ID')['FF_INT_EXP_DEBT']
            .ffill()
        )
        return df


