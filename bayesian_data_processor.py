"""
ベイジアンモデリング用データ前処理モジュール

FactSetデータからStanモデリング用のデータ構造を作成する
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path
import os
import time

from gppm.pipeline.data_manager import FactSetDataManager
from gppm.finance.roic_calculator import ROICCalculator
from gppm.finance.wacc_calculator import WACCCalculator, WACCColumnConfig
from gppm.utils.country_risk_parameters import CountryRiskParams
from gppm.finance.product_score_calculator import ProductScoreCalculator
from gppm.finance.geographic_processor import GeographicProcessor
from gppm.utils.country_code_manager import get_country_name
from gppm.utils.country_risk_parameters import get_country_risk_manager
from gppm.utils.config_manager import get_logger

# 既存の資産を使用してロガーを取得
logger = get_logger(__name__)


class BayesianDataProcessor:
    """
    ベイジアンモデリング用データ前処理クラス
    
    FactSetのセグメントデータ、連結データ、製品シェアデータを
    Stanモデリング用の形式に変換する
    """
    
    def __init__(self, mapping_df_path: Optional[str] = None):
        """
        初期化
        
        Args:
            mapping_df_path: マッピングデータファイルのパス（オプション）
        """
        self.le = LabelEncoder()
        self.month_map = {
            1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6,
            7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12,
        }
        self.mapping_df_path = mapping_df_path or \
            "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202501/mapping_df.pkl"
        
    def load_from_data_manager(self, data_manager: FactSetDataManager) -> Dict[str, pd.DataFrame]:
        """
        FactSetDataManagerからデータを取得してBayesianDataProcessor用に変換
        
        Args:
            data_manager: 初期化済みのFactSetDataManager
                
        Returns:
            Bayesian処理用のデータ辞書
        """
        logger.info("FactSetDataManagerからデータを取得中...")
        
        # 基本データの取得
        initial_data = data_manager.initialize_data()
        segment_mapping = data_manager.merge_segment_mapping(initial_data)
        
        # 製品スコア計算（セグメント財務指標を作成）
        geo_processor = GeographicProcessor()
        product_calc = ProductScoreCalculator(geo_processor)
        
        # マッピングデータを読み込み
        mapping_df = self._load_mapping_df(segment_mapping, initial_data["rbics_master"])
        
        segment_scores = product_calc.calculate_segment_scores(
            initial_data["financial"], 
            segment_mapping, 
            initial_data["rbics_master"],
            mapping_df
        )
        
        # ピボットテーブル作成（FactSetDataManagerに移動）
        pivot_tables = data_manager.create_pivot_tables(
            initial_data["financial"], 
            segment_scores
        )
        financial_pivot = pivot_tables['financial']
        segment_pivot = pivot_tables['segment']
        
        # ROIC計算
        roic_calc = ROICCalculator()
        consol_roic = roic_calc.calculate_consol_roic(financial_pivot)
        segment_roic = roic_calc.calculate_segment_roic(segment_pivot)
        
        # WACC計算（連結のみ）
        logger.info("WACC計算を実行中...")
        wacc_config = WACCColumnConfig()
        wacc_calculator = WACCCalculator(config=wacc_config)
        
        # 国別リスクパラメータの設定
        risk_manager = get_country_risk_manager()
        country_params = risk_manager.get_all_country_params()
        wacc_calculator.set_country_params(country_params)
        
        # デフォルトパラメータの設定
        default_params = risk_manager.get_default_params()
        wacc_calculator.set_default_params(default_params)
        
        logger.info(f"国別リスクパラメータを設定: {len(country_params)} か国")
        # 主要国のパラメータ表示
        major_countries = ['US', 'JP', 'CN', 'DE', 'GB', 'FR', 'IT', 'CA', 'AU', 'BR', 'IN', 'KR']
        logger.info("主要国のリスクパラメータ:")
        for code in major_countries:
            if risk_manager.is_available(code):
                params = risk_manager.get_country_params(code)
                country_name = get_country_name(code)
                logger.info(f"  {country_name} ({code}): RF {params.risk_free_rate:.2%}, "
                      f"MRP {params.market_risk_premium:.2%}, TAX {params.country_tax_rate:.2%}")
        
        # 企業データと財務データをマージしてWACC計算用データを準備
        wacc_financial_data = initial_data["financial"].merge(
            initial_data["entity"][["FACTSET_ENTITY_ID", "ISO_COUNTRY_FACT"]],
            on="FACTSET_ENTITY_ID",
            how="left"
        )
        
        # WACC計算結果を取得
        wacc_results = wacc_calculator.calculate_wacc_from_merged_data(
            wacc_financial_data,
            cost_of_equity=0.10,  # 株主資本コスト10%（例）
            tax_rate=0.25         # 法人税率25%（例）
        )
        
        # WACC結果から連結データを抽出
        if 'processed_data' in wacc_results and 'WACC' in wacc_results['processed_data'].columns:
            wacc_calculation_cols = [
                'FACTSET_ENTITY_ID', 'FTERM_2', 'FISCAL_YEAR', 'WACC',
                '時価総額', '平均有利子負債', '企業価値',
                '株主資本比率', '負債比率', '株主資本コスト', '負債コスト（税引後）',
                '負債コスト（税引前）', '実効税率'
            ]
            
            wacc_consol_data = wacc_results['processed_data'][wacc_calculation_cols].copy()
            
            logger.info(f"WACC計算完了: {len(wacc_consol_data)} 件")
        else:
            logger.error("WACC計算でエラーが発生しました")
            wacc_consol_data = pd.DataFrame()
        
        # セグメントデータの構築
        segment_data = self._prepare_segment_data(segment_scores, segment_roic)
        
        # 連結データの構築（ROICとWACCを統合）
        consol_data = self._prepare_consol_data(initial_data["financial"], consol_roic, wacc_consol_data)
        
        # ピボットテーブルの作成
        pivot_data = product_calc.create_pivot_tables(
            segment_scores, 
            initial_data["rbics_master"]
        )
        
        # 製品シェアデータの構築
        segment_product_share = self._prepare_product_share(pivot_data['segment_l5'])
        consol_product_share = self._prepare_product_share(pivot_data['consol_l5'])
        
        # 製品名の抽出
        product_names = list(pivot_data['segment_l5'].columns)
        
        result_data = {
            'segment': segment_data,
            'consol': consol_data,
            'segment_product_share': segment_product_share,
            'consol_product_share': consol_product_share,
            'product_names': product_names
        }
        
        logger.info(f"データ取得完了:")
        logger.info(f"- セグメントデータ: {len(segment_data)} 件")
        logger.info(f"- 連結データ: {len(consol_data)} 件")
        logger.info(f"- 製品数: {len(product_names)} 種類")
        
        return result_data
    
    def _load_mapping_df(self, segment_mapping: pd.DataFrame, 
                        rbics_master: pd.DataFrame) -> pd.DataFrame:
        """
        マッピングデータをファイルから読み込み
        
        Args:
            segment_mapping: セグメントマッピングデータ
            rbics_master: RBICS マスターデータ
            
        Returns:
            マッピングデータ
        """
        # ファイルからマッピングデータを読み込み
        if not os.path.exists(self.mapping_df_path):
            raise FileNotFoundError(f"マッピングファイルが見つかりません: {self.mapping_df_path}")
        
        with open(self.mapping_df_path, "rb") as f:
            mapping_df = pickle.load(f)
            # 必要な列のみを抽出
            mapping_df = mapping_df[["FACTSET_ENTITY_ID", "PRODUCT_L6_ID", "RELABEL_L6_ID"]]
            logger.info(f"マッピングデータを読み込みました: {len(mapping_df)} 件")
        
        return mapping_df
    
    def _prepare_segment_data(self, segment_scores: pd.DataFrame, 
                            segment_roic: pd.DataFrame) -> pd.DataFrame:
        """セグメントデータをBayesian用に変換"""
        # セグメントROICをlong形式に変換
        segment_roic_long = segment_roic.stack().to_frame("SEG_ROIC(運用ベース)").reset_index()
        segment_roic_long.rename(columns={'level_0': 'CODE_SEGMENT_ADJ'}, inplace=True)
        
        # セグメントスコアデータから必要な情報を抽出
        segment_calculation_cols = [
            'FACTSET_ENTITY_ID', 'LABEL', 'FTERM_2', 'CODE_SEGMENT_ADJ',
            'SEG_営業利益(税引後)', 'SEG_投下資本(運用ベース)'
        ]
        
        segment_info = segment_scores[segment_calculation_cols].drop_duplicates()
        
        # ROICデータとマージ
        result = segment_info.merge(
            segment_roic_long,
            on=['CODE_SEGMENT_ADJ', 'FTERM_2'],
            how='inner'
        )
        
        return result
    
    def _prepare_consol_data(self, financial_data: pd.DataFrame, 
                           consol_roic: pd.DataFrame,
                           wacc_data: pd.DataFrame = None) -> pd.DataFrame:
        """連結データをBayesian用に変換（ROICとWACCを統合）"""
        # 連結ROICをlong形式に変換
        consol_roic_long = consol_roic.stack().to_frame("ROIC(運用ベース)").reset_index()
        
        # 財務データから基本情報を抽出
        consol_calculation_cols = [
            'FACTSET_ENTITY_ID', 'FTERM_2', 'FISCAL_YEAR',
            '営業利益(税引後)', '投下資本(運用ベース)'
        ]
        
        financial_info = financial_data[consol_calculation_cols].drop_duplicates()
        
        # ROICデータとマージ
        result = financial_info.merge(
            consol_roic_long,
            on=['FACTSET_ENTITY_ID', 'FTERM_2'],
            how='inner'
        )
        
        # WACCデータも追加（ある場合）
        if wacc_data is not None and not wacc_data.empty:
            # WACCデータとマージ
            result = result.merge(
                wacc_data,  # 全ての計算詳細列を含める
                on=['FACTSET_ENTITY_ID', 'FTERM_2'],
                how='left'  # ROICに合わせてleft join
            )
            logger.info(f"WACCデータを連結データに統合: {result['WACC'].notna().sum()} 件")
        else:
            # WACCデータがない場合はNaNで埋める
            result['WACC'] = np.nan
            logger.warning("WACCデータが利用できません")
        
        return result
    
    def _prepare_product_share(self, pivot_data: pd.DataFrame) -> pd.DataFrame:
        """製品シェアデータを準備（セグメント・連結共通）"""
        # MultiIndexを平坦化
        result = pivot_data.reset_index()
        
        # FTERM_2を適切な形式に変換
        if 'FTERM_2' in result.columns:
            # 既にYYYYMM形式の整数の場合はそのまま保持
            if result['FTERM_2'].dtype == 'int64' or result['FTERM_2'].dtype == 'int32':
                pass
            else:
                # 文字列や他の形式の場合はYYYYMM形式に変換
                try:
                    # まず文字列として処理
                    fterm_str = result['FTERM_2'].astype(str)
                    # YYYY-MM-DD形式やYYYY-MM形式の場合
                    result['FTERM_2'] = pd.to_datetime(fterm_str, format='%Y%m', errors='coerce')
                    result['FTERM_2'] = result['FTERM_2'].dt.strftime('%Y%m').astype(int)
                except:
                    # 変換に失敗した場合は直接YYYYMM形式として処理
                    result['FTERM_2'] = result['FTERM_2'].astype(str).str.replace('-', '').str[:6].astype(int)
        
        return result
    
    def filter_data_by_period(self, data: Dict[str, pd.DataFrame], 
                            start_period: int = 201909, 
                            end_period: int = 202406) -> Dict[str, pd.DataFrame]:
        """
        期間でデータをフィルタリング
        
        Args:
            data: データ辞書
            start_period: 開始期間（YYYYMM形式）
            end_period: 終了期間（YYYYMM形式）
            
        Returns:
            フィルタリングされたデータ辞書
        """
        filtered_data = {}
        
        # 製品名はそのまま保持
        filtered_data['product_names'] = data['product_names']
        
        # セグメントデータとコンソルデータ
        for key in ['segment', 'consol']:
            filtered_data[key] = data[key].query(f"FTERM_2 >= {start_period} and FTERM_2 <= {end_period}")
        
        # 製品シェアデータ（MultiIndexの場合）
        for key in ['segment_product_share', 'consol_product_share']:
            df = data[key]
            if isinstance(df.index, pd.MultiIndex):
                filtered_data[key] = df.loc[
                    (df.index.get_level_values('FTERM_2') >= start_period) &
                    (df.index.get_level_values('FTERM_2') <= end_period)
                ]
            else:
                filtered_data[key] = df.query(f"FTERM_2 >= {start_period} and FTERM_2 <= {end_period}")
            
        return filtered_data
    
    def filter_valid_entities(self, data: Dict[str, pd.DataFrame], 
                            na_threshold: float = 1.0) -> Dict[str, pd.DataFrame]:
        """
        有効なエンティティでデータをフィルタリング
        
        Args:
            data: データ辞書
            na_threshold: 欠損値の閾値（1.0未満で欠損率フィルタリング）
            
        Returns:
            フィルタリングされたデータ辞書
        """
        filtered_data = data.copy()
        
        # セグメントデータの有効性チェック
        segment_data = data['segment']
        na_rate = (
            segment_data.groupby("CODE_SEGMENT_ADJ")["SEG_ROIC(運用ベース)"]
            .apply(lambda x: x.isna().mean())
        )
        valid_segment_ids = na_rate[na_rate < na_threshold].index
        filtered_data['segment'] = segment_data.query("CODE_SEGMENT_ADJ.isin(@valid_segment_ids)")
            
        # 連結データの有効性チェック
        consol_data = data['consol']
        na_rate = (
            consol_data.groupby("FACTSET_ENTITY_ID")["ROIC(運用ベース)"]
            .apply(lambda x: x.isna().mean())
        )
        valid_entity_ids = na_rate[na_rate < na_threshold].index
        filtered_data['consol'] = consol_data.query("FACTSET_ENTITY_ID.isin(@valid_entity_ids)")
            
        return filtered_data
    
    def align_data_entities(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        データ間でエンティティIDを整合させる
        
        Args:
            data: データ辞書
            
        Returns:
            整合されたデータ辞書
        """
        aligned_data = data.copy()
        
        # 連結データの整合
        consol_active_ids = (
            set(data['consol']["FACTSET_ENTITY_ID"].unique()) &
            set(data['consol_product_share'].index.get_level_values(0).unique() 
                if isinstance(data['consol_product_share'].index, pd.MultiIndex) 
                else data['consol_product_share']["FACTSET_ENTITY_ID"].unique())
        )
        aligned_data['consol'] = data['consol'].query("FACTSET_ENTITY_ID.isin(@consol_active_ids)")
        
        aligned_data['consol_product_share'] = data['consol_product_share'].query("FACTSET_ENTITY_ID.isin(@consol_active_ids)")                
        # セグメントデータの整合
        segment_active_ids = (
            set(data['segment']["CODE_SEGMENT_ADJ"].unique()) &
            set(data['segment_product_share']["CODE_SEGMENT_ADJ"].unique())
        )
        aligned_data['segment'] = data['segment'].query("CODE_SEGMENT_ADJ.isin(@segment_active_ids)")
        aligned_data['segment_product_share'] = data['segment_product_share'].query("CODE_SEGMENT_ADJ.isin(@segment_active_ids)")
            
        return aligned_data
    
    def clean_product_names(self, data: Dict[str, pd.DataFrame]) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
        """
        製品名をクリーニング（シェアが0の製品を除去）
        
        Args:
            data: データ辞書
            
        Returns:
            クリーニングされたデータ辞書と有効な製品名リスト
        """
        prod_names = data['product_names'].copy()
        cleaned_data = data.copy()
        
        # 0シェアの製品を特定
        zero_products = set()
        
        zeros1 = data['segment_product_share'][prod_names].columns[
            (data['segment_product_share'][prod_names].sum(axis=0) == 0)
        ]
        zero_products.update(zeros1)
        
        zeros2 = data['consol_product_share'][prod_names].columns[
            (data['consol_product_share'][prod_names].sum(axis=0) == 0)
        ]
        zero_products.update(zeros2)
        
        # 有効な製品名
        valid_prod_names = [name for name in prod_names if name not in zero_products]
        
        # データから0シェア製品を除去
        cleaned_data['segment_product_share'] = cleaned_data['segment_product_share'].drop(columns=list(zero_products), errors='ignore')
        cleaned_data['consol_product_share'] = cleaned_data['consol_product_share'].drop(columns=list(zero_products), errors='ignore')
        
        return cleaned_data, valid_prod_names
    
    def convert_date_format(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        日付フォーマットを変換
        
        Args:
            data: データ辞書
            
        Returns:
            日付変換されたデータ辞書
        """
        converted_data = {}
        
        # 製品名はそのまま保持
        converted_data['product_names'] = data['product_names']
        
        # セグメントデータとコンソルデータ（FTERM_2列を持つ）
        for key in ['segment', 'consol']:
            df = data[key].copy()
            df["FTERM_2"] = pd.to_datetime(df["FTERM_2"], format="%Y%m")
            converted_data[key] = df
        
        # 製品シェアデータ（MultiIndexの場合）
        for key in ['segment_product_share', 'consol_product_share']:
            df = data[key].copy()
            if isinstance(df.index, pd.MultiIndex):
                level_idx = df.index.names.index('FTERM_2')
                new_levels = list(df.index.levels)
                new_levels[level_idx] = pd.to_datetime(
                    df.index.levels[level_idx].astype(str), format="%Y%m"
                )
                df.index = df.index.set_levels(new_levels, level='FTERM_2')
            else:
                df["FTERM_2"] = pd.to_datetime(df["FTERM_2"], format="%Y%m")
            converted_data[key] = df
            
        return converted_data
    
    def create_segment_mapping(self, segment_data: pd.DataFrame) -> pd.DataFrame:
        """
        セグメント情報からマッピングデータを作成
        
        Args:
            segment_data: セグメントデータ
            
        Returns:
            マッピングされたセグメントデータ
        """
        segment_mapped = segment_data.reset_index(drop=True)
        segment_mapped[["FACTSET_ENTITY_ID", "SEGMENT_ADJ"]] = segment_mapped["CODE_SEGMENT_ADJ"].str.extract(r"(.+?)_(.+)")
        return segment_mapped
    
    def merge_product_share_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        製品シェアデータとメインデータをマージ
        
        Args:
            data: データ辞書
            
        Returns:
            マージされたデータ辞書
        """
        merged_data = data.copy()
        
        # セグメントデータのマージ
        segment_mapped = self.create_segment_mapping(data['segment'])
        merged_data['segment_product_share'] = data['segment_product_share'].merge(
            segment_mapped, on=["CODE_SEGMENT_ADJ", "FTERM_2"], how="right"
        )
        merged_data['segment'] = segment_mapped
        
        # 連結データのマージ
        merged_data['consol_product_share'] = data['consol_product_share'].merge(
            data['consol'], on=["FACTSET_ENTITY_ID", "FTERM_2"], how="right"
        )
            
        return merged_data
    
    def create_pivot_tables(self, data: Dict[str, pd.DataFrame], 
                          prod_names: List[str]) -> Dict[str, pd.DataFrame]:
        """
        ピボットテーブルを作成
        
        Args:
            data: データ辞書
            prod_names: 製品名リスト
            
        Returns:
            ピボットテーブル辞書
        """
        pivot_tables = {}
        
        # セグメントROICピボット
        pivot_tables['Y_segment'] = pd.pivot_table(
            data=data['segment'],
            columns=["FTERM_2"],
            index=["CODE_SEGMENT_ADJ"],
            values="SEG_ROIC(運用ベース)",
        )
        pivot_tables['Y_segment'].sort_index(inplace=True)
            
        # 連結ROICピボット
        pivot_tables['Y_consol'] = pd.pivot_table(
            data=data['consol'],
            columns=["FTERM_2"],
            index="FACTSET_ENTITY_ID",
            values="ROIC(運用ベース)",
        )
        pivot_tables['Y_consol'].sort_index(inplace=True)
        
        # 連結WACCピボット
        if 'WACC' in data['consol'].columns:
            pivot_tables['Y_consol_wacc'] = pd.pivot_table(
                data=data['consol'],
                columns=["FTERM_2"],
                index="FACTSET_ENTITY_ID",
                values="WACC",
            )
            pivot_tables['Y_consol_wacc'].sort_index(inplace=True)
            logger.info(f"WACC ピボットテーブル作成: {pivot_tables['Y_consol_wacc'].shape}")
        else:
            logger.warning("WACC データが見つかりません - ピボットテーブルをスキップ")
        
        # セグメント製品シェアピボット
        pivot_tables['X2_segment'] = (
            data['segment_product_share'][
                ["CODE_SEGMENT_ADJ", "FTERM_2", *prod_names]
            ]
            .set_index(["CODE_SEGMENT_ADJ", "FTERM_2"])
        )
        
        # 時系列補完
        pivot_tables['X2_segment'].loc[:, prod_names] = (
            pivot_tables['X2_segment']
            .groupby(by=["CODE_SEGMENT_ADJ"], group_keys=False)
            .apply(lambda _df: _df[prod_names].ffill().bfill())
            .to_numpy()
        )
        pivot_tables['X2_segment'].sort_index(inplace=True)
            
        # 連結製品シェアピボット
        pivot_tables['X2_consol'] = (
            data['consol_product_share'][
                ["FACTSET_ENTITY_ID", "FTERM_2", *prod_names]
            ]
            .set_index(["FACTSET_ENTITY_ID", "FTERM_2"])
        )
        
        # 時系列補完
        pivot_tables['X2_consol'].loc[:, prod_names] = (
            pivot_tables['X2_consol']
            .groupby(by=["FACTSET_ENTITY_ID"], group_keys=False)
            .apply(lambda _df: _df[prod_names].ffill().bfill())
            .to_numpy()
        )
        pivot_tables['X2_consol'].sort_index(inplace=True)
        
        # 元のnotebookと完全に同様の処理
        
        # セグメント側の処理
        # X2_segment に存在する月だけを Y_segment の列として残す
        first_segment_date = pivot_tables['X2_segment'].index.get_level_values(1).unique()[0]
        segment_entities_in_first_date = pivot_tables['X2_segment'].xs(first_segment_date, level=1).index
        pivot_tables['Y_segment'] = pivot_tables['Y_segment'].loc[segment_entities_in_first_date]
        
        # 連結側の処理
        # X2_consol に存在する月だけを Y_consol の列として残す  
        first_consol_date = pivot_tables['X2_consol'].index.get_level_values(1).unique()[0]
        consol_entities_in_first_date = pivot_tables['X2_consol'].xs(first_consol_date, level=1).index
        pivot_tables['Y_consol'] = pivot_tables['Y_consol'].loc[consol_entities_in_first_date]
            
        return pivot_tables
    
    def align_pivot_tables(self, pivot_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        ピボットテーブル間でインデックスを整合させる（np.stackエラー対策）
        
        Args:
            pivot_tables: ピボットテーブル辞書
            
        Returns:
            整合されたピボットテーブル辞書
        """
        aligned_tables = pivot_tables.copy()
        
        # セグメントデータの追加整合
        logger.info("セグメントデータの形状チェック中...")
        segment_dates = aligned_tables['X2_segment'].index.get_level_values(1).unique()
        logger.info(f"セグメント時点数: {len(segment_dates)}")
        
        for i, date in enumerate(segment_dates):
            entities = aligned_tables['X2_segment'].xs(date, level=1).index
            logger.info(f"時点 {i} ({date}): {len(entities)} セグメント")
            
        # 全ての時点で共通するセグメントのみを残す
        common_segments = None
        for date in segment_dates:
            date_segments = set(aligned_tables['X2_segment'].xs(date, level=1).index)
            if common_segments is None:
                common_segments = date_segments
            else:
                common_segments = common_segments.intersection(date_segments)
        
        logger.info(f"共通セグメント数: {len(common_segments)}")
        
        # 共通セグメントのみを残す
        common_segments = sorted(list(common_segments))
        aligned_tables['X2_segment'] = aligned_tables['X2_segment'].loc[
            aligned_tables['X2_segment'].index.get_level_values(0).isin(common_segments)
        ]
        aligned_tables['Y_segment'] = aligned_tables['Y_segment'].loc[common_segments]
        
        # 連結データの追加整合
        logger.info("連結データの形状チェック中...")
        consol_dates = aligned_tables['X2_consol'].index.get_level_values(1).unique()
        logger.info(f"連結時点数: {len(consol_dates)}")
        
        for i, date in enumerate(consol_dates):
            entities = aligned_tables['X2_consol'].xs(date, level=1).index
            logger.info(f"時点 {i} ({date}): {len(entities)} エンティティ")
        
        # 全ての時点で共通するエンティティのみを残す
        common_entities = None
        for date in consol_dates:
            date_entities = set(aligned_tables['X2_consol'].xs(date, level=1).index)
            if common_entities is None:
                common_entities = date_entities
            else:
                common_entities = common_entities.intersection(date_entities)
        
        logger.info(f"共通エンティティ数: {len(common_entities)}")
        
        # 共通エンティティのみを残す
        common_entities = sorted(list(common_entities))
        aligned_tables['X2_consol'] = aligned_tables['X2_consol'].loc[
            aligned_tables['X2_consol'].index.get_level_values(0).isin(common_entities)
        ]
        aligned_tables['Y_consol'] = aligned_tables['Y_consol'].loc[common_entities]
        
        # WACCピボットテーブルも整合（存在する場合）
        if 'Y_consol_wacc' in aligned_tables:
            # WACCテーブルの共通エンティティと時点の整合
            wacc_entities = set(aligned_tables['Y_consol_wacc'].index)
            final_common_entities = set(common_entities).intersection(wacc_entities)
            final_common_entities = sorted(list(final_common_entities))
            
            # 全テーブルを最終的な共通エンティティに限定
            aligned_tables['Y_consol'] = aligned_tables['Y_consol'].loc[final_common_entities]
            aligned_tables['Y_consol_wacc'] = aligned_tables['Y_consol_wacc'].loc[final_common_entities]
            aligned_tables['X2_consol'] = aligned_tables['X2_consol'].loc[
                aligned_tables['X2_consol'].index.get_level_values(0).isin(final_common_entities)
            ]
            
            logger.info(f"WACC整合後の共通エンティティ数: {len(final_common_entities)}")
        
        logger.info("ピボットテーブル整合完了")
        return aligned_tables
    
    def create_stan_data_structure(self, pivot_tables: Dict[str, pd.DataFrame], 
                                 prod_names: List[str]) -> Dict:
        """
        Stan用のデータ構造を作成
        
        Args:
            pivot_tables: ピボットテーブル辞書
            prod_names: 製品名リスト
            
        Returns:
            Stan用データ辞書
        """
        stan_data = {}
        
        # 基本次元数
        stan_data["Segment_N"], stan_data["Time_N"] = pivot_tables['Y_segment'].shape
        stan_data["Product_N"] = len(prod_names)
        stan_data["Company_N"] = len(pivot_tables['Y_consol'].index)
        stan_data["Company_N_c"] = pivot_tables['Y_consol'].shape[0]
        
        # シェアマトリックス（時間×エンティティ×製品）
        shared_segment = np.stack(
            pivot_tables['X2_segment'].groupby(level=['FTERM_2'])
            .apply(lambda _df: _df.to_numpy()).to_list()
        )
        stan_data["Share"] = shared_segment
        stan_data["N_Share_index"] = shared_segment.shape[1]
        
        shared_consol = np.stack(
            pivot_tables['X2_consol'].groupby(level=['FTERM_2'])
            .apply(lambda _df: _df.to_numpy()).to_list()
        )
        stan_data["Share_consol"] = shared_consol
        stan_data["N_Share_consol_index"] = shared_consol.shape[1]
        
        # 観測値（欠損値除去）
        y_segment_flat = pivot_tables['Y_segment'].to_numpy().ravel(order='F')
        nonna_segment = np.argwhere(~np.isnan(y_segment_flat)).ravel(order='F') + 1
        
        stan_data["Seg_ROIC"] = y_segment_flat[~np.isnan(y_segment_flat)]
        stan_data["non_na_index"] = nonna_segment
        stan_data["N_obs"] = len(stan_data["Seg_ROIC"]) 
        
        y_consol_flat = pivot_tables['Y_consol'].to_numpy().ravel(order='F')
        nonna_consol = np.argwhere(~np.isnan(y_consol_flat)).ravel(order='F') + 1
        
        stan_data["Consol_ROIC"] = y_consol_flat[~np.isnan(y_consol_flat)]
        stan_data["non_na_index_consol"] = nonna_consol
        stan_data["N_obs_consol"] = len(stan_data["Consol_ROIC"]) 
        
        # WACCデータの処理（存在する場合）
        if 'Y_consol_wacc' in pivot_tables:
            y_wacc_flat = pivot_tables['Y_consol_wacc'].to_numpy().ravel(order='F')
            nonna_wacc = np.argwhere(~np.isnan(y_wacc_flat)).ravel(order='F') + 1
            
            stan_data["Consol_WACC"] = y_wacc_flat[~np.isnan(y_wacc_flat)]
            stan_data["non_na_index_wacc"] = nonna_wacc
            stan_data["N_obs_wacc"] = len(stan_data["Consol_WACC"]) 
            
            # WACCエンティティインデックス（ROICと同じ構造）
            wacc_codes = pivot_tables['Y_consol_wacc'].index.values
            self.le.fit(wacc_codes)
            codes_cat_wacc = np.repeat(
                self.le.transform(wacc_codes).astype("int32").reshape(-1, 1),
                pivot_tables['Y_consol_wacc'].shape[1],
                axis=1,
            )
            stan_data["wacc_index"] = codes_cat_wacc.ravel(order="F")[~np.isnan(y_wacc_flat)] + 1
            logger.info(f"WACC Stan データ作成: {stan_data['N_obs_wacc']} 観測値")
        else:
            # WACCデータがない場合
            stan_data["Consol_WACC"] = np.array([])
            stan_data["non_na_index_wacc"] = np.array([])
            stan_data["N_obs_wacc"] = 0
            stan_data["wacc_index"] = np.array([])
            logger.warning("WACC データが利用できません")
        
        # エンティティインデックス
        segment_codes = pivot_tables['Y_segment'].index.values
        self.le.fit(segment_codes)
        codes_cat = np.repeat(
            self.le.transform(segment_codes).astype("int32").reshape(-1, 1),
            pivot_tables['Y_segment'].shape[1],
            axis=1,
        )
        stan_data["segment_index"] = codes_cat.ravel(order="F")[~np.isnan(y_segment_flat)] + 1
        stan_data["segment_index_vec"] = codes_cat[:, 0] + 1
        stan_data["N_segment_index_vec"] = len(stan_data["segment_index_vec"])
        
        consol_codes = pivot_tables['Y_consol'].index.values
        self.le.fit(consol_codes)
        codes_cat_consol = np.repeat(
            self.le.transform(consol_codes).astype("int32").reshape(-1, 1),
            pivot_tables['Y_consol'].shape[1],
            axis=1,
        )
        stan_data["consol_index"] = codes_cat_consol.ravel(order="F")[~np.isnan(y_consol_flat)] + 1
        stan_data["consol_index_vec"] = codes_cat_consol[:, 0] + 1
        stan_data["N_consol_index_vec"] = len(stan_data["consol_index_vec"])
        
        # 予測期間
        stan_data["N_pred_term"] = 4  # 1年先まで予測
        
        return stan_data
    
    def save_processed_data(self, processed_data: Dict, save_path: str):
        """
        処理済みデータを保存
        
        Args:
            processed_data: 処理済みデータ辞書
            save_path: 保存先パス
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"データを保存しました: {save_path}")
    
    def process_full_pipeline(self, data_manager: FactSetDataManager,
                            save_path: Optional[str] = None,
                            start_period: int = 201909,
                            end_period: int = 202406) -> Dict:
        """
        ベイジアンモデリング用データ前処理の全パイプラインを実行
        
        Args:
            data_manager: 初期化済みのFactSetDataManager
            save_path: 保存先パス（Noneの場合は保存しない）
            start_period: 開始期間
            end_period: 終了期間
            
        Returns:
            処理済みデータ辞書
        """
        logger.info("=== ベイジアンモデリング用データ前処理開始 ===")
        
        # 1. データ取得
        logger.info("1. FactSetDataManagerからデータ取得中...")
        data = self.load_from_data_manager(data_manager)
        
        # 2. Entity情報の準備（国別可視化用）
        logger.info("2. Entity情報準備中...")
        entity_info = data_manager.get_entity_info()[["FACTSET_ENTITY_ID", "ISO_COUNTRY_FACT"]].copy()
        entity_info['COUNTRY_NAME'] = entity_info['ISO_COUNTRY_FACT'].apply(get_country_name)
        
        # 3. 期間フィルタリング
        logger.info(f"3. 期間フィルタリング中... ({start_period}-{end_period})")
        data = self.filter_data_by_period(data, start_period, end_period)
        
        # 4. 有効エンティティフィルタリング
        logger.info("4. 有効エンティティフィルタリング中...")
        data = self.filter_valid_entities(data)
        
        # 5. エンティティ整合
        logger.info("5. エンティティ整合中...")
        data = self.align_data_entities(data)
        
        # 6. 製品名クリーニング
        logger.info("6. 製品名クリーニング中...")
        data, prod_names = self.clean_product_names(data)
        
        # 7. 日付フォーマット変換
        logger.info("7. 日付フォーマット変換中...")
        data = self.convert_date_format(data)
        
        # 8. 製品シェアデータマージ
        logger.info("8. 製品シェアデータマージ中...")
        data = self.merge_product_share_data(data)
        
        # 9. ピボットテーブル作成
        logger.info("9. ピボットテーブル作成中...")
        pivot_tables = self.create_pivot_tables(data, prod_names)
        
        # 10. ピボットテーブル整合
        logger.info("10. ピボットテーブル整合中...")
        pivot_tables = self.align_pivot_tables(pivot_tables)
                
        # 11. Stan用データ構造作成
        logger.info("11. Stan用データ構造作成中...")
        stan_data = self.create_stan_data_structure(pivot_tables, prod_names)
        
        # 結果の集約
        processed_data = {
            'raw_data': data,
            'pivot_tables': pivot_tables,
            'stan_data': stan_data,
            'product_names': prod_names,
            'entity_info': entity_info,
            'processing_info': {
                'start_period': start_period,
                'end_period': end_period,
                'n_products': len(prod_names),
                'n_segments': stan_data['Segment_N'],
                'n_companies': stan_data['Company_N_c'],
                'n_time_periods': stan_data['Time_N']
            }
        }
        
        # 保存
        if save_path:
            self.save_processed_data(processed_data, save_path)
        
        logger.info("=== データ前処理完了 ===")
        logger.info(f"製品数: {len(prod_names)}")
        logger.info(f"セグメント数: {stan_data['Segment_N']}")
        logger.info(f"企業数: {stan_data['Company_N_c']}")
        logger.info(f"時間期間数: {stan_data['Time_N']}")
        
        return processed_data


