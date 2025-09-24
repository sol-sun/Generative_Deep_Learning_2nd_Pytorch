"""
ベイジアンモデリング用データ前処理モジュール

FactSetデータからStanモデリング用のデータ構造を作成する。
このモジュールは、財務データの前処理、ROIC/WACC計算、
製品シェアデータの統合、Stan用データ構造の生成を行う。

Classes:
    BayesianDataProcessor: ベイジアンモデリング用データ前処理のメインクラス

Constants:
    DEFAULT_MAPPING_PATH (str): デフォルトのマッピングファイルパス
    MAJOR_COUNTRIES (List[str]): 主要国の国コードリスト
    QUARTER_MAPPING (Dict[int, int]): 月から四半期へのマッピング
    PREDICTION_PERIODS (int): 予測期間数（四半期単位）
    WACC_DEFAULT_COST_OF_EQUITY (float): デフォルトの株主資本コスト
    WACC_DEFAULT_TAX_RATE (float): デフォルトの法人税率
"""

import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from gppm.analysis.financial_metrics.product_score_calculator import ProductScoreCalculator
from gppm.analysis.financial_metrics.roic_calculator import ROICCalculator
from gppm.analysis.risk_capital.wacc_calculator import WACCCalculator, WACCColumnConfig
from gppm.core.config_manager import get_logger
from gppm.core.data_manager import FactSetDataManager
from gppm.core.monitoring import monitor_data_processing
from gppm.utils.country_code_manager import get_country_name
from gppm.utils.country_risk_parameters import get_country_risk_manager
from gppm.utils.geographic_processor import GeographicProcessor

# 定数定義
DEFAULT_MAPPING_PATH = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202501/mapping_df.pkl"
MAJOR_COUNTRIES = ['US', 'JP', 'CN', 'DE', 'GB', 'FR', 'IT', 'CA', 'AU', 'BR', 'IN', 'KR']
QUARTER_MAPPING = {
    1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6,
    7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12,
}
#PREDICTION_PERIODS = 4  # 1年先まで予測
WACC_DEFAULT_COST_OF_EQUITY = 0.10  # 株主資本コスト10%
WACC_DEFAULT_TAX_RATE = 0.25  # 法人税率25%

# ロガーの初期化
logger = get_logger(__name__)


class BayesianDataProcessor:
    """
    ベイジアンモデリング用データ前処理クラス
    
    FactSetのセグメントデータ、連結データ、製品シェアデータを
    Stanモデリング用の形式に変換する。データの取得、前処理、
    統合、Stan用データ構造の生成まで一貫して処理する。
    
    Attributes:
        label_encoder (LabelEncoder): エンティティIDのエンコーディング用
        quarter_mapping (Dict[int, int]): 月から四半期へのマッピング
        mapping_df_path (str): マッピングデータファイルのパス
        
    Examples:
        >>> processor = BayesianDataProcessor()
        >>> data_manager = FactSetDataManager()
        >>> result = processor.process_full_pipeline(
        ...     data_manager=data_manager,
        ...     start_period=201909,
        ...     end_period=202406
        ... )
        >>> print(f"処理完了: {result['processing_info']}")
    """
    
    def __init__(self, mapping_df_path: Optional[str] = None) -> None:
        """
        初期化
        
        Args:
            mapping_df_path: マッピングデータファイルのパス（オプション）
                Noneの場合はデフォルトパスを使用
                
        Note:
            マッピングファイルは製品分類の再ラベリングに使用される
        """
        self.label_encoder = LabelEncoder()
        self.quarter_mapping = QUARTER_MAPPING.copy()
        self.mapping_df_path = mapping_df_path or DEFAULT_MAPPING_PATH
        
    def load_from_data_manager(self, data_manager: FactSetDataManager) -> Dict[str, pd.DataFrame]:
        """
        FactSetDataManagerからデータを取得してBayesianDataProcessor用に変換
        
        FactSetDataManagerから基本データを取得し、製品スコア計算、
        ROIC/WACC計算、データ統合を行ってBayesian処理用のデータ構造を作成する。
        
        Args:
            data_manager (FactSetDataManager): 初期化済みのFactSetDataManager
                
        Returns:
            Dict[str, pd.DataFrame]: Bayesian処理用のデータ辞書。以下のキーを含む:
                - segment: セグメントデータ（ROIC、財務指標）
                - consol: 連結データ（ROIC、WACC、財務指標）
                - segment_product_share: セグメント製品シェアデータ
                - consol_product_share: 連結製品シェアデータ
                - product_names: 製品名リスト
                
        Raises:
            FileNotFoundError: マッピングファイルが見つからない場合
            KeyError: 必要なデータが存在しない場合
            
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> data_manager = FactSetDataManager()
            >>> data = processor.load_from_data_manager(data_manager)
            >>> print(f"セグメントデータ: {len(data['segment'])} 件")
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
        logger.info("主要国のリスクパラメータ:")
        for code in MAJOR_COUNTRIES:
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
            cost_of_equity=WACC_DEFAULT_COST_OF_EQUITY,
            tax_rate=WACC_DEFAULT_TAX_RATE
        )
        logger.debug(wacc_results)
        
        # WACC結果から連結データを抽出
        if 'processed_data' in wacc_results and 'WACC' in wacc_results['processed_data'].columns:
            wacc_calculation_cols = [
                'FACTSET_ENTITY_ID', 'FTERM_2', 'FISCAL_YEAR', 'WACC',
                '時価総額', '平均有利子負債', '企業価値',
                '株主資本比率', '負債比率', '株主資本コスト', '負債コスト(税引後)',
                '負債コスト(税引前)', '実効税率'
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
        
        # 製品名の抽出（重複除去とソート）
        product_names = sorted(list(set(pivot_data['segment_l5'].columns)))
        
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
        
        製品分類の再ラベリング用のマッピングデータをpickleファイルから読み込む。
        必要な列（FACTSET_ENTITY_ID, PRODUCT_L6_ID, RELABEL_L6_ID）のみを抽出する。
        
        Args:
            segment_mapping (pd.DataFrame): セグメントマッピングデータ（未使用）
            rbics_master (pd.DataFrame): RBICS マスターデータ（未使用）
            
        Returns:
            pd.DataFrame: マッピングデータ。以下の列を含む:
                - FACTSET_ENTITY_ID: エンティティID
                - PRODUCT_L6_ID: 製品L6 ID
                - RELABEL_L6_ID: 再ラベルL6 ID
                
        Raises:
            FileNotFoundError: マッピングファイルが見つからない場合
            KeyError: 必要な列が存在しない場合
            
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> mapping_df = processor._load_mapping_df(segment_mapping, rbics_master)
            >>> print(f"マッピングデータ: {len(mapping_df)} 件")
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
        """
        セグメントデータをBayesian用に変換
        
        セグメントスコアデータとROICデータを統合し、
        Bayesianモデリング用のセグメントデータを作成する。
        
        Args:
            segment_scores (pd.DataFrame): セグメントスコアデータ
            segment_roic (pd.DataFrame): セグメントROICデータ（ピボット形式）
            
        Returns:
            pd.DataFrame: 統合されたセグメントデータ。以下の列を含む:
                - FACTSET_ENTITY_ID: エンティティID
                - LABEL: ラベル
                - FTERM_2: 期間
                - CODE_SEGMENT_ADJ: セグメントコード
                - SEG_営業利益(税引後): セグメント営業利益
                - SEG_投下資本(運用ベース): セグメント投下資本
                - SEG_ROIC(運用ベース): セグメントROIC
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> segment_data = processor._prepare_segment_data(scores, roic)
            >>> print(f"セグメントデータ: {len(segment_data)} 件")
        """
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
                           wacc_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        連結データをBayesian用に変換（ROICとWACCを統合）
        
        財務データ、ROICデータ、WACCデータを統合し、
        Bayesianモデリング用の連結データを作成する。
        
        Args:
            financial_data (pd.DataFrame): 財務データ
            consol_roic (pd.DataFrame): 連結ROICデータ（ピボット形式）
            wacc_data (Optional[pd.DataFrame]): WACCデータ（オプション）
            
        Returns:
            pd.DataFrame: 統合された連結データ。以下の列を含む:
                - FACTSET_ENTITY_ID: エンティティID
                - FTERM_2: 期間
                - FISCAL_YEAR: 会計年度
                - 営業利益(税引後): 営業利益
                - 投下資本(運用ベース): 投下資本
                - ROIC(運用ベース): ROIC
                - WACC: WACC（利用可能な場合）
                - その他WACC計算詳細列（利用可能な場合）
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> consol_data = processor._prepare_consol_data(financial, roic, wacc)
            >>> print(f"連結データ: {len(consol_data)} 件")
        """
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
        """
        製品シェアデータを準備（セグメント・連結共通）
        
        ピボット形式の製品シェアデータを平坦化し、
        FTERM_2列を適切な形式に変換する。
        
        Args:
            pivot_data (pd.DataFrame): ピボット形式の製品シェアデータ
            
        Returns:
            pd.DataFrame: 平坦化された製品シェアデータ。
                FTERM_2列がYYYYMM形式の整数に変換される。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> share_data = processor._prepare_product_share(pivot_df)
            >>> print(f"製品シェアデータ: {len(share_data)} 件")
        """
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
                            start_period: int, 
                            end_period: int) -> Dict[str, pd.DataFrame]:
        """
        期間でデータをフィルタリング
        
        指定された期間（YYYYMM形式）に基づいて、
        セグメントデータ、連結データ、製品シェアデータをフィルタリングする。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            start_period (int): 開始期間（YYYYMM形式、例：201909）
            end_period (int): 終了期間（YYYYMM形式、例：202406）
            
        Returns:
            Dict[str, pd.DataFrame]: フィルタリングされたデータ辞書。
                元のデータ辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> filtered_data = processor.filter_data_by_period(
            ...     data, start_period=201909, end_period=202406
            ... )
            >>> print(f"フィルタリング後: {len(filtered_data['consol'])} 件")
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
        
        欠損値の割合が閾値未満のエンティティのみを残す。
        セグメントデータと連結データの両方に対して適用される。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            na_threshold (float): 欠損値の閾値（0.0-1.0）。
                1.0未満で欠損率フィルタリングを実行。
                デフォルトは1.0（フィルタリングなし）。
            
        Returns:
            Dict[str, pd.DataFrame]: フィルタリングされたデータ辞書。
                元のデータ辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> filtered_data = processor.filter_valid_entities(
            ...     data, na_threshold=0.5
            ... )
            >>> print(f"有効エンティティ: {len(filtered_data['consol'])} 件")
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
        
        セグメントデータ、連結データ、製品シェアデータ間で
        エンティティIDの整合性を確保する。全てのデータセットに
        共通して存在するエンティティのみを残す。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            
        Returns:
            Dict[str, pd.DataFrame]: 整合されたデータ辞書。
                元のデータ辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> aligned_data = processor.align_data_entities(data)
            >>> print(f"整合後: {len(aligned_data['consol'])} 件")
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
        
        セグメント製品シェアデータと連結製品シェアデータから、
        全ての期間でシェアが0の製品を特定し、除去する。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            
        Returns:
            Tuple[Dict[str, pd.DataFrame], List[str]]: 
                - クリーニングされたデータ辞書
                - 有効な製品名リスト
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> cleaned_data, valid_products = processor.clean_product_names(data)
            >>> print(f"有効製品数: {len(valid_products)}")
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
        
        FTERM_2列をYYYYMM形式の整数からpandasのdatetime型に変換する。
        セグメントデータ、連結データ、製品シェアデータの全てに適用される。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            
        Returns:
            Dict[str, pd.DataFrame]: 日付変換されたデータ辞書。
                元のデータ辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> converted_data = processor.convert_date_format(data)
            >>> print(f"変換後: {converted_data['consol']['FTERM_2'].dtype}")
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
        
        CODE_SEGMENT_ADJ列から正規表現を使用して
        FACTSET_ENTITY_IDとSEGMENT_ADJを抽出する。
        
        Args:
            segment_data (pd.DataFrame): セグメントデータ
            
        Returns:
            pd.DataFrame: マッピングされたセグメントデータ。
                元のデータにFACTSET_ENTITY_IDとSEGMENT_ADJ列が追加される。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> mapped_data = processor.create_segment_mapping(segment_df)
            >>> print(f"マッピング後: {mapped_data.columns.tolist()}")
        """
        segment_mapped = segment_data.reset_index(drop=True)
        segment_mapped[["FACTSET_ENTITY_ID", "SEGMENT_ADJ"]] = segment_mapped["CODE_SEGMENT_ADJ"].str.extract(r"(.+?)_(.+)")
        return segment_mapped
    
    def merge_product_share_data(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        製品シェアデータとメインデータをマージ
        
        セグメントデータと連結データのそれぞれについて、
        製品シェアデータとメインデータをマージする。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            
        Returns:
            Dict[str, pd.DataFrame]: マージされたデータ辞書。
                元のデータ辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> merged_data = processor.merge_product_share_data(data)
            >>> print(f"マージ後: {len(merged_data['segment_product_share'])} 件")
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
        
        セグメントROIC、連結ROIC、WACC、製品シェアデータから
        ピボットテーブルを作成する。時系列補完も実行される。
        
        Args:
            data (Dict[str, pd.DataFrame]): データ辞書
            prod_names (List[str]): 製品名リスト
            
        Returns:
            Dict[str, pd.DataFrame]: ピボットテーブル辞書。以下のキーを含む:
                - Y_segment: セグメントROICピボットテーブル
                - Y_consol: 連結ROICピボットテーブル
                - Y_consol_wacc: WACCピボットテーブル（利用可能な場合）
                - X2_segment: セグメント製品シェアピボットテーブル
                - X2_consol: 連結製品シェアピボットテーブル
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> pivot_tables = processor.create_pivot_tables(data, product_names)
            >>> print(f"ピボットテーブル数: {len(pivot_tables)}")
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
        logger.info(f"連結ROICピボットテーブル作成開始: {len(data['consol'])} エンティティ")
        pivot_tables['Y_consol'] = pd.pivot_table(
            data=data['consol'],
            columns=["FTERM_2"],
            index="FACTSET_ENTITY_ID",
            values="ROIC(運用ベース)",
        )
        pivot_tables['Y_consol'].sort_index(inplace=True)
        logger.info(f"連結ROICピボットテーブル作成完了: {pivot_tables['Y_consol'].shape}")
        
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
        
        # 時系列補完（セグメント）
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
        
        # 時系列補完（連結）
        pivot_tables['X2_consol'].loc[:, prod_names] = (
            pivot_tables['X2_consol']
            .groupby(by=["FACTSET_ENTITY_ID"], group_keys=False)
            .apply(lambda _df: _df[prod_names].ffill().bfill())
            .to_numpy()
        )
        pivot_tables['X2_consol'].sort_index(inplace=True)
        
        # データ整合処理（元のnotebookと完全に同様の処理）
        
        # セグメント側の処理：X2_segment に存在する月だけを Y_segment の列として残す
        first_segment_date = pivot_tables['X2_segment'].index.get_level_values(1).unique()[0]
        segment_entities_in_first_date = pivot_tables['X2_segment'].xs(first_segment_date, level=1).index
        pivot_tables['Y_segment'] = pivot_tables['Y_segment'].loc[segment_entities_in_first_date]
        
        # 連結側の処理：X2_consol に存在する月だけを Y_consol の列として残す  
        first_consol_date = pivot_tables['X2_consol'].index.get_level_values(1).unique()[0]
        consol_entities_in_first_date = pivot_tables['X2_consol'].xs(first_consol_date, level=1).index
        logger.info(f"連結データ整合前: {len(pivot_tables['Y_consol'])} エンティティ")
        logger.info(f"X2_consol初回日付({first_consol_date})のエンティティ数: {len(consol_entities_in_first_date)}")
        pivot_tables['Y_consol'] = pivot_tables['Y_consol'].loc[consol_entities_in_first_date]
        logger.info(f"連結データ整合後: {len(pivot_tables['Y_consol'])} エンティティ")
            
        return pivot_tables
    
    def align_pivot_tables(self, pivot_tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        ピボットテーブル間でインデックスを整合させる（np.stackエラー対策）
        
        全ての時点で共通するエンティティのみを残すことで、
        ピボットテーブル間の整合性を確保する。
        Stan用データ構造作成時のnp.stackエラーを防ぐ。
        
        Args:
            pivot_tables (Dict[str, pd.DataFrame]): ピボットテーブル辞書
            
        Returns:
            Dict[str, pd.DataFrame]: 整合されたピボットテーブル辞書。
                元のピボットテーブル辞書と同じ構造を持つ。
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> aligned_tables = processor.align_pivot_tables(pivot_tables)
            >>> print(f"整合後: {aligned_tables['Y_consol'].shape}")
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
                
            # 早期終了：共通セグメントが空になった場合
            if not common_segments:
                logger.warning("共通セグメントが存在しません")
                break
        
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
        logger.info(f"Y_consol整合前エンティティ数: {len(aligned_tables['Y_consol'])}")
        common_entities = None
        for date in consol_dates:
            date_entities = set(aligned_tables['X2_consol'].xs(date, level=1).index)
            if common_entities is None:
                common_entities = date_entities
            else:
                common_entities = common_entities.intersection(date_entities)
                
            # 早期終了：共通エンティティが空になった場合
            if not common_entities:
                logger.warning("共通エンティティが存在しません")
                break
        
        logger.info(f"全時点共通エンティティ数: {len(common_entities)}")
        
        # 共通エンティティのみを残す
        common_entities = sorted(list(common_entities))
        aligned_tables['X2_consol'] = aligned_tables['X2_consol'].loc[
            aligned_tables['X2_consol'].index.get_level_values(0).isin(common_entities)
        ]
        aligned_tables['Y_consol'] = aligned_tables['Y_consol'].loc[common_entities]
        logger.info(f"Y_consol整合後エンティティ数: {len(aligned_tables['Y_consol'])}")
        
        # WACCピボットテーブルも整合（存在する場合）
        if 'Y_consol_wacc' in aligned_tables:
            logger.info(f"WACC整合前Y_consolエンティティ数: {len(aligned_tables['Y_consol'])}")
            logger.info(f"WACC整合前Y_consol_waccエンティティ数: {len(aligned_tables['Y_consol_wacc'])}")
            
            # WACCテーブルの共通エンティティと時点の整合
            wacc_entities = set(aligned_tables['Y_consol_wacc'].index)
            final_common_entities = set(common_entities).intersection(wacc_entities)
            final_common_entities = sorted(list(final_common_entities))
            
            logger.info(f"WACCとの共通エンティティ数: {len(final_common_entities)}")
            
            # 全テーブルを最終的な共通エンティティに限定
            aligned_tables['Y_consol'] = aligned_tables['Y_consol'].loc[final_common_entities]
            aligned_tables['Y_consol_wacc'] = aligned_tables['Y_consol_wacc'].loc[final_common_entities]
            aligned_tables['X2_consol'] = aligned_tables['X2_consol'].loc[
                aligned_tables['X2_consol'].index.get_level_values(0).isin(final_common_entities)
            ]
            
            logger.info(f"WACC整合後Y_consolエンティティ数: {len(aligned_tables['Y_consol'])}")
            logger.info(f"WACC整合後Y_consol_waccエンティティ数: {len(aligned_tables['Y_consol_wacc'])}")
        
        logger.info("ピボットテーブル整合完了")
        return aligned_tables
    
    def create_stan_data_structure(self, pivot_tables: Dict[str, pd.DataFrame], 
                                 prod_names: List[str]) -> Dict[str, Union[int, np.ndarray]]:
        """
        Stan用のデータ構造を作成
        
        ピボットテーブルからStanモデリング用のデータ構造を作成する。
        次元数、観測値、インデックス、シェアマトリックスなどを含む。
        
        Args:
            pivot_tables (Dict[str, pd.DataFrame]): ピボットテーブル辞書
            prod_names (List[str]): 製品名リスト
            
        Returns:
            Dict[str, Union[int, np.ndarray]]: Stan用データ辞書。以下のキーを含む:
                - Segment_N, Time_N, Product_N, Company_N_c: 次元数
                - Share, Share_consol: シェアマトリックス
                - Seg_ROIC, Consol_ROIC, Consol_WACC: 観測値
                - segment_index, consol_index, wacc_index: エンティティインデックス
                - non_na_index, non_na_index_consol, non_na_index_wacc: 非欠損値インデックス
                - N_obs, N_obs_consol, N_obs_wacc: 観測値数
                - N_pred_term: 予測期間数
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> stan_data = processor.create_stan_data_structure(pivot_tables, products)
            >>> print(f"Stanデータ次元: {stan_data['Company_N_c']} 企業")
        """
        stan_data = {}
        
        # 基本次元数
        stan_data["Segment_N"], stan_data["Time_N"] = pivot_tables['Y_segment'].shape
        stan_data["Product_N"] = len(prod_names)
#        stan_data["Company_N"] = len(pivot_tables['Y_consol'].index)
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
        
        # 観測値の処理（欠損値除去）
        # セグメントROICデータ
        y_segment_flat = pivot_tables['Y_segment'].to_numpy().ravel(order='F')
        segment_valid_mask = ~np.isnan(y_segment_flat)
        stan_data["Seg_ROIC"] = y_segment_flat[segment_valid_mask]
        stan_data["non_na_index"] = np.argwhere(segment_valid_mask).ravel(order='F') + 1
        stan_data["N_obs"] = len(stan_data["Seg_ROIC"]) 
        
        # 連結ROICデータ
        y_consol_flat = pivot_tables['Y_consol'].to_numpy().ravel(order='F')
        consol_valid_mask = ~np.isnan(y_consol_flat)
        stan_data["Consol_ROIC"] = y_consol_flat[consol_valid_mask]
        stan_data["non_na_index_consol"] = np.argwhere(consol_valid_mask).ravel(order='F') + 1
        stan_data["N_obs_consol"] = len(stan_data["Consol_ROIC"]) 
        
        # WACCデータの処理（存在する場合）
        if 'Y_consol_wacc' in pivot_tables:
            y_wacc_flat = pivot_tables['Y_consol_wacc'].to_numpy().ravel(order='F')
            wacc_valid_mask = ~np.isnan(y_wacc_flat)
            
            stan_data["Consol_WACC"] = y_wacc_flat[wacc_valid_mask]
            stan_data["non_na_index_wacc"] = np.argwhere(wacc_valid_mask).ravel(order='F') + 1
            stan_data["N_obs_wacc"] = len(stan_data["Consol_WACC"]) 
            
            # WACCエンティティインデックス（ROICと同じ構造）
            wacc_codes = pivot_tables['Y_consol_wacc'].index.values
            self.label_encoder.fit(wacc_codes)
            codes_cat_wacc = np.repeat(
                self.label_encoder.transform(wacc_codes).astype("int32").reshape(-1, 1),
                pivot_tables['Y_consol_wacc'].shape[1],
                axis=1,
            )
            stan_data["wacc_index"] = codes_cat_wacc.ravel(order="F")[wacc_valid_mask] + 1
            logger.info(f"WACC Stan データ作成: {stan_data['N_obs_wacc']} 観測値")
        else:
            # WACCデータがない場合のデフォルト値
            stan_data["Consol_WACC"] = np.array([])
            stan_data["non_na_index_wacc"] = np.array([])
            stan_data["N_obs_wacc"] = 0
            stan_data["wacc_index"] = np.array([])
            logger.warning("WACC データが利用できません")
        
        # エンティティインデックスの作成
        # セグメントインデックス
        segment_codes = pivot_tables['Y_segment'].index.values
        self.label_encoder.fit(segment_codes)
        codes_cat = np.repeat(
            self.label_encoder.transform(segment_codes).astype("int32").reshape(-1, 1),
            pivot_tables['Y_segment'].shape[1],
            axis=1,
        )
        stan_data["segment_index"] = codes_cat.ravel(order="F")[segment_valid_mask] + 1
        stan_data["segment_index_vec"] = codes_cat[:, 0] + 1
        stan_data["N_segment_index_vec"] = len(stan_data["segment_index_vec"])
        
        # 連結インデックス
        consol_codes = pivot_tables['Y_consol'].index.values
        self.label_encoder.fit(consol_codes)
        codes_cat_consol = np.repeat(
            self.label_encoder.transform(consol_codes).astype("int32").reshape(-1, 1),
            pivot_tables['Y_consol'].shape[1],
            axis=1,
        )
        stan_data["consol_index"] = codes_cat_consol.ravel(order="F")[consol_valid_mask] + 1
        stan_data["consol_index_vec"] = codes_cat_consol[:, 0] + 1
        stan_data["N_consol_index_vec"] = len(stan_data["consol_index_vec"])
        
        # 予測期間
        #stan_data["N_pred_term"] = PREDICTION_PERIODS
        
        return stan_data
    
    def save_processed_data(self, processed_data: Dict[str, Any], save_path: str) -> None:
        """
        処理済みデータを保存
        
        処理済みデータをpickle形式で指定されたパスに保存する。
        保存先ディレクトリが存在しない場合は自動作成する。
        
        Args:
            processed_data (Dict[str, Any]): 処理済みデータ辞書
            save_path (str): 保存先パス
            
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> processor.save_processed_data(data, "/path/to/output.pkl")
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"データを保存しました: {save_path}")
    
    def process_full_pipeline(self, data_manager: FactSetDataManager,
                            start_period: int,
                            end_period: int,
                            save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        ベイジアンモデリング用データ前処理の全パイプラインを実行
        
        データ取得からStan用データ構造作成まで、
        ベイジアンモデリング用のデータ前処理を一貫して実行する。
        各ステップでモニタリングとログ出力を行う。
        
        Args:
            data_manager (FactSetDataManager): 初期化済みのFactSetDataManager
            start_period (int): 開始期間（YYYYMM形式、例：201909）
            end_period (int): 終了期間（YYYYMM形式、例：202406）
            save_path (Optional[str]): 保存先パス（Noneの場合は保存しない）
            
        Returns:
            Dict[str, Any]: 処理済みデータ辞書。以下のキーを含む:
                - raw_data: 生データ（セグメント、連結、製品シェアデータ）
                - pivot_tables: ピボットテーブル辞書
                - stan_data: Stan用データ構造
                - product_names: 製品名リスト
                - entity_info: エンティティ情報
                - processing_info: 処理情報（期間、件数など）
                
        Examples:
            >>> processor = BayesianDataProcessor()
            >>> data_manager = FactSetDataManager()
            >>> result = processor.process_full_pipeline(
            ...     data_manager=data_manager,
            ...     start_period=201909,
            ...     end_period=202406,
            ...     save_path="/path/to/output.pkl"
            ... )
            >>> print(f"処理完了: {result['processing_info']}")
        """
        logger.info("=== ベイジアンモデリング用データ前処理開始 ===")
        
        # 1. データ取得
        with monitor_data_processing("データ取得", logger) as monitor:
            logger.info("1. FactSetDataManagerからデータ取得中...")
            data = self.load_from_data_manager(data_manager)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 2. Entity情報の準備（国別可視化用）
        with monitor_data_processing("Entity情報準備", logger) as monitor:
            logger.info("2. Entity情報準備中...")
            entity_info = data_manager.get_entity_info()[["FACTSET_ENTITY_ID", "ISO_COUNTRY_FACT"]].copy()
            entity_info['COUNTRY_NAME'] = entity_info['ISO_COUNTRY_FACT'].apply(get_country_name)
            monitor.set_final_data(len(entity_info))
        
        # 3. 期間フィルタリング
        with monitor_data_processing("期間フィルタリング", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info(f"3. 期間フィルタリング中... ({start_period}-{end_period})")
            data = self.filter_data_by_period(data, start_period, end_period)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 4. 有効エンティティフィルタリング
        with monitor_data_processing("有効エンティティフィルタリング", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info("4. 有効エンティティフィルタリング中...")
            data = self.filter_valid_entities(data)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 5. エンティティ整合
        with monitor_data_processing("エンティティ整合", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info("5. エンティティ整合中...")
            data = self.align_data_entities(data)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 6. 製品名クリーニング
        with monitor_data_processing("製品名クリーニング", logger) as monitor:
            logger.info("6. 製品名クリーニング中...")
            data, prod_names = self.clean_product_names(data)
            monitor.set_final_data(len(prod_names))
        
        # 7. 日付フォーマット変換
        with monitor_data_processing("日付フォーマット変換", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info("7. 日付フォーマット変換中...")
            data = self.convert_date_format(data)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 8. 製品シェアデータマージ
        with monitor_data_processing("製品シェアデータマージ", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info("8. 製品シェアデータマージ中...")
            data = self.merge_product_share_data(data)
            monitor.set_final_data(data['consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 9. ピボットテーブル作成
        with monitor_data_processing("ピボットテーブル作成", logger, initial_data=data['consol'], entity_id_col="FACTSET_ENTITY_ID", track_columns=["ISO_COUNTRY_FACT"]) as monitor:
            logger.info("9. ピボットテーブル作成中...")
            pivot_tables = self.create_pivot_tables(data, prod_names)
            monitor.set_final_data(pivot_tables['Y_consol'], entity_id_col="FACTSET_ENTITY_ID")
        
        # 10. ピボットテーブル整合
        with monitor_data_processing("ピボットテーブル整合", logger, initial_data=pivot_tables['Y_consol'], entity_id_col="FACTSET_ENTITY_ID") as monitor:
            logger.info("10. ピボットテーブル整合中...")
            pivot_tables = self.align_pivot_tables(pivot_tables)
            monitor.set_final_data(pivot_tables['Y_consol'], entity_id_col="FACTSET_ENTITY_ID")
            
            # 最終的な国別企業数のモニタリング
            final_entities = set(pivot_tables['Y_consol'].index)
            final_entity_info = entity_info[entity_info['FACTSET_ENTITY_ID'].isin(final_entities)]
            if len(final_entity_info) > 0:
                monitor.log_column_distribution(final_entity_info, "ISO_COUNTRY_FACT", "最終的な")
                    
        # 11. Stan用データ構造作成
        with monitor_data_processing("Stan用データ構造作成", logger) as monitor:
            logger.info("11. Stan用データ構造作成中...")
            stan_data = self.create_stan_data_structure(pivot_tables, prod_names)
            monitor.set_final_data(stan_data['Company_N_c'])
        
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


