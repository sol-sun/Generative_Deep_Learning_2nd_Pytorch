"""
WACC計算クラス

国別・企業別WACC（加重平均資本コスト）の計算機能を提供
"""

from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from gppm.utils.country_code_manager import get_country_name, convert_to_alpha2
from gppm.utils.config_manager import get_logger
from gppm.utils.country_risk_parameters import CountryRiskParams

# ロギング設定
logger = get_logger(__name__)


@dataclass
class WACCColumnConfig:
    """WACC計算用のカラム名設定"""
    # 株主資本の市場価値関連
    market_value_col: str = "FF_MKT_VAL"
    price_col: str = "FF_PRICE_CLOSE_FP"
    shares_out_col: str = "FF_COM_SHS_OUT"
    
    # 負債関連
    total_debt_col: str = "FF_DEBT"
    debt_lt_col: str = "FF_DEBT_LT"
    debt_st_col: str = "FF_DEBT_ST"
    
    # 金利・税率関連
    tax_rate_col: str = "FF_TAX_RATE"
    eff_int_rate_col: str = "FF_EFF_INT_RATE"
    int_exp_debt_col: str = "FF_INT_EXP_DEBT"      # 有利子負債利息（優先）
    int_exp_total_col: str = "FF_INT_EXP_TOT"      # 総支払利息（代替）
    
    # 国・エンティティ関連
    country_col: str = "ISO_COUNTRY"
    country_fact_col: str = "ISO_COUNTRY_FACT"     # FactSetで実際に使用されているカラム名
    entity_id_col: str = "FACTSET_ENTITY_ID"


# CountryRiskParamsはgppm.utils.country_risk_parametersからインポート


class WACCCalculator:
    """WACC（加重平均資本コスト）計算クラス"""
    
    def __init__(self, config: Optional[WACCColumnConfig] = None):
        self.config = config or WACCColumnConfig()
        self.country_params: Dict[str, CountryRiskParams] = {}
        self.default_params: Optional[CountryRiskParams] = None
    
    def set_country_params(self, country_params: Dict[str, CountryRiskParams]) -> None:
        """
        国別リスクパラメータを設定
        """
        # 国コードを2文字形式に正規化してから保存
        self.country_params = {}
        for code, params in country_params.items():
            normalized_code = convert_to_alpha2(code)
            if normalized_code:
                self.country_params[normalized_code] = params
                country_name = get_country_name(normalized_code)
                logger.info(f"{country_name} ({normalized_code}) のリスクパラメータを設定")
            else:
                logger.warning(f"無効な国コード '{code}' はスキップされました")
    
    def set_default_params(self, default_params: CountryRiskParams) -> None:
        """
        デフォルトリスクパラメータを設定
        国別パラメータが設定されていない国に適用される
        """
        self.default_params = default_params
        logger.info(
            f"デフォルトパラメータを設定: リスクフリーレート={default_params.risk_free_rate:.1%}, "
            f"市場リスクプレミアム={default_params.market_risk_premium:.1%}, "
            f"法人税率={default_params.country_tax_rate:.1%}"
        )
    
    def get_country_params(self, country_code: str) -> Optional[CountryRiskParams]:
        """国別パラメータを取得（デフォルト値フォールバック付き）"""
        normalized_code = convert_to_alpha2(country_code)
        if not normalized_code:
            logger.warning(f"無効な国コード: {country_code}")
            return self.default_params
        
        if normalized_code in self.country_params:
            return self.country_params[normalized_code]
        elif self.default_params is not None:
            country_name = get_country_name(normalized_code)
            logger.debug(f"国コード '{normalized_code}' ({country_name}) のパラメータが未設定のため、デフォルト値を使用")
            return self.default_params
        else:
            country_name = get_country_name(normalized_code)
            logger.warning(f"国コード '{normalized_code}' ({country_name}) のパラメータとデフォルト値が両方とも未設定")
            return None
    
    def _ensure_series(self, value: Union[float, pd.Series], index: pd.Index) -> pd.Series:
        """スカラー値をSeriesに変換してベクトル化"""
        if np.isscalar(value):
            return pd.Series(value, index=index)
        return value
    
    def calculate_average_debt(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        期中平均有利子負債を計算してデータフレームに追加
        """
        data = financial_data.copy()
        
        if self.config.total_debt_col not in data.columns:
            # 長期負債+短期負債で計算
            if self.config.debt_lt_col in data.columns and self.config.debt_st_col in data.columns:
                total_debt = data[self.config.debt_lt_col] + data[self.config.debt_st_col]
            else:
                logger.warning("負債データが不足しています")
                data['平均有利子負債'] = np.nan
                return data
        else:
            total_debt = data[self.config.total_debt_col]
        
        # 期首・期末平均（1期前とのaverage）
        avg_debt = (total_debt + total_debt.shift(1)) / 2
        
        # 最初の期間は期末値をそのまま使用
        avg_debt.fillna(total_debt, inplace=True)
        
        # 結果をデータフレームに設定
        data['平均有利子負債'] = avg_debt
        
        return data
    
    def calculate_market_value_equity(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        株主資本の市場価値を計算
        """
        data = financial_data.copy()
        
        # 市場価値が直接提供されている場合はそれを使用
        if self.config.market_value_col in data.columns:
            data['時価総額'] = data[self.config.market_value_col]
        # そうでなければ株価 × 発行済み株式数で計算
        elif self.config.price_col in data.columns and self.config.shares_out_col in data.columns:
            data['時価総額'] = data[self.config.price_col] * data[self.config.shares_out_col]
        else:
            logger.warning("時価総額の計算に必要なデータが不足しています")
            data['時価総額'] = np.nan
            
        return data
    
    def calculate_cost_of_debt(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        負債コストを計算（税引前・税引後を明示的に分離）
        """
        data = financial_data.copy()
        
        # 税引前負債コストを計算
        if self.config.eff_int_rate_col in data.columns:
            # 1. 実効金利が直接提供されている場合（最優先）
            # データプロバイダーで既に小数点形式に変換済み
            data['負債コスト（税引前）'] = data[self.config.eff_int_rate_col]
            
        elif (self.config.int_exp_debt_col in data.columns and 
              '平均有利子負債' in data.columns):
            # 2. 有利子負債利息 ÷ 平均有利子負債で計算（2番目の優先度）
            data['負債コスト（税引前）'] = (
                data[self.config.int_exp_debt_col] / data['平均有利子負債']
            )
            
            # 有利子負債利息がない場合は総支払利息で代替
            if self.config.int_exp_total_col in data.columns:
                mask = pd.isna(data['負債コスト（税引前）'] )
                data.loc[mask, '負債コスト（税引前）'] = (
                    data.loc[mask, self.config.int_exp_total_col] / 
                    data.loc[mask, '平均有利子負債']
                )
                
        elif (self.config.int_exp_total_col in data.columns and 
              '平均有利子負債' in data.columns):
            # 3. 総支払利息のみ利用可能な場合（最後の選択肢）
            logger.warning("有利子負債利息が不足しているため、総支払利息を使用します")
            data['負債コスト（税引前）'] = (
                data[self.config.int_exp_total_col] / data['平均有利子負債']
            )
        else:
            logger.warning("負債コスト計算に必要なデータが不足しています")
            data['負債コスト（税引前）'] = np.nan
        
        # 税引後負債コストを計算（WACCで使用）
        if self.config.tax_rate_col in data.columns:
            # データプロバイダーで既に小数点形式に変換済み
            tax_rate = data[self.config.tax_rate_col]
        else:
            tax_rate = 0.30  # デフォルト税率30%
            logger.warning("税率データが不足しているため、デフォルト値30%を使用します")
        
        # ベクトル化
        tax_rate = self._ensure_series(tax_rate, data.index)
        data['負債コスト（税引後）'] = data['負債コスト（税引前）'] * (1 - tax_rate)
        
        return data
    
    def estimate_cost_of_equity_capm(self, 
                                    risk_free_rate: Union[float, pd.Series], 
                                    market_risk_premium: Union[float, pd.Series],
                                    beta: pd.Series) -> pd.Series:
        """
        CAPM（資本資産価格モデル）による株主資本コスト推定
        """
        # ベクトル化
        rf = self._ensure_series(risk_free_rate, beta.index)
        erp = self._ensure_series(market_risk_premium, beta.index)
        
        return rf + beta * erp
    
    def calculate_wacc(self, 
                      financial_data: pd.DataFrame, 
                      cost_of_equity: Union[float, pd.Series, None] = None,
                      beta: Optional[pd.Series] = None,
                      risk_free_rate: Union[float, pd.Series, None] = None,
                      market_risk_premium: Union[float, pd.Series, None] = None,
                      tax_rate: Union[float, pd.Series, None] = None,
                      use_column_tax_rate: bool = True,
                      use_country_params: bool = False) -> pd.DataFrame:
        """
        WACC計算
        """
        data = financial_data.copy()
        
        # 必要なカラムが存在するかチェック
        required_cols = ['時価総額', '平均有利子負債', '負債コスト（税引前）']
        if not all(col in data.columns for col in required_cols):
            logger.warning("WACC計算に必要なカラムが不足しています")
            data['WACC'] = np.nan
            return data
        
        # 企業価値を計算
        data['企業価値'] = data['時価総額'] + data['平均有利子負債']
        
        # 株主資本比率と負債比率を計算
        data['株主資本比率'] = data['時価総額'] / data['企業価値']
        data['負債比率'] = data['平均有利子負債'] / data['企業価値']
        
        # 株主資本コストの設定
        if cost_of_equity is not None:
            cost_of_equity_series = self._ensure_series(cost_of_equity, data.index)
        elif beta is not None and risk_free_rate is not None and market_risk_premium is not None:
            # CAPMで計算
            cost_of_equity_series = self.estimate_cost_of_equity_capm(
                risk_free_rate, market_risk_premium, beta
            )
        elif use_country_params and self.country_params:
            # 国別パラメータを使用
            cost_of_equity_series = self._calculate_country_based_cost_of_equity(data, beta)
        else:
            logger.warning("株主資本コストが指定されていません。別途計算が必要です")
            data['WACC'] = np.nan
            return data
        
        # 税率の設定（負債コストはすでに税引後）
        # ここでは記録用として保持
        if use_column_tax_rate and self.config.tax_rate_col in data.columns:
            effective_tax_rate = data[self.config.tax_rate_col]
        elif tax_rate is not None:
            effective_tax_rate = tax_rate
        else:
            effective_tax_rate = 0.30
            logger.warning("税率が指定されていません。デフォルト値30%を使用します")
        
        effective_tax_rate = self._ensure_series(effective_tax_rate, data.index)
        
        # WACC計算: (E/V) * Re + (D/V) * Rd(after-tax)
        data['WACC'] = (
            data['株主資本比率'] * cost_of_equity_series +
            data['負債比率'] * data['負債コスト（税引後）']
        )
        
        # 記録用として保持
        data['株主資本コスト'] = cost_of_equity_series
        data['実効税率'] = effective_tax_rate
        
        return data
    
    def _calculate_country_based_cost_of_equity(self, data: pd.DataFrame, beta: Optional[pd.Series]) -> pd.Series:
        """国別パラメータに基づく株主資本コスト計算"""
        if beta is None:
            logger.warning("ベータ値が指定されていません")
            return pd.Series(np.nan, index=data.index)
        
        cost_of_equity = pd.Series(index=data.index, dtype=float)
        
        # 利用可能な国コードカラムを確認
        country_col = None
        if self.config.country_col in data.columns:
            country_col = self.config.country_col
        elif self.config.country_fact_col in data.columns:
            country_col = self.config.country_fact_col
        
        if country_col is None:
            logger.warning("国コードカラムが見つかりません")
            return pd.Series(np.nan, index=data.index)
        
        # データ内のユニークな国を取得
        unique_countries = data[country_col].dropna().unique()
        # 各国についてパラメータを適用
        for country in unique_countries:
            mask = data[country_col] == country
            if not mask.any():
                continue
            
            # 国コードを正規化
            normalized_country = convert_to_alpha2(country)
            if not normalized_country:
                logger.warning(f"無効な国コード: {country}")
                continue
                
            # 国別パラメータを取得（デフォルト値フォールバック付き）
            params = self.get_country_params(normalized_country)
            if params is not None:
                country_cost_of_equity = self.estimate_cost_of_equity_capm(
                    params.risk_free_rate,
                    params.market_risk_premium,
                    beta[mask]
                )
                cost_of_equity[mask] = country_cost_of_equity
                
                country_name = get_country_name(normalized_country)
                logger.debug(f"{country_name} ({normalized_country}) のCAPMパラメータを適用")
            else:
                country_name = get_country_name(normalized_country)
                logger.warning(f"国コード '{normalized_country}' ({country_name}) のパラメータが設定されておらず、デフォルト値も未設定です")
        
        return cost_of_equity
    
    def create_financial_pivot(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        財務データからピボットテーブルを作成
        """
        value_cols = []
        
        # 利用可能なカラムを動的に選択
        potential_cols = [
            self.config.market_value_col, self.config.price_col, self.config.shares_out_col,
            self.config.total_debt_col, self.config.debt_lt_col, self.config.debt_st_col,
            self.config.tax_rate_col, self.config.eff_int_rate_col, 
            self.config.int_exp_debt_col, self.config.int_exp_total_col
        ]
        
        for col in potential_cols:
            if col in financial_data.columns:
                value_cols.append(col)
        
        if not value_cols:
            logger.warning("ピボットテーブル作成に必要なカラムが見つかりません")
            return pd.DataFrame()
        
        return pd.pivot_table(
            financial_data,
            index=["FTERM_2"],
            columns=[self.config.entity_id_col],
            values=value_cols,
            sort=False
        )
    
    def create_country_pivot(self, financial_data: pd.DataFrame) -> pd.DataFrame:
        """
        国別データからピボットテーブルを作成
        """
        # 利用可能な国コードカラムを確認
        country_col_to_use = None
        if self.config.country_col in financial_data.columns:
            country_col_to_use = self.config.country_col
        elif self.config.country_fact_col in financial_data.columns:
            country_col_to_use = self.config.country_fact_col
        else:
            logger.warning("国別ピボットテーブル作成に必要な国コードカラムが見つかりません")
            return pd.DataFrame()
        
        value_cols = ['WACC'] if 'WACC' in financial_data.columns else []
        
        if not value_cols:
            logger.warning("国別ピボットテーブル作成に必要なWACCカラムが見つかりません")
            return pd.DataFrame()
        
        return pd.pivot_table(
            financial_data,
            index=["FTERM_2"],
            columns=[country_col_to_use],
            values=value_cols,
            aggfunc='mean',
            sort=False
        )
    
    def _remove_outliers(self, df: pd.DataFrame, lower_quantile: float = 0.01, 
                        upper_quantile: float = 0.99) -> pd.DataFrame:
        """
        アウトライヤー除去
        """
        df_cleaned = df.copy()
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            upper_threshold = df[col].quantile(upper_quantile)
            lower_threshold = df[col].quantile(lower_quantile)
            df_cleaned[col] = df_cleaned[col].mask(
                (df_cleaned[col] > upper_threshold) | (df_cleaned[col] < lower_threshold)
            )
        
        return df_cleaned
    
    def prepare_wacc_analysis_data(self, company_wacc: pd.DataFrame, 
                                  country_wacc: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """WACC分析用のデータを準備"""
        # データをlong形式に変換
        company_data = company_wacc.stack().to_frame("WACC").reset_index()
        country_data = country_wacc.stack().to_frame("国別平均WACC").reset_index()
        
        # 時期情報を追加
        for data in [company_data, country_data]:
            if 'FTERM_2' in data.columns:
                data['年'] = data['FTERM_2'].astype(str).str[:4].astype(int)
                data['月'] = data['FTERM_2'].astype(str).str[-2:].astype(int)
                data = data.sort_values(by="FTERM_2")
        
        return {
            "company": company_data,
            "country": country_data
        }
    
    def calculate_wacc_from_merged_data(self, 
                                       financial_data: pd.DataFrame,
                                       cost_of_equity: Union[float, pd.Series, None] = None,
                                       beta: Optional[pd.Series] = None,
                                       risk_free_rate: Union[float, pd.Series, None] = None,
                                       market_risk_premium: Union[float, pd.Series, None] = None,
                                       tax_rate: Union[float, pd.Series, None] = None,
                                       use_country_params: bool = False) -> Dict[str, pd.DataFrame]:
        """
        マージ済みデータからWACC計算を実行
        """
        # 市場価値の計算
        data_with_equity = self.calculate_market_value_equity(financial_data)
        data_with_debt = self.calculate_average_debt(data_with_equity)
        data_with_cost_debt = self.calculate_cost_of_debt(data_with_debt)
        
        # WACC計算
        data_with_wacc = self.calculate_wacc(
            data_with_cost_debt, 
            cost_of_equity=cost_of_equity,
            beta=beta,
            risk_free_rate=risk_free_rate,
            market_risk_premium=market_risk_premium,
            tax_rate=tax_rate,
            use_country_params=use_country_params
        )
        
        # ピボットテーブルの作成
        financial_pivot = self.create_financial_pivot(financial_data)
        
        # 企業別WACCピボット
        if 'WACC' in data_with_wacc.columns:
            company_wacc_pivot = pd.pivot_table(
                data_with_wacc,
                index=["FTERM_2"],
                columns=[self.config.entity_id_col],
                values="WACC",
                sort=False
            )
            
            # アウトライヤー除去
            company_wacc_cleaned = self._remove_outliers(company_wacc_pivot)
            company_wacc_cleaned = company_wacc_cleaned.dropna(how='all', axis=0)
            
            # 国別WACCピボット
            country_wacc_pivot = self.create_country_pivot(data_with_wacc)
            country_wacc_cleaned = (
                self._remove_outliers(country_wacc_pivot) 
                if not country_wacc_pivot.empty else pd.DataFrame()
            )
            
            # 分析用データ準備
            analysis_data = self.prepare_wacc_analysis_data(company_wacc_cleaned, country_wacc_cleaned)
        else:
            company_wacc_cleaned = pd.DataFrame()
            country_wacc_cleaned = pd.DataFrame()
            analysis_data = {"company": pd.DataFrame(), "country": pd.DataFrame()}
        
        return {
            "financial_pivot": financial_pivot,
            "processed_data": data_with_wacc,
            "company_wacc": company_wacc_cleaned,
            "country_wacc": country_wacc_cleaned,
            "analysis_data": analysis_data
        }
    
    def calculate_country_average_wacc(self, wacc_data: pd.DataFrame) -> pd.DataFrame:
        """
        国別平均WACCを計算
        """
        # 利用可能な国コードカラムを確認
        country_col_to_use = None
        if self.config.country_col in wacc_data.columns:
            country_col_to_use = self.config.country_col
        elif self.config.country_fact_col in wacc_data.columns:
            country_col_to_use = self.config.country_fact_col
        else:
            logger.warning("国別平均WACC計算に必要な国コードカラムが不足しています")
            return pd.DataFrame()
        
        if 'WACC' not in wacc_data.columns:
            logger.warning("国別平均WACC計算に必要なWACCカラムが不足しています")
            return pd.DataFrame()
        
        country_avg = wacc_data.groupby([country_col_to_use, 'FTERM_2'])['WACC'].mean().reset_index()
        country_avg.columns = [country_col_to_use, 'FTERM_2', '国別平均WACC']
        
        return country_avg


