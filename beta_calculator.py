"""
ベータ計算クラス

企業の株価リターンとインデックスリターンからベータ値を計算
グローバルベータとローカルベータの両方をサポート（ローカルはまだ未実装）
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from datetime import date, timedelta
import pandas as pd
import numpy as np
import logging
from enum import Enum
from gppm.utils.config_manager import get_logger
from wolf_period import WolfPeriod, PeriodFrequency

logger = get_logger(__name__)


class ReturnFrequency(Enum):
    """リターン計算の頻度"""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'


class BetaType(Enum):
    """ベータの種類"""
    GLOBAL = 'global'
    LOCAL = 'local'


@dataclass
class BetaResult:
    """ベータ計算結果"""
    beta: float
    beta_type: BetaType
    security_id: str
    index_code: str
    index_name: str
    start_date: date
    end_date: date
    frequency: ReturnFrequency
    num_observations: int
    r_squared: float
    correlation: float
    security_volatility: float
    index_volatility: float
    levered: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'beta': self.beta,
            'beta_type': self.beta_type.value,
            'security_id': self.security_id,
            'index_code': self.index_code,
            'index_name': self.index_name,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'frequency': self.frequency.value,
            'num_observations': self.num_observations,
            'r_squared': self.r_squared,
            'correlation': self.correlation,
            'security_volatility': self.security_volatility,
            'index_volatility': self.index_volatility,
            'levered': self.levered
        }


class BetaCalculator:
    """
    ベータ計算クラス
    
    与えられたリターン時系列からベータと統計量を算出します。
    データ取得や集約は行いません。
    """

    @staticmethod
    def calculate_from_returns(
        returns: pd.DataFrame,
        *,
        security_id: str,
        index_code: str,
        index_name: str,
        frequency: ReturnFrequency = ReturnFrequency.MONTHLY,
        beta_type: BetaType = BetaType.GLOBAL,
        levered: bool = True,
        calc_start_date: Optional[date] = None,
        calc_end_date: Optional[date] = None
    ) -> Optional[BetaResult]:
        """リターンデータからベータを計算
        期待カラム: 'security_return', 'index_return'（DatetimeIndex）
        """
        # デバッグ: 受け取ったデータの状態（DEBUGレベルのみ）
        
        logger.debug(f"{security_id}: 受信データ shape={returns.shape}, columns={returns.columns.tolist() if not returns.empty else '[]'}")
        if not returns.empty:
            logger.debug(f"{security_id}: データ期間 {returns.index[0]} ~ {returns.index[-1]}")
            logger.debug(f"{security_id}: 欠損値 security_return={returns['security_return'].isna().sum()}, index_return={returns['index_return'].isna().sum()}")
        
        returns = returns.dropna()
        logger.debug(f"{security_id}: dropna後 shape={returns.shape}")
        
        if len(returns) < 20:
            logger.debug(f"観測数不足: {security_id} len={len(returns)} < 20")
            if len(returns) > 0:
                logger.debug(f"{security_id}: 存在するデータ数={len(returns)}, 期間={returns.index[0]}~{returns.index[-1]}")
            return None

        beta_stats = BetaCalculator._calculate_beta_statistics(
            returns['security_return'],
            returns['index_return']
        )

        # WolfPeriodを使用して日付を取得
        start_date = WolfPeriod.from_date(returns.index[0], PeriodFrequency.DAILY).start_time.date()
        end_date = WolfPeriod.from_date(returns.index[-1], PeriodFrequency.DAILY).end_time.date()

        return BetaResult(
            beta=beta_stats['beta'],
            beta_type=beta_type,
            security_id=security_id,
            index_code=index_code,
            index_name=index_name,
            start_date=start_date,
            end_date=end_date,
            frequency=frequency,
            num_observations=len(returns),
            r_squared=beta_stats['r_squared'],
            correlation=beta_stats['correlation'],
            security_volatility=beta_stats['security_volatility'],
            index_volatility=beta_stats['index_volatility'],
            levered=levered
        )
    
    def _convert_to_isin(self, security_id: str, identifier_type: str) -> Optional[str]:
        """証券識別子をISINに変換"""
        if identifier_type.lower() == 'isin':
            return security_id
        
        company = None
        if identifier_type.lower() == 'fsym':
            company = self.company_manager.get_company_by_fsym_id(security_id)
        elif identifier_type.lower() == 'sedol':
            company = self.company_manager.get_company_by_sedol(security_id)
        elif identifier_type.lower() == 'cusip':
            company = self.company_manager.get_company_by_cusip(security_id)
        
        if company and company.isin:
            return company.isin
        return None
    
    # ===== ヘルパー =====
    @staticmethod
    def merge_returns(security_returns: pd.DataFrame, index_returns: pd.DataFrame) -> pd.DataFrame:
        logger.debug(f"マージ前: security shape={security_returns.shape}, index shape={index_returns.shape}")
        merged = pd.merge(
            security_returns,
            index_returns,
            left_index=True,
            right_index=True,
            how='inner'
        )
        logger.debug(f"マージ後(dropna前): shape={merged.shape}")
        result = merged.dropna()
        logger.debug(f"マージ後(dropna後): shape={result.shape}")
        return result
    
    @staticmethod
    def compute_daily_returns_from_prices(daily_prices: pd.Series) -> pd.Series:
        """日次の価格系列から日次リターンを算出"""
        prices = daily_prices.dropna().copy()
        logger.debug(f"価格数(欠損値除外後): {len(prices)}")
        prices.index = pd.to_datetime(prices.index)
        prices = prices.sort_index()
        returns = prices.pct_change()
        logger.debug(f"日次リターン数: {len(returns)}, 欠損値={returns.isna().sum()}")
        return returns

    @staticmethod
    def compute_monthly_compounded_returns_from_daily(daily_returns: pd.Series) -> pd.Series:
        """日次のリターン系列から月次複利リターンを算出"""
        daily = daily_returns.dropna().copy()
        logger.debug(f"日次リターン数(欠損値除外後): {len(daily)}")
        daily.index = pd.to_datetime(daily.index)
        
        # 月次でグループ化して複利リターンを計算
        monthly = (
            daily.groupby(daily.index.to_period('M'))
            .apply(lambda s: np.prod(1.0 + s) - 1.0)
        )
        monthly.index = monthly.index.to_timestamp()
        logger.debug(f"月次複利リターン数: {len(monthly)}")
        return monthly

    @staticmethod
    def compute_index_returns_from_values(values: pd.DataFrame, frequency: str = 'monthly') -> pd.Series:
        """指数価格（INDEX_VALUE, USD_RATE）から指数リターンを算出
        
        Args:
            values: カラムに 'DATA_DATE','INDEX_VALUE','USD_RATE' を期待
            frequency: サンプリング頻度（'weekly' または 'monthly'）
            
        Returns:
            pd.Series: DatetimeIndex のリターン系列
        """
        required_columns = {'DATA_DATE', 'INDEX_VALUE', 'USD_RATE'}
        if not required_columns.issubset(values.columns):
            missing = required_columns - set(values.columns)
            raise ValueError(f"必要なカラムが不足しています: {missing}")
        
        logger.debug(f"インデックス値入力: shape={values.shape}")
        df = values.copy()
        df = df.sort_values('DATA_DATE')
        df['DATA_DATE'] = pd.to_datetime(df['DATA_DATE'])
        df = df.set_index('DATA_DATE')

        logger.debug(f"インデックス値(USD変換前) head:\n{df.head().to_string()}")
        index_in_usd = (df['INDEX_VALUE'] / df['USD_RATE']).dropna()
        logger.debug(f"USD換算後の指数値数: {len(index_in_usd)}")
        if not index_in_usd.empty:
            logger.debug(f"USD換算後の指数値 head:\n{index_in_usd.head().to_string()}")
            logger.debug(f"USD換算後の指数値 tail:\n{index_in_usd.tail().to_string()}")
        
        freq = (frequency or 'monthly').lower()
        if freq == 'weekly':
            sampled = index_in_usd.resample('W').last()
        else:
            # 月次リサンプル
            sampled = index_in_usd.resample('M').last()
        logger.debug(f"リサンプル後({freq}): {len(sampled)}件")
        if not sampled.empty:
            logger.debug(f"リサンプル後のデータ head:\n{sampled.head().to_string()}")
            logger.debug(f"リサンプル後のデータ tail:\n{sampled.tail().to_string()}")
        
        returns = sampled.pct_change()
        logger.debug(f"インデックスリターン数: {len(returns)}, 欠損値={returns.isna().sum()}")
        return returns
    
    @staticmethod
    def _calculate_beta_statistics(security_returns: pd.Series, index_returns: pd.Series) -> Dict[str, float]:
        """ベータと関連統計量を計算"""
        covariance = np.cov(security_returns, index_returns)[0, 1]
        index_variance = np.var(index_returns)
        beta = covariance / index_variance if index_variance != 0 else np.nan
        correlation = np.corrcoef(security_returns, index_returns)[0, 1]
        r_squared = correlation ** 2
        security_volatility = np.std(security_returns) * np.sqrt(12)
        index_volatility = np.std(index_returns) * np.sqrt(12)
        return {
            'beta': beta,
            'r_squared': r_squared,
            'correlation': correlation,
            'security_volatility': security_volatility,
            'index_volatility': index_volatility
        }
    
    def unleverage_beta(self, levered_beta: float, debt_to_equity: float, tax_rate: float = 0.30) -> float:
        """レバードベータをアンレバードベータに変換"""
        return levered_beta / (1 + (1 - tax_rate) * debt_to_equity)
    
    def releverage_beta(self, unlevered_beta: float, target_debt_to_equity: float, tax_rate: float = 0.30) -> float:
        """アンレバードベータをレバードベータに変換"""
        return unlevered_beta * (1 + (1 - tax_rate) * target_debt_to_equity)
    