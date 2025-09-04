"""
ベイジアンモデルビルダーモジュール

Stanモデルの定義、コンパイル、管理を行う
"""

from pathlib import Path
from typing import Optional
from cmdstanpy import CmdStanModel


class BayesianModelBuilder:
    """ベイジアンモデル（Stan）の管理クラス"""
    
    def __init__(self, model_dir: Optional[str] = None):
        self.model_dir = Path(model_dir) if model_dir else Path("./stan_models")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.compiled_models = {}
        
    def get_hierarchical_roic_model_code(self) -> str:
        """階層ベイジアンROICモデルのStanコードを取得"""
        stan_code = """
data {
  int N_Share_consol_index;
  int N_Share_index;
  int N_segment_index_vec;
  int N_consol_index_vec;

  int Company_N;
  int Segment_N;
  int Product_N;
  int Time_N;

  int           N_obs;
  vector[N_obs] Seg_ROIC;
  array[Time_N] matrix[N_Share_index,  Product_N] Share;
  array[N_obs] int  non_na_index;

  int                    N_pred_term;
  array[N_obs]           int segment_index;
  array[N_segment_index_vec] int segment_index_vec;

  int Company_N_c;
  int N_obs_consol;
  array[Time_N] matrix[N_Share_consol_index, Product_N] Share_consol;
  array[N_obs_consol] int  non_na_index_consol;
  array[N_obs_consol]    int consol_index;
  array[N_consol_index_vec] int consol_index_vec;
  vector[N_obs_consol]     Consol_ROIC;
}

parameters {
  matrix<lower = -1, upper = 1>[Product_N, Time_N] Item_ROIC;
  vector<lower = 0>[Product_N]   s_t;
  vector<lower = 0>[Segment_N]   seg_sigma;
  matrix<lower = -3, upper = 3>[Segment_N,  Time_N] segment_private;
  vector<lower = 0>[Segment_N]   s_segment_private;
  vector<lower = 0>[Company_N_c] consol_sigma;
  matrix<lower = -3, upper = 3>[Company_N_c, Time_N]  consol_private;
  vector<lower = 0>[Company_N_c] s_consol_private;
  real<lower = 1> nu_consol_roic;
  real<lower = 1> nu_seg_roic;
}

transformed parameters {
  matrix[Segment_N,    Time_N] segment_private_eff;
  matrix[Company_N_c,  Time_N] consol_private_eff;
  for (t in 1:Time_N) {
    segment_private_eff[:, t] = segment_private[:, t] - mean(segment_private[:, t]);
    consol_private_eff[:,  t] =  consol_private[:, t] - mean(consol_private[:,  t]);
  }
}

model {
  seg_sigma        ~ student_t(3, 0, 0.1);
  consol_sigma     ~ student_t(3, 0, 0.1);
  s_t              ~ student_t(3, 0, 0.5);
  s_segment_private~ student_t(3, 0, 0.5);
  s_consol_private ~ student_t(3, 0, 0.5);

  Item_ROIC[:, 1]      ~ normal(0.04, 0.5);
  Item_ROIC[:, 2]      ~ normal(0.04, 0.5);
  segment_private[:, 1] ~ student_t(3, 0, 0.5);
  consol_private[:, 1]  ~ student_t(3, 0, 0.5);

  for (t in 2:Time_N) {
    segment_private[:, t] ~ student_t(3, segment_private[:, t - 1], s_segment_private);
    consol_private[:,  t] ~ student_t(3,  consol_private[:,  t - 1], s_consol_private);
  }

  for (t in 3:Time_N) {
    Item_ROIC[:, t] ~ normal( (2 * Item_ROIC[:, t - 1]) - Item_ROIC[:, t - 2], s_t );
  }

  matrix[N_Share_index, Time_N] mu;
  for (t in 1:Time_N) {
    mu[, t] = Share[t] * Item_ROIC[, t] + segment_private_eff[segment_index_vec, t];
  }
  target += student_t_lpdf(
              Seg_ROIC | nu_seg_roic,
              to_vector(mu)[non_na_index],
              seg_sigma[segment_index]
            );

  matrix[N_Share_consol_index, Time_N] mu_consol;
  for (t in 1:Time_N) {
    mu_consol[, t] = Share_consol[t] * Item_ROIC[, t] + consol_private_eff[consol_index_vec, t];
  }
  target += student_t_lpdf(
              Consol_ROIC | nu_consol_roic,
              to_vector(mu_consol)[non_na_index_consol],
              consol_sigma[consol_index]
            );
}
"""
        return stan_code


