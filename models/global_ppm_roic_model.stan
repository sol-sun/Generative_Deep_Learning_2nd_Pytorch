data {
  int N_Share_consol_index;
  int N_Share_index;
  int N_segment_index_vec;
  int N_consol_index_vec;

  int Company_N; // 会社の数
  int Segment_N; // 会社のセグメント総数
  int Product_N; // 製品個数
  int Time_N; // 時点の数

  //------------------------------------------------------------------
  // セグメント ROIC 観測値
  //------------------------------------------------------------------
  int           N_obs; // セグメント ROIC 観測値総数
  vector[N_obs] Seg_ROIC; // セグメント ROIC 観測値
  array[Time_N] matrix[N_Share_index,  Product_N] Share; // 企業セグメントごとの製品シェア
  array[N_obs] int  non_na_index;
  //int                    N_pred_term;
  array[N_obs]           int segment_index; // セグメントID
  array[N_segment_index_vec] int segment_index_vec;

  //------------------------------------------------------------------
  // 連結 ROIC 観測値
  //------------------------------------------------------------------
  int Company_N_c; // 連結 ROIC を持つ企業数
  int N_obs_consol; // 連結 ROIC 観測値総数
  array[Time_N] matrix[N_Share_consol_index, Product_N] Share_consol; //連結×製品シェア
  array[N_obs_consol] int  non_na_index_consol;
  array[N_obs_consol]    int consol_index;
  array[N_consol_index_vec] int consol_index_vec;
  vector[N_obs_consol]     Consol_ROIC; // 連結 ROIC 観測値
}

parameters {
  //------------------------------------------------------------------
  // 製品 ROIC（時系列パラメータ）
  //------------------------------------------------------------------
  matrix<lower = -1, upper = 1>[Product_N, Time_N] Item_ROIC; // 製品 ROIC

  //------------------------------------------------------------------
  // 分散パラメータ
  //------------------------------------------------------------------
  vector<lower = 0>[Product_N]   s_t; // 製品ROICの変動の大きさを表す標準偏差
  vector<lower = 0>[Segment_N]   seg_sigma; // セグメント ROIC 観測誤差
  matrix<lower = -3, upper = 3>[Segment_N,  Time_N] segment_private;
  vector<lower = 0>[Segment_N]   s_segment_private;

  vector<lower = 0>[Company_N_c] consol_sigma; // 連結 ROIC 観測誤差
  matrix<lower = -3, upper = 3>[Company_N_c, Time_N]  consol_private;
  vector<lower = 0>[Company_N_c] s_consol_private;

  //------------------------------------------------------------------
  // Student-t の自由度
  //------------------------------------------------------------------
  real<lower = 1> nu_consol_roic;
  real<lower = 1> nu_seg_roic;
}

transformed parameters {
  //------------------------------------------------------------------
  // 各期の平均を除去した private 効果
  //------------------------------------------------------------------
  matrix[Segment_N,    Time_N] segment_private_eff;
  matrix[Company_N_c,  Time_N] consol_private_eff;
  for (t in 1:Time_N) {
    segment_private_eff[:, t] = segment_private[:, t] - mean(segment_private[:, t]);
    consol_private_eff[:,  t] =  consol_private[:, t] - mean(consol_private[:,  t]);
  }
}

model {
  //------------------------------------------------------------------
  // 分散パラメータの事前分布
  //------------------------------------------------------------------
  seg_sigma        ~ student_t(3, 0, 0.1);
  consol_sigma     ~ student_t(3, 0, 0.1);

  s_t              ~ student_t(3, 0, 0.5);
  s_segment_private~ student_t(3, 0, 0.5);
  s_consol_private ~ student_t(3, 0, 0.5);

  //------------------------------------------------------------------
  // 製品 ROIC の事前分布
  //------------------------------------------------------------------

  Item_ROIC[:, 1]      ~ normal(0.04, 0.5); // 製品ROICの1時点目の事前分布
  Item_ROIC[:, 2]      ~ normal(0.04, 0.5); // 製品ROICの2時点目の事前分布

  segment_private[:, 1] ~ student_t(3, 0, 0.5); // セグメント ROIC の1時点目の事前分布
  consol_private[:, 1]  ~ student_t(3, 0, 0.5);

  //------------------------------------------------------------------
  // 状態遷移（random-walk）
  //------------------------------------------------------------------
  for (t in 2:Time_N) {
    segment_private[:, t] ~ student_t(3, segment_private[:, t - 1],
                                      s_segment_private);
    consol_private[:,  t] ~ student_t(3,  consol_private[:,  t - 1],
                                      s_consol_private);
  }

  //------------------------------------------------------------------
  // 製品 ROIC の状態遷移: AR(2)
  //------------------------------------------------------------------
  for (t in 3:Time_N) {
    Item_ROIC[:, t] ~ normal( (2 * Item_ROIC[:, t - 1]) - Item_ROIC[:, t - 2], 
                          s_t);
  }

  //------------------------------------------------------------------
  // セグメント ROIC の尤度
  //------------------------------------------------------------------
  matrix[N_Share_index, Time_N] mu;
  for (t in 1:Time_N) {
    mu[, t] = Share[t] * Item_ROIC[, t]
            + segment_private_eff[segment_index_vec, t];
  }
  target += student_t_lpdf(
              Seg_ROIC | nu_seg_roic,
              to_vector(mu)[non_na_index],
              seg_sigma[segment_index]
            );

  //------------------------------------------------------------------
  // 連結 ROIC の尤度
  //------------------------------------------------------------------
  matrix[N_Share_consol_index, Time_N] mu_consol;
  for (t in 1:Time_N) {
    mu_consol[, t] = Share_consol[t] * Item_ROIC[, t]
                   + consol_private_eff[consol_index_vec, t];
  }
  target += student_t_lpdf(
              Consol_ROIC | nu_consol_roic,
              to_vector(mu_consol)[non_na_index_consol],
              consol_sigma[consol_index]
            );
}

