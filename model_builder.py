"""
ベイジアンモデルビルダーモジュール

Stanモデルの定義、コンパイル、管理を行う
"""

from pathlib import Path
from typing import Dict, Optional, Union

import cmdstanpy
from cmdstanpy import CmdStanModel

from gppm.core.config_manager import get_logger


class BayesianModelBuilder:
    """
    ベイジアンモデル（Stan）の管理クラス
    
    Stanモデルの定義，コンパイル，保存を行う
    outputディレクトリの下にdata, result, stanのディレクトリを管理
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Args:
            output_dir: 出力ディレクトリのパス（デフォルト: ./output）
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output")
        self.data_dir = self.output_dir / "data"
        self.result_dir = self.output_dir / "result"
        self.stan_dir = self.output_dir / "stan"
        
        # ディレクトリを作成
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.stan_dir.mkdir(parents=True, exist_ok=True)
        
        self.compiled_models = {}
        self.logger = get_logger(__name__)
        
    def get_global_ppm_roic_model_code(self) -> str:
        """
        グローバルPPMモデル(ROIC)のStanコードを取得

        Returns:
            Stanモデルコード
        """
        stan_code = """
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
"""
        return stan_code

    def create_model_file(self, model_name: str, stan_code: str) -> Path:
        """
        Stanモデルファイルを作成

        Args:
            model_name: モデル名（拡張子なし）
            stan_code: Stanコード文字列

        Returns:
            作成されたファイルのパス

        Raises:
            OSError: ファイル作成に失敗した場合
        """
        model_path = self.stan_dir / f"{model_name}.stan"

        with open(model_path, 'w', encoding='utf-8') as f:
            f.write(stan_code)

        self.logger.info(f"Stanモデルファイルを作成しました: {model_path}")
        return model_path

    def compile_model(self, model_name: str, stan_code: Optional[str] = None,
                      force_compile: bool = True) -> CmdStanModel:
        """
        Stanモデルをコンパイル

        Args:
            model_name: モデル名（拡張子なし）
            stan_code: Stanコード文字列 (Noneの場合は既存ファイルを使用)
            force_compile: 強制再コンパイルフラグ

        Returns:
            コンパイルされたCmdStanModelオブジェクト

        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
            RuntimeError: コンパイルに失敗した場合
        """
        # モデルファイルの作成または取得
        if stan_code:
            model_path = self.create_model_file(model_name, stan_code)
        else:
            model_path = self.stan_dir / f"{model_name}.stan"
            if not model_path.exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

        # モデルのコンパイル
        self.logger.info(f"Stanモデルをコンパイル中: {model_name}")
        try:
            model = CmdStanModel(stan_file=str(model_path), force_compile=force_compile)
            # キャッシュに保存
            self.compiled_models[model_name] = model
            self.logger.info(f"コンパイル完了: {model_name}")
            return model
        except Exception as e:
            self.logger.error(f"Stanモデルのコンパイルに失敗しました: {model_name}, エラー: {e}")
            raise

    def create_global_ppm_roic_model(self, model_name: str = None,
                                     force_compile: bool = True) -> CmdStanModel:
        """
        グローバルPPMモデル（ROIC）を作成・コンパイル

        Args:
            model_name: モデル名
            force_compile: 強制再コンパイル
        Returns:
            コンパイルされたモデル
        """
        stan_code = self.get_global_ppm_roic_model_code()
        return self.compile_model(model_name, stan_code, force_compile)


    def save_data_json(self, data: Dict, file_name: str) -> Path:
        """
        Stan用データをJSONファイルに保存

        Args:
            data: Stan用データ辞書
            file_name: ファイル名（拡張子なし）

        Returns:
            保存されたファイルのパス

        Raises:
            OSError: ファイル保存に失敗した場合
        """
        file_path = self.data_dir / f"{file_name}.json"
        cmdstanpy.write_stan_json(str(file_path), data)
        self.logger.info(f"Stan用データを保存しました: {file_path}")
        return file_path

    def save_result(self, result, file_name: str, file_format: str = "csv") -> Path:
        """
        推論結果を保存

        Args:
            result: 推論結果オブジェクト
            file_name: ファイル名（拡張子なし）
            file_format: ファイル形式（csv, json, pickle）

        Returns:
            保存されたファイルのパス

        Raises:
            OSError: ファイル保存に失敗した場合
        """
        file_path = self.result_dir / f"{file_name}.{file_format}"
        
        if file_format == "csv":
            result.to_csv(file_path, index=False)
        elif file_format == "json":
            result.to_json(file_path, orient="records", indent=2)
        elif file_format == "pickle":
            result.to_pickle(file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式: {file_format}")
        
        self.logger.info(f"推論結果を保存しました: {file_path}")
        return file_path

    def get_data_path(self, file_name: str) -> Path:
        """
        データディレクトリ内のファイルパスを取得

        Args:
            file_name: ファイル名

        Returns:
            ファイルのパス
        """
        return self.data_dir / file_name

    def get_result_path(self, file_name: str) -> Path:
        """
        結果ディレクトリ内のファイルパスを取得

        Args:
            file_name: ファイル名

        Returns:
            ファイルのパス
        """
        return self.result_dir / file_name

    def get_stan_path(self, file_name: str) -> Path:
        """
        Stanディレクトリ内のファイルパスを取得

        Args:
            file_name: ファイル名

        Returns:
            ファイルのパス
        """
        return self.stan_dir / file_name