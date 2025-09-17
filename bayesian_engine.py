"""
統合ベイジアンエンジンモジュール

Stanモデルの管理とベイジアン推論の実行を統合したモジュール
"""

from pathlib import Path
from typing import Dict, Optional, Union, Any
import cmdstanpy
from cmdstanpy import CmdStanModel, CmdStanVB

from gppm.core.config_manager import get_logger


class BayesianEngine:
    """
    統合ベイジアンエンジンクラス
    
    Stanモデルの管理、コンパイル、推論実行を統合して提供
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
        self.fit_results = {}
        self.analysis_cache = {}
        self.logger = get_logger(__name__)
    
    
    def compile_model(self, model_path: Path, 
                     force_compile: bool = True) -> CmdStanModel:
        """
        Stanモデルをコンパイル
        
        Args:
            model_path: Stanファイルのパス（必須）
            force_compile: 強制再コンパイルフラグ
            
        Returns:
            コンパイルされたCmdStanModelオブジェクト
            
        Raises:
            FileNotFoundError: モデルファイルが見つからない場合
            RuntimeError: コンパイルに失敗した場合
        """
        # ファイルの存在確認
        if not model_path.exists():
            raise FileNotFoundError(f"Stanモデルファイルが見つかりません: {model_path}")
        
        model_name = model_path.stem
        
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
    
    def run_variational_inference(self, 
                                 model: CmdStanModel,
                                 data: Union[Dict, str],
                                 draws: int = 2000,
                                 iter: int = 500000,
                                 seed: int = 1234,
                                 show_console: bool = True,
                                 require_converged: bool = False,
                                 **kwargs) -> CmdStanVB:
        """
        変分推論を実行
        
        Args:
            model: コンパイルされたCmdStanModel
            data: Stan用データ（辞書またはJSONファイルパス）
            draws: サンプル数
            iter: 最適化反復数
            seed: 乱数シード
            show_console: コンソール出力の表示
            require_converged: 収束を要求するか
            **kwargs: その他のパラメータ
            
        Returns:
            変分推論結果（CmdStanVBオブジェクト）
        """
        self.logger.info(f"変分推論開始 - サンプル数: {draws}, 最適化反復数: {iter}")
        
        fit = model.variational(
            data=data,
            draws=draws,
            iter=iter,
            seed=seed,
            output_dir=str(self.result_dir),
            show_console=show_console,
            require_converged=require_converged,
            **kwargs
        )
        
        model_name = getattr(model, 'name', 'unknown_model')
        self.fit_results[f"{model_name}_vb"] = fit
        
        self.logger.info("変分推論完了")
        return fit
    
    def save_data_json(self, data: Dict, file_name: str) -> Path:
        """
        Stan用データをJSONファイルに保存
        
        Args:
            data: Stan用データ辞書
            file_name: ファイル名（拡張子なし）
            
        Returns:
            保存されたファイルのパス
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

