"""ベイジアン推論実行モジュール"""

from pathlib import Path
from typing import Dict, Optional, Union
import arviz as az
from cmdstanpy import CmdStanModel, CmdStanVB, CmdStanMCMC


class BayesianInference:
    """ベイジアン推論実行・結果管理クラス"""
    
    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./bayesian_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fit_results = {}
        self.analysis_cache = {}
    
    def run_variational_inference(self, 
                                 model: CmdStanModel,
                                 data: Union[Dict, str],
                                 draws: int = 2000,
                                 iter: int = 500000,
                                 seed: int = 1234,
                                 show_console: bool = True,
                                 require_converged: bool = False,
                                 **kwargs) -> CmdStanVB:
        fit = model.variational(
            data=data,
            draws=draws,
            iter=iter,
            seed=seed,
            output_dir=str(self.output_dir),
            show_console=show_console,
            require_converged=require_converged,
            **kwargs
        )
        model_name = getattr(model, 'name', 'unknown_model')
        self.fit_results[f"{model_name}_vb"] = fit
        return fit


