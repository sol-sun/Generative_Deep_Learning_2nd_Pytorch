#!/usr/bin/env python
"""
変分推論実行CLI

このモジュールは、データ前処理で生成したStan入力用データ（pickle）を読み込み、
変分推論によるベイズ分析を実行するためのCLIです。

主な機能:
- 入力データ（prepare_data.pyで保存したpickle）の検証
- データの形状と内容のサマリ出力
- 変分推論の実行（CmdStanPy/PyStan/Numpyro対応）
- 実行時オプション（推論エンジン、サンプル数、最適化反復数、乱数シード、モデルパス、出力先）

使用例:
    # 設定ファイルのデフォルト値を使用
    $ python -m gppm.cli.bayes_infer
    
    # 一部のパラメータを指定
    $ python -m gppm.cli.bayes_infer \
        --data "/tmp/gppm_output/processed_data/bayesian_dataset.pkl" \
        --engine cmdstanpy \
        --model /path/to/model.stan \
        --draws 2000 --iter 500000 --seed 123 \
        --output "/tmp/gppm_output/inference"

注意:
    - 現在は基本的な検証とメタ情報の出力を行います
    - 実際の変分推論実行は、指定されたエンジンに応じて実装予定です
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict

from gppm.core.config_manager import ConfigManager, get_logger


def _load_processed(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"前処理データが見つかりません: {path}")
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError("pickleの内容が辞書ではありません。生成処理を確認してください。")
    # 必須キーの存在チェック
    required = ["stan_data", "processing_info"]
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"必須キーが不足しています: {missing}")
    return obj


def _summarize(processed: Dict[str, Any]) -> Dict[str, Any]:
    stan = processed.get("stan_data", {})
    info = processed.get("processing_info", {})
    summary = {
        "Segment_N": stan.get("Segment_N"),
        "Company_N": stan.get("Company_N"),
        "Company_N_c": stan.get("Company_N_c"),
        "Product_N": stan.get("Product_N"),
        "Time_N": stan.get("Time_N"),
        "N_obs_segment": stan.get("N_obs"),
        "N_obs_consol": stan.get("N_obs_consol"),
        "N_obs_wacc": stan.get("N_obs_wacc"),
        "period": {
            "start": info.get("start_period"),
            "end": info.get("end_period"),
        },
    }
    return summary


def _ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_variational_inference(
    *,
    processed: Dict[str, Any],
    engine: str,
    model_path: Path | None,
    draws: int,
    variational_iter: int,
    seed: int,
    outdir: Path,
) -> Dict[str, Any]:
    """
    変分推論の実行。

    指定されたエンジンに応じて変分推論を実行します。
    現在は基本的な検証とメタ情報の出力を行います。
    """
    _ensure_outdir(outdir)
    summary = _summarize(processed)

    meta = {
        "status": "validation_completed",
        "message": "データ検証完了。変分推論の実装は進行中です。",
        "engine": engine,
        "model_path": str(model_path) if model_path else None,
        "variational_draws": draws,
        "variational_iter": variational_iter,
        "seed": seed,
        "data_summary": summary,
    }

    # 出力にメタ情報を保存
    meta_path = outdir / "inference_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def parse_args(argv: list[str]) -> argparse.Namespace:
    # 設定を読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 設定の検証
    validation_results = config_manager.validate_config(config)
    if not all(validation_results.values()):
        logger = get_logger(__name__)
        logger.warning(f"設定検証で警告があります: {validation_results}")
    
    # 設定ファイルから変分推論のデフォルト値を取得
    variational_config = config.variational_inference
    new_inference_config = variational_config.new_inference
    
    # デフォルト値の設定
    default_engine = new_inference_config.engine
    default_draws = new_inference_config.samples
    default_iter = new_inference_config.optimization_iterations
    default_seed = new_inference_config.random_seed
    default_output = config.output.directory
    
    p = argparse.ArgumentParser(
        prog="gppm.bayes_infer",
        description="Global PPM: 変分推論実行CLI",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=False,
        help="前処理済みデータファイル（pickle）のパス（未指定時は設定ファイルから自動取得）",
    )
    p.add_argument(
        "--engine",
        type=str,
        default=default_engine,
        choices=["cmdstanpy", "pystan", "numpyro"],
        help="変分推論エンジンの種類",
    )
    p.add_argument(
        "--model",
        type=Path,
        help="Stanモデルファイル（.stan）のパス",
    )
    p.add_argument("--draws", type=int, default=default_draws, help="変分推論で生成するサンプル数")
    p.add_argument("--iter", dest="variational_iter", type=int, default=default_iter, help="変分推論の最適化反復数")
    p.add_argument("--seed", type=int, default=default_seed, help="乱数シード（再現性のため）")
    p.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="変分推論結果の出力ディレクトリ",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    # ロガーの初期化
    logger = get_logger(__name__)
    
    try:
        ns = parse_args(argv or sys.argv[1:])
        
        # 設定を取得
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # 入力データパスの解決: 明示指定 > 設定の推定 > エラー
        data_path = ns.data
        if data_path is None:
            # 設定ファイルからデータパスを推定
            data_path = config.get_output_path()
            logger.info(f"データパスを設定から推定: {data_path}")

        logger.info(f"データファイル読み込み開始: {data_path}")
        processed = _load_processed(data_path)
        logger.info("データファイル読み込み完了")

        # モデルパスの任意性について: 現在は基本的な検証のみ実行
        if ns.model is None:
            logger.info("--model 未指定: 基本的なデータ検証とメタ情報出力を実行します。")

        logger.info(f"変分推論開始 - エンジン: {ns.engine}, サンプル数: {ns.draws}, 最適化反復数: {ns.variational_iter}")
        meta = run_variational_inference(
            processed=processed,
            engine=ns.engine,
            model_path=ns.model,
            draws=ns.draws,
            variational_iter=ns.variational_iter,
            seed=ns.seed,
            outdir=ns.output,
        )
        logger.info("変分推論処理完了")

        # コンソールにサマリを出力
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return meta
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        raise


if __name__ == "__main__":
    main()

