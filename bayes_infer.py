#!/usr/bin/env python
"""
ベイズ推論 実行CLI（スキャフォールド）

このモジュールは、データ前処理で生成したStan入力用データ（pickle）を読み込み、
将来的なベイズ推論（例: CmdStanPy/PyStan/Numpyro）を実行するためのCLIの骨組みです。

現時点では、以下を提供します:
- 入力データ（prepare_data.pyで保存したpickle）の検証
- 事前条件・形状のサマリ出力
- 実行時オプション（推論エンジン、反復数、チェーン数、乱数シード、モデルパス、出力先）

使用例:
    $ python -m gppm.cli.bayes_infer \
        --data "/tmp/gppm_output/processed_data/bayesian_dataset.pkl" \
        --engine cmdstanpy \
        --model /path/to/model.stan \
        --iter 1000 --chains 4 --seed 42 \
        --output "/tmp/gppm_output/inference"

注意:
    - 本CLIはスキャフォールドです。実際のサンプリング呼び出しは未実装です。
    - 今後、指定エンジン（cmdstanpy/pystan/numpyro等）に応じた実装を追加予定です。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Dict


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


def run_inference_stub(
    *,
    processed: Dict[str, Any],
    engine: str,
    model_path: Path | None,
    iter_sampling: int,
    chains: int,
    seed: int,
    outdir: Path,
) -> Dict[str, Any]:
    """
    ベイズ推論のスタブ実装。

    将来的にエンジン別の実行ブロックを追加します。
    現状はサマリをJSONで保存し、未実装ステータスを返します。
    """
    _ensure_outdir(outdir)
    summary = _summarize(processed)

    meta = {
        "status": "not_implemented",
        "engine": engine,
        "model_path": str(model_path) if model_path else None,
        "iter_sampling": iter_sampling,
        "chains": chains,
        "seed": seed,
        "data_summary": summary,
    }

    # 出力にメタ情報を保存
    meta_path = outdir / "inference_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="gppm.bayes_infer",
        description="Global PPM: ベイズ推論CLI（スキャフォールド）",
    )
    p.add_argument(
        "--data",
        type=Path,
        required=False,
        help="前処理pickleへのパス（未指定時は設定のデフォルト推定を使用）",
    )
    p.add_argument(
        "--engine",
        type=str,
        default="cmdstanpy",
        choices=["cmdstanpy", "pystan", "numpyro"],
        help="推論エンジンの種類（将来対応予定）",
    )
    p.add_argument(
        "--model",
        type=Path,
        help="Stanモデルファイルへのパス（.stan）",
    )
    p.add_argument("--iter", dest="iter_sampling", type=int, default=1000, help="サンプリング反復数")
    p.add_argument("--chains", type=int, default=4, help="チェーン数")
    p.add_argument("--seed", type=int, default=42, help="乱数シード")
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/gppm_output/inference"),
        help="推論結果の出力ディレクトリ",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> Dict[str, Any]:
    ns = parse_args(argv or sys.argv[1:])

    # 入力データパスの解決: 明示指定 > 設定の推定 > エラー
    data_path = ns.data
    if data_path is None:
        # prepare_data.py のデフォルト保存先に合わせて推定
        data_path = Path("/tmp/gppm_output/processed_data/bayesian_dataset.pkl")

    processed = _load_processed(data_path)

    # モデルパスの任意性について: 現状は未実装のため必須ではない
    if ns.model is None:
        print("[info] --model 未指定: 現状は検証とメタ出力のみを実行します。")

    meta = run_inference_stub(
        processed=processed,
        engine=ns.engine,
        model_path=ns.model,
        iter_sampling=ns.iter_sampling,
        chains=ns.chains,
        seed=ns.seed,
        outdir=ns.output,
    )

    # コンソールにサマリを出力
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    return meta


if __name__ == "__main__":
    main()

