#!/usr/bin/env python
"""
Mapping DataFrame生成CLI
======================

このモジュールは、FactSetデータを使用したマッピングDataFrame生成の
CLIインターフェースを提供します。

GPU環境での実行が必要で、埋め込みモデルを使用してセクターデータと
商品名データのマッピングを生成します。

使用方法:
    $ python -m gppm.cli.gen_tradename_mapping --output-path /path/to/mapping_df.pkl

必要な環境:
    - GPU環境（CUDA有効）
    - 埋め込みモデルファイル
    - データベース接続

設定例:
    python -m gppm.cli.gen_tradename_mapping \\
        --output-path /home/user/mapping_df.pkl \\
        --model-path /path/to/embedding/model \\
        --chunk-size 100000 \\
        --batch-size 10000
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from gppm.analysis.tradename_segment_mapper import TradenameSegmentMapper, TradenameSegmentMapperConfig
from gppm.core.config_manager import ConfigManager, get_logger

logger = get_logger(__name__)


def create_argument_parser() -> argparse.ArgumentParser:
    """コマンドライン引数パーサーの作成。

    Returns:
        設定済みのArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Mapping DataFrame生成CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
    # 基本的な使用方法（設定ファイルの値を使用）
    python -m gppm.cli.gen_tradename_mapping
    
    # 出力パスのみ指定
    python -m gppm.cli.gen_tradename_mapping --output-path /path/to/mapping_df.pkl
    
    # 詳細設定での使用方法
    python -m gppm.cli.gen_tradename_mapping \\
        --output-path /home/user/mapping_df.pkl \\
        --model-path /path/to/embedding/model \\
        --chunk-size 100000 \\
        --batch-size 10000 \\
        --max-items-per-entity 100
        """
    )

    # オプション引数（デフォルト値はNoneにして、後で設定ファイルから取得）
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="出力ファイルパス（.pkl形式）（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="埋め込みモデルのパス（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="ドキュメント分割サイズ（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="バッチサイズ（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--max-items-per-entity",
        type=int,
        default=None,
        help="企業あたりの最大商品数（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--initial-k",
        type=int,
        default=None,
        help="初期検索結果数（設定ファイルの値を使用する場合は省略）"
    )

    parser.add_argument(
        "--increment-factor",
        type=int,
        default=None,
        help="k増加倍率（設定ファイルの値を使用する場合は省略）"
    )

    return parser


def validate_arguments(args: argparse.Namespace) -> None:
    """引数の検証。

    Args:
        args: 解析された引数

    Raises:
        ValueError: 引数が不正な場合
        FileNotFoundError: 必要なファイルが見つからない場合
    """
    # 設定ファイルからデフォルト値を取得
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        if config.tradename_segment_mapper:
            default_output_path = config.tradename_segment_mapper.output.file_path
            default_model_path = config.tradename_segment_mapper.model_path
            default_chunk_size = config.tradename_segment_mapper.processing.chunk_size
            default_batch_size = config.tradename_segment_mapper.processing.batch_size
            default_max_items = config.tradename_segment_mapper.processing.max_items_per_entity
            default_initial_k = config.tradename_segment_mapper.search.initial_k
            default_increment_factor = config.tradename_segment_mapper.search.increment_factor
        else:
            default_output_path = "/tmp/mapping_df.pkl"
            default_model_path = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202411/BAAI-bge-m3-langchain"
            default_chunk_size = 100000
            default_batch_size = 10000
            default_max_items = 100
            default_initial_k = 100000
            default_increment_factor = 2
    except Exception:
        default_output_path = "/tmp/mapping_df.pkl"
        default_model_path = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202411/BAAI-bge-m3-langchain"
        default_chunk_size = 100000
        default_batch_size = 10000
        default_max_items = 100
        default_initial_k = 100000
        default_increment_factor = 2

    # 実際に使用される値を決定
    output_path = args.output_path if args.output_path is not None else default_output_path
    model_path = args.model_path if args.model_path is not None else default_model_path
    chunk_size = args.chunk_size if args.chunk_size is not None else default_chunk_size
    batch_size = args.batch_size if args.batch_size is not None else default_batch_size
    max_items = args.max_items_per_entity if args.max_items_per_entity is not None else default_max_items
    initial_k = args.initial_k if args.initial_k is not None else default_initial_k
    increment_factor = args.increment_factor if args.increment_factor is not None else default_increment_factor

    # 出力パスの検証
    output_path_obj = Path(output_path)
    if not output_path_obj.suffix == '.pkl':
        raise ValueError("出力ファイルは.pkl形式である必要があります")
    
    # 親ディレクトリの作成
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # モデルパスの検証
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"モデルパスが存在しません: {model_path}")

    # 数値引数の検証
    if chunk_size <= 0:
        raise ValueError("chunk-sizeは正の数である必要があります")
    
    if batch_size <= 0:
        raise ValueError("batch-sizeは正の数である必要があります")
    
    if max_items <= 0:
        raise ValueError("max-items-per-entityは正の数である必要があります")
    
    if initial_k <= 0:
        raise ValueError("initial-kは正の数である必要があります")
    
    if increment_factor <= 0:
        raise ValueError("increment-factorは正の数である必要があります")


def create_mapping_config(args: argparse.Namespace) -> TradenameSegmentMapperConfig:
    """引数と設定ファイルからTradenameSegmentMapperConfigを作成。

    Args:
        args: 解析された引数

    Returns:
        作成されたTradenameSegmentMapperConfig
    """
    # 設定ファイルからデフォルト値を取得
    try:
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        if config.tradename_segment_mapper:
            # 設定ファイルの値を使用
            default_output_path = config.tradename_segment_mapper.output.file_path
            default_model_path = config.tradename_segment_mapper.model_path
            default_chunk_size = config.tradename_segment_mapper.processing.chunk_size
            default_batch_size = config.tradename_segment_mapper.processing.batch_size
            default_max_items = config.tradename_segment_mapper.processing.max_items_per_entity
            default_initial_k = config.tradename_segment_mapper.search.initial_k
            default_increment_factor = config.tradename_segment_mapper.search.increment_factor
        else:
            # 設定ファイルにtradename_segment_mapperがない場合はデフォルト値を使用
            default_output_path = "/tmp/mapping_df.pkl"
            default_model_path = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202411/BAAI-bge-m3-langchain"
            default_chunk_size = 100000
            default_batch_size = 10000
            default_max_items = 100
            default_initial_k = 100000
            default_increment_factor = 2
    except Exception as e:
        logger.warning(f"設定ファイルの読み込みに失敗: {e}")
        # 設定ファイルの読み込みに失敗した場合はデフォルト値を使用
        default_output_path = "/tmp/mapping_df.pkl"
        default_model_path = "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202411/BAAI-bge-m3-langchain"
        default_chunk_size = 100000
        default_batch_size = 10000
        default_max_items = 100
        default_initial_k = 100000
        default_increment_factor = 2

    # コマンドライン引数が指定されている場合はそれを使用、そうでなければ設定ファイルの値を使用
    return TradenameSegmentMapperConfig(
        model_path=args.model_path if args.model_path is not None else default_model_path,
        chunk_size=args.chunk_size if args.chunk_size is not None else default_chunk_size,
        max_items_per_entity=args.max_items_per_entity if args.max_items_per_entity is not None else default_max_items,
        initial_k=args.initial_k if args.initial_k is not None else default_initial_k,
        increment_factor=args.increment_factor if args.increment_factor is not None else default_increment_factor,
        batch_size=args.batch_size if args.batch_size is not None else default_batch_size,
        output_path=args.output_path if args.output_path is not None else default_output_path,
    )


def main() -> int:
    """メイン関数。

    Returns:
        終了コード（0: 成功, 1: エラー）
    """
    try:
        # 引数の解析
        parser = create_argument_parser()
        args = parser.parse_args()

        # 引数の検証
        validate_arguments(args)

        logger.info("Tradename Segment Mapping生成開始")
        
        # 設定の作成
        config = create_mapping_config(args)
        
        # 実際に使用される値をログ出力
        logger.info(f"出力パス: {config.output_path}")
        logger.info(f"モデルパス: {config.model_path}")
        logger.info(f"チャンクサイズ: {config.chunk_size}")
        logger.info(f"バッチサイズ: {config.batch_size}")
        logger.info(f"最大商品数: {config.max_items_per_entity}")
        logger.info(f"初期K: {config.initial_k}")
        logger.info(f"増加倍率: {config.increment_factor}")

        # TradenameSegmentMapperの実行
        mapper = TradenameSegmentMapper(config)
        result = mapper.generate_mapping_df()

        # 結果の表示
        print("\n" + "="*60)
        print("Tradename Segment Mapping生成完了")
        print("="*60)
        print(f"生成日時: {result.generated_at}")
        print(f"総マッピング数: {result.total_mappings:,}")
        print(f"総企業数: {result.total_entities:,}")
        print(f"成功率: {result.success_rate:.2%}")
        print(f"セクターレコード数: {result.sector_records_count:,}")
        print(f"商品名レコード数: {result.tradename_records_count:,}")
        print(f"出力ファイル: {result.output_path}")
        print(f"ファイルサイズ: {result.file_size_mb:.2f} MB")
        print("="*60)

        logger.info("Tradename Segment Mapping生成完了")
        return 0

    except KeyboardInterrupt:
        logger.warning("処理が中断されました")
        return 1
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
