#!/usr/bin/env python
"""
ベイジアンデータ処理 実行CLI

このモジュールは、FactSetデータを使用したベイジアンモデリング用の
データ前処理パイプラインを実行するCLIインターフェースを提供します。

設定ファイル（gppm_config.yml）を読み込み、指定された期間のデータを
処理してモデリング用のデータセットを生成します。

使用方法:
    $ python -m gppm.cli.prepare_data

必要な設定ファイル:
    プロジェクトルートに gppm_config.yml ファイルを配置してください。
    
設定例:
    analysis_period:
      start: 201909
      end: 202406
    output:
      directory: "/tmp/gppm_output"
      files:
        dataset: "dataset.pkl"
    optional_data:
      mapping_file: "/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202501/mapping_df.pkl"

Attributes:
    DEFAULT_MAPPING_PATH (str): デフォルトのマッピングファイルパス
    
Examples:
    >>> # CLIから実行
    >>> python -m gppm.cli.prepare_data
    
    >>> # プログラムから実行
    >>> from gppm.cli.prepare_data import main
    >>> processed_data = main()
    >>> print(f"処理済みエンティティ数: {len(processed_data['raw_data']['consol'])}")
"""

from pathlib import Path
from typing import Dict, Any, Optional

from gppm.analysis.bayesian.data_processor import BayesianDataProcessor
from gppm.core.data_manager import FactSetDataManager
from gppm.core.config_manager import ConfigManager


def main() -> Dict[str, Any]:
    """
    ベイジアンデータ処理のメイン関数
    
    ConfigManagerを使用して設定を読み込み、ベイジアンモデリング用の
    データ前処理パイプラインを実行します。
    
    Returns:
        Dict[str, Any]: 処理済みデータ辞書。以下のキーを含む:
            - raw_data: 生データ（セグメント、連結、製品シェアデータ）
            - pivot_tables: ピボットテーブル辞書
            - stan_data: Stan用データ構造
            - product_names: 製品名リスト
            - entity_info: エンティティ情報
            - processing_info: 処理情報（期間、件数など）
        
    Raises:
        ValueError: 必須パラメータが設定ファイルに存在しない場合
        FileNotFoundError: 設定ファイルが見つからない場合
        KeyError: 設定ファイルの構造が期待される形式でない場合
        
    Examples:
        >>> # CLIから実行
        >>> python -m gppm.cli.prepare_data
        
        >>> # プログラムから実行
        >>> from gppm.cli.prepare_data import main
        >>> processed_data = main()
        >>> print(f"処理済みエンティティ数: {len(processed_data['raw_data']['consol'])}")
        >>> print(f"製品数: {processed_data['processing_info']['n_products']}")
    """
    # 設定管理の初期化
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # 設定の検証と警告表示
    validation_results = config_manager.validate_config(config)
    if not all(validation_results.values()):
        print(f"設定検証で警告があります: {validation_results}")
    
    # 出力パスの構築
    save_path = Path(config.output.directory) / config.output.files.dataset
    
    # データ処理コンポーネントの初期化
    data_manager = FactSetDataManager()
    mapping_df_path = getattr(config.optional_data, 'mapping_df_path', None)
    bayesian_processor = BayesianDataProcessor(mapping_df_path=mapping_df_path)
    
    # データ処理パイプラインの実行
    processed_data = bayesian_processor.process_full_pipeline(
        data_manager=data_manager,
        start_period=config.analysis_period.start,
        end_period=config.analysis_period.end,
        save_path=str(save_path)
    )
    
    return processed_data


if __name__ == "__main__":
    """
    CLIエントリーポイント
    
    このスクリプトが直接実行された場合に、
    ベイジアンデータ処理パイプラインを実行します。
    """
    try:
        result = main()
        print("データ処理が正常に完了しました。")
        print(f"処理結果: {result['processing_info']}")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        raise


