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
    data:
      start_period: 201909
      end_period: 202406
    output:
      base_directory: "/tmp/gppm_output"
      dataset_filename: "bayesian_dataset.pkl"
    mapping_df_path: null
"""

import yaml
from pathlib import Path
from gppm.analysis.bayesian.data_processor import BayesianDataProcessor
from gppm.core.data_manager import FactSetDataManager


def load_config(config_path: str = "gppm_config.yml") -> dict:
    """
    設定ファイルを読み込む
    
    Args:
        config_path: 設定ファイルのパス（デフォルト: "gppm_config.yml"）
        
    Returns:
        設定辞書
        
    Raises:
        FileNotFoundError: 設定ファイルが見つからない場合
        
    Examples:
        >>> # デフォルトの設定ファイルを読み込み
        >>> config = load_config()
        >>> print(config['data']['start_period'])
        201909
        
        >>> # カスタム設定ファイルを指定
        >>> config = load_config("custom_config.yml")
    """
    config_file = Path(__file__).parents[3] / config_path
    if not config_file.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {config_file}")
    with open(config_file, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    """
    ベイジアンデータ処理のメイン関数
    
    設定ファイル（gppm_config.yml）を読み込み、ベイジアンモデリング用の
    データ前処理パイプラインを実行します。
    
    Returns:
        処理済みデータ辞書
        
    Raises:
        ValueError: 必須パラメータが設定ファイルに存在しない場合
        FileNotFoundError: 設定ファイルが見つからない場合
        
    Examples:
        >>> # CLIから実行
        >>> python -m gppm.cli.prepare_data
        
        >>> # プログラムから実行
        >>> from gppm.cli.prepare_data import main
        >>> processed_data = main()
        >>> print(f"処理済みエンティティ数: {len(processed_data['financial_df'])}")
        
        配置すべき設定ファイル (gppm_config.yml):
        ```yaml
        data:
          start_period: 201909
          end_period: 202406
        output:
          base_directory: "/tmp/gppm_output"
          dataset_filename: "bayesian_dataset.pkl"
        mapping_df_path: null  # オプション
        ```
    """
    config = load_config()
    
    # データ設定の確認
    if 'data' not in config:
        raise ValueError("設定ファイルに 'data' セクションが存在しません")
    
    data_config = config['data']
    required_data_params = ['start_period', 'end_period']
    for param in required_data_params:
        if param not in data_config:
            raise ValueError(f"データ設定の必須パラメータ '{param}' が存在しません")
    
    # 出力設定の確認
    if 'output' not in config:
        raise ValueError("設定ファイルに 'output' セクションが存在しません")
    
    output_config = config['output']
    required_output_params = ['base_directory', 'dataset_filename']
    for param in required_output_params:
        if param not in output_config:
            raise ValueError(f"出力設定の必須パラメータ '{param}' が存在しません")
    
    # 保存パスの設定
    save_path = Path(output_config['base_directory']) / 'processed_data' / output_config['dataset_filename']
    
    # データ処理の実行
    data_manager = FactSetDataManager()
    mapping_df_path = config.get('mapping_df_path')
    bayesian_processor = BayesianDataProcessor(mapping_df_path=mapping_df_path)
    
    processed_data = bayesian_processor.process_full_pipeline(
        data_manager=data_manager,
        save_path=str(save_path),
        start_period=data_config['start_period'],
        end_period=data_config['end_period']
    )
    return processed_data


if __name__ == "__main__":
    main()


