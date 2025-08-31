"""
GPPM設定管理モジュール
====================================================

目的
- YAML設定ファイルと環境変数による統合設定管理
- スレッドセーフなシングルトンパターンによる設定アクセス
- 設定の検証・ログ設定・パス管理の自動化

主要コンポーネント
- `GPPMConfig`: メイン設定クラス（Pydantic Settings対応）
- `ConfigManager`: スレッドセーフな設定管理クラス（シングルトン）
- `ProcessingConfig`: 処理パフォーマンス設定
- `DataConfig`: データ期間・範囲設定
- `OutputConfig`: 出力パス・ファイル設定

設定の優先順位（高→低）
- 環境変数（GPPM_で始まる）
- YAML設定ファイル（gppm_config.yml）
- デフォルト値

パフォーマンス設計（要点）
- シングルトンパターン: 設定インスタンスの重複作成を防止
- スレッドセーフ: ロックによる並行アクセス制御
- 遅延初期化: 初回アクセス時のみ設定読み込み
- キャッシュ: 設定インスタンスの再利用

使用例
    from gppm.utils.config_manager import ConfigManager, get_logger

    # 設定マネージャーの初期化
    config_manager = ConfigManager()
    config = config_manager.get_config()

    # 設定値の取得
    batch_size = config.processing.batch_size
    start_period = config.data.start_period
    output_dir = config.output.base_directory

    # ロガーの取得
    logger = get_logger(__name__)
    logger.info("設定読み込み完了: batch_size=%d", batch_size)

    # 設定の検証
    validation_results = config_manager.validate_config()
    if not all(validation_results.values()):
        logger.warning("設定検証エラー: %s", validation_results)

設定ファイル例（gppm_config.yml）
    log_level: INFO
    log_file: /tmp/gppm.log
    
    processing:
      batch_size: 1000
      max_workers: 4
    
    data:
      start_period: 201909
      end_period: 202406
    
    output:
      base_directory: /tmp/gppm_output
      dataset_filename: bayesian_dataset.pkl

    mapping_df_path: /data/mapping.pkl
"""

import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Type, Dict, Any, Self

from pydantic import BaseModel, Field, field_validator, ValidationError
from pydantic_settings import (
    BaseSettings, 
    SettingsConfigDict,
    YamlConfigSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource
)


class ProcessingConfig(BaseModel):
    """処理パフォーマンス設定。

    概要:
    - バッチ処理サイズと並列処理ワーカー数を管理します。
    - データ処理の効率性とリソース使用量のバランスを制御します。

    特徴:
    - バッチサイズによるメモリ使用量の最適化
    - 並列処理による計算速度の向上
    - リソース制約に基づく安全なデフォルト値
    """
    batch_size: int = Field(
        default=1000, 
        gt=0, 
        description="バッチサイズ（メモリ使用量と処理速度のバランス）",
        examples=[100, 1000, 5000]
    )
    max_workers: int = Field(
        default=4, 
        ge=1, 
        le=32, 
        description="最大ワーカー数（並列処理の同時実行数）",
        examples=[1, 4, 8, 16]
    )


class DataConfig(BaseModel):
    """データ期間・範囲設定。

    概要:
    - データ処理対象の期間範囲を管理します。
    - YYYYMM形式での期間指定と妥当性検証を提供します。

    特徴:
    - 開始期間と終了期間の自動検証
    - 2000年以降の期間制約
    - 期間の論理的一貫性チェック
    """
    start_period: int = Field(
        default=201909, 
        ge=200001, 
        le=999912, 
        description="開始期間（YYYYMM形式、2000年以降）",
        examples=[201909, 202001, 202406]
    )
    end_period: int = Field(
        default=202406, 
        ge=200001, 
        le=999912, 
        description="終了期間（YYYYMM形式、2000年以降）",
        examples=[202001, 202406, 202412]
    )
    
    @field_validator('end_period')
    @classmethod
    def validate_end_after_start(cls, v: int, info) -> int:
        """終了期間が開始期間より後であることを検証。

        Args:
            v: 検証対象の終了期間値
            info: バリデーション情報（開始期間を含む）

        Returns:
            検証済みの終了期間値

        Raises:
            ValueError: 終了期間が開始期間以前の場合
        """
        if 'start_period' in info.data and v < info.data['start_period']:
            raise ValueError('終了期間は開始期間より後でなければなりません。')
        return v


class OutputConfig(BaseModel):
    """出力パス・ファイル設定。

    概要:
    - 処理結果の出力先ディレクトリとファイル名を管理します。
    - パスオブジェクトの自動変換とディレクトリ作成を提供します。

    特徴:
    - 文字列からPathオブジェクトへの自動変換
    - 出力ディレクトリの存在確認
    - デフォルト出力先の安全な設定
    """
    base_directory: Path = Field(
        default=Path("/tmp/gppm_output"), 
        description="出力ベースディレクトリ（処理結果の保存先）",
        examples=["/tmp/gppm_output", "./output", "/data/gppm"]
    )
    dataset_filename: str = Field(
        default="bayesian_dataset.pkl", 
        description="データセットファイル名（出力ファイルの名前）",
        examples=["bayesian_dataset.pkl", "processed_data.pkl"]
    )
    
    @field_validator('base_directory', mode='before')
    @classmethod
    def convert_to_path(cls, v) -> Path:
        """文字列をPathオブジェクトに変換。

        Args:
            v: 変換対象の値（文字列またはPathオブジェクト）

        Returns:
            変換後のPathオブジェクト
        """
        return Path(v) if isinstance(v, str) else v


class GPPMConfig(BaseSettings):
    """
    GPPM メイン設定クラス（Pydantic Settings対応）。

    概要:
    - アプリケーション全体の設定を統合管理します。
    - 環境変数、YAMLファイル、デフォルト値の優先順位制御を提供します。
    - ログ設定、パス検証、設定の自動検証機能を含みます。

    設定の優先順位（高→低）:
    1. 環境変数 (GPPM_で始まる)
    2. YAML設定ファイル (gppm_config.yml)
    3. デフォルト値

    主要機能:
    - 設定ソースのカスタマイズ（YAMLファイル自動検出）
    - ログ設定の自動適用
    - パス存在の検証
    - 設定値の型安全性確保

    例外処理:
    - FileNotFoundError: 設定ファイルが見つからない場合
    - ValidationError: 設定値の検証に失敗した場合
    - ValueError: 設定ファイルの読み込みエラー
    """
    
    model_config = SettingsConfigDict(
        env_prefix="GPPM_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",  # 未知のフィールドを無視
        validate_default=True,
    )
    
    # ログ設定
    log_level: str = Field(
        default="INFO", 
        description="ログレベル（DEBUG/INFO/WARNING/ERROR/CRITICAL）",
        examples=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="ログメッセージのフォーマット"
    )
    log_file: Optional[Path] = Field(
        default=None, 
        description="ログファイルパス（指定時はファイル出力を有効化）",
        examples=["/tmp/gppm.log", "./logs/gppm.log"]
    )
    
    # マッピングデータファイルパス
    mapping_df_path: Optional[Path] = Field(
        default=None, 
        description="マッピングデータファイルパス（オプショナル）",
        examples=["/path/to/mapping_df.pkl", "./data/mapping.pkl"]
    )
    
    # サブ設定
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """ログレベルの検証。

        Args:
            v: 検証対象のログレベル文字列

        Returns:
            正規化されたログレベル（大文字）

        Raises:
            ValueError: 無効なログレベルの場合
        """
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f'log_level must be one of {valid_levels}')
        return v.upper()
    
    @field_validator('log_file', mode='before')
    @classmethod
    def convert_log_file_to_path(cls, v) -> Optional[Path]:
        """ログファイルパスをPathオブジェクトに変換。

        Args:
            v: 変換対象の値（文字列またはPathオブジェクト）

        Returns:
            変換後のPathオブジェクト（Noneの場合はNone）
        """
        return Path(v) if isinstance(v, str) and v else v
    
    @field_validator('mapping_df_path', mode='before')
    @classmethod
    def convert_mapping_path_to_path(cls, v) -> Optional[Path]:
        """マッピングファイルパスをPathオブジェクトに変換。

        Args:
            v: 変換対象の値（文字列またはPathオブジェクト）

        Returns:
            変換後のPathオブジェクト（Noneの場合はNone）
        """
        return Path(v) if isinstance(v, str) and v else v
    
    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        """設定ソースのカスタマイズ（YAMLファイル自動検出）。

        概要:
        - プロジェクトルートのgppm_config.ymlを自動検出
        - 設定ソースの優先順位を定義
        - YAMLファイルの読み込みテストを実行

        Args:
            settings_cls: 設定クラス
            init_settings: 初期化設定ソース
            env_settings: 環境変数設定ソース
            dotenv_settings: .envファイル設定ソース
            file_secret_settings: シークレットファイル設定ソース

        Returns:
            設定ソースのタプル（優先順位順）

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValueError: 設定ファイルの読み込みエラー
        """
        # プロジェクトルートディレクトリを検出
        # このファイルから見たプロジェクトルート: ../../../../
        current_file = Path(__file__)
        project_root = current_file.parent.parent.parent.parent
        yaml_file = project_root / "gppm_config.yml"
        
        if not yaml_file.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {yaml_file.absolute()}")
        
        try:
            yaml_source = YamlConfigSettingsSource(settings_cls, yaml_file)
            # YAML読み込みテスト
            yaml_source._read_files(yaml_file)
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みエラー: {yaml_file} - {e}")
        
        return (
            init_settings,  # 初期化時の引数
            env_settings,   # 環境変数
            yaml_source,    # YAML設定ファイル
            file_secret_settings,  # シークレットファイル
        )
    
    def get_output_path(self, filename: Optional[str] = None) -> Path:
        """出力パスを取得。

        Args:
            filename: ファイル名（指定時はそのファイルのパス、None時はデフォルトファイル名）

        Returns:
            出力ファイルの完全パス
        """
        if filename:
            return self.output.base_directory / filename
        return self.output.base_directory / self.output.dataset_filename
    
    def validate_paths(self) -> Dict[str, bool]:
        """パスの存在を検証。

        概要:
        - 出力ディレクトリ、マッピングファイル、ログファイルの存在確認
        - ディレクトリ作成可能性の検証

        Returns:
            各パスの検証結果辞書（キー: パス種別、値: 存在フラグ）
        """
        results = {}
        
        # 出力ディレクトリの確認
        results['output_directory'] = self.output.base_directory.exists() or self.output.base_directory.parent.exists()
        
        # マッピングファイルの確認
        if self.mapping_df_path:
            results['mapping_file'] = self.mapping_df_path.exists()
        else:
            results['mapping_file'] = True  # オプショナルなので存在しなくてもOK
        
        # ログファイルディレクトリの確認
        if self.log_file:
            results['log_directory'] = self.log_file.parent.exists() or self.log_file.parent.parent.exists()
        else:
            results['log_directory'] = True
        
        return results
    
    def setup_logging(self) -> None:
        """ログ設定を適用。

        概要:
        - gppmロガーの設定を適用
        - コンソールとファイルの両方にログ出力
        - 他のライブラリのログレベルを制御

        特徴:
        - フォーマッターによる統一されたログ形式
        - ファイルハンドラーの自動ディレクトリ作成
        - ログレベルの動的設定
        """
        level = getattr(logging, self.log_level, logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            fmt=self.log_format,
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # gppmロガー設定
        logger = logging.getLogger("gppm")
        logger.setLevel(level)
        logger.handlers.clear()
        
        # コンソールハンドラー
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # ファイルハンドラー（指定時）
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(
                self.log_file, mode="a", encoding="utf-8"
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # 他のライブラリのログレベルを制御
        logging.getLogger().setLevel(logging.WARNING)
        logger.propagate = False
    
    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """設定を辞書形式で出力。

        Args:
            **kwargs: Pydanticのmodel_dumpに渡す引数

        Returns:
            設定の辞書表現
        """
        return super().model_dump(**kwargs)


class ConfigManager:
    """
    スレッドセーフな設定管理クラス（シングルトンパターン）。

    概要:
    - アプリケーション全体で一つの設定インスタンスを共有
    - スレッドセーフな設定アクセスを提供
    - 設定の遅延初期化とキャッシュ機能

    主要機能:
    - シングルトンパターンによる設定インスタンス管理
    - スレッドセーフな並行アクセス制御
    - 設定の検証とリセット機能
    - 設定インスタンスの遅延初期化

    パフォーマンス最適化:
    - 設定インスタンスの重複作成を防止
    - ロックによる並行アクセス制御
    - 初回アクセス時のみ設定読み込み

    例外処理:
    - FileNotFoundError: 設定ファイルが見つからない場合
    - ValidationError: 設定値の検証に失敗した場合
    - ValueError: 設定作成時のエラー
    """
    
    _instance: Optional[Self] = None
    _lock: threading.Lock = threading.Lock()
    _initialized: bool = False
    
    def __new__(cls) -> Self:
        """スレッドセーフなシングルトン実装。

        Returns:
            設定マネージャーのインスタンス（常に同じインスタンス）
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        """初期化（一度のみ実行）。

        概要:
        - 設定インスタンスの遅延初期化
        - スレッドセーフな初期化制御
        """
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._config: Optional[GPPMConfig] = None
                    self._initialized = True
    
    def get_config(self) -> GPPMConfig:
        """設定インスタンスを取得（遅延初期化）。

        概要:
        - 初回呼び出し時に設定インスタンスを作成
        - 以降の呼び出しではキャッシュされたインスタンスを返却

        Returns:
            設定済みのGPPMConfigインスタンス

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValidationError: 設定値の検証に失敗した場合
            ValueError: 設定作成時のエラー
        """
        if self._config is None:
            with self._lock:
                if self._config is None:
                    self._config = self._create_config()
        return self._config
    
    def reset_config(self) -> None:
        """設定インスタンスをリセット。

        概要:
        - キャッシュされた設定インスタンスをクリア
        - 次回get_config()呼び出し時に再初期化
        """
        with self._lock:
            self._config = None
    
    def _create_config(self, **overrides) -> GPPMConfig:
        """設定インスタンスを作成。

        Args:
            **overrides: 設定値のオーバーライド

        Returns:
            作成されたGPPMConfigインスタンス

        Raises:
            FileNotFoundError: 設定ファイルが見つからない場合
            ValidationError: 設定値の検証に失敗した場合
            ValueError: 設定作成時のエラー
        """
        try:
            config = GPPMConfig(**overrides)
            config.setup_logging()
            return config
        except (FileNotFoundError, ValueError, ValidationError) as e:
            print(f"設定エラー: {e}")
            raise
    
    def validate_config(self, config: Optional[GPPMConfig] = None) -> Dict[str, bool]:
        """設定の検証。

        概要:
        - パス存在の検証
        - 期間の論理的一貫性チェック
        - 設定値の妥当性確認

        Args:
            config: 検証対象の設定（None時は現在の設定を使用）

        Returns:
            各項目の検証結果辞書（キー: 検証項目、値: 成功フラグ）
        """
        if config is None:
            config = self.get_config()
        
        validation_results = config.validate_paths()
        
        # 追加の検証
        try:
            # 期間の妥当性
            if config.data.end_period < config.data.start_period:
                validation_results['period_validity'] = False
            else:
                validation_results['period_validity'] = True
        except Exception:
            validation_results['period_validity'] = False
        
        return validation_results


def get_logger(name: str) -> logging.Logger:
    """gppmロガーを取得。

    概要:
    - gppm名前空間のロガーを取得
    - 名前の自動正規化（gppmプレフィックス付与）
    - 設定の自動初期化によるログレベル適用

    Args:
        name: ロガー名（モジュール名など）

    Returns:
        設定済みのlogging.Loggerインスタンス

    使用例:
        logger = get_logger(__name__)
        logger.info("処理開始")
    """
    # 設定を初期化してログレベルを適用
    try:
        config_manager = ConfigManager()
        config_manager.get_config()
    except Exception:
        # 設定初期化に失敗した場合はデフォルト設定で続行
        pass
    
    if not name.startswith("gppm"):
        name = f"gppm.{name}"
    
    return logging.getLogger(name)