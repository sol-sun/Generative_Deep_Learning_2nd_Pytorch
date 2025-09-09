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
    from gppm.core.config_manager import ConfigManager, get_logger

    # 設定マネージャーの初期化
    config_manager = ConfigManager()
    config = config_manager.get_config()

    # 設定値の取得
    start_period = config.analysis_period.start
    end_period = config.analysis_period.end
    output_dir = config.output.directory
    parallel_workers = config.data_processing.parallel_workers
    inference_engine = config.variational_inference.inference_config.engine

    # ロガーの取得
    logger = get_logger(__name__)
    logger.info("設定読み込み完了: 期間=%d-%d, ワーカー数=%d", 
                start_period, end_period, parallel_workers)

    # 設定の検証
    validation_results = config_manager.validate_config()
    if not all(validation_results.values()):
        logger.warning("設定検証エラー: %s", validation_results)
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


class AnalysisPeriodConfig(BaseModel):
    """分析期間設定。

    概要:
    - 分析対象の期間範囲を管理します。
    - YYYYMM形式での期間指定と妥当性検証を提供します。

    特徴:
    - 開始期間と終了期間の自動検証
    - 期間の論理的一貫性チェック
    """
    start: int = Field(
        description="開始期間（YYYYMM形式）",
        examples=[201909, 202001, 202406]
    )
    end: int = Field(
        description="終了期間（YYYYMM形式）",
        examples=[202001, 202406, 202412]
    )
    
    @field_validator('end')
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
        if 'start' in info.data and v < info.data['start']:
            raise ValueError('終了期間は開始期間より後でなければなりません。')
        return v


class OutputFilesConfig(BaseModel):
    """出力ファイル設定。

    概要:
    - 各種出力ファイルの名前を管理します。
    """
    variational_result: str = Field(
        description="変分推論結果ファイル名",
        examples=["variational_inference_result.pkl"]
    )
    dataset: str = Field(
        description="データセットファイル名",
        examples=["dataset.pkl"]
    )
    csv_processed: str = Field(
        description="CSV処理結果ファイル名",
        examples=["csv_processed_result.pkl"]
    )
    integrated: str = Field(
        description="統合結果ファイル名",
        examples=["integrated_analysis_result.pkl"]
    )


class OutputConfig(BaseModel):
    """出力設定。

    概要:
    - 処理結果の出力先ディレクトリとファイル名を管理します。
    - パスオブジェクトの自動変換とディレクトリ作成を提供します。

    特徴:
    - 文字列からPathオブジェクトへの自動変換
    - 出力ディレクトリの存在確認
    """
    directory: Path = Field(
        description="出力ディレクトリ（処理結果の保存先）",
        examples=["/tmp/gppm_output", "./output", "/data/gppm"]
    )
    files: OutputFilesConfig = Field(
        description="出力ファイル名設定"
    )
    
    @field_validator('directory', mode='before')
    @classmethod
    def convert_to_path(cls, v) -> Path:
        """文字列をPathオブジェクトに変換。

        Args:
            v: 変換対象の値（文字列またはPathオブジェクト）

        Returns:
            変換後のPathオブジェクト
        """
        return Path(v) if isinstance(v, str) else v


class InferenceConfig(BaseModel):
    """変分推論設定。

    概要:
    - 変分推論によるベイズ分析の設定を管理します。
    """
    engine: str = Field(
        description="推論エンジン",
        examples=["cmdstanpy", "pymc"]
    )
    model_file: Optional[str] = Field(
        default=None,
        description="Stanモデルファイル（nullの場合はデフォルトモデル）"
    )
    samples: int = Field(
        description="サンプル数（事後分布からのサンプル数）",
        examples=[2000, 5000]
    )
    optimization_iterations: int = Field(
        description="最適化反復数（変分推論の最適化回数）",
        examples=[500000, 1000000]
    )
    random_seed: int = Field(
        description="乱数シード（再現性のため）",
        examples=[42, 123]
    )
    require_convergence: bool = Field(
        description="収束を要求するか",
        examples=[True, False]
    )


class VariationalInferenceConfig(BaseModel):
    """変分推論全体設定。

    概要:
    - 既存の変分推論結果ファイルと新しい推論設定を管理します。
    """
    existing_result_path: Optional[str] = Field(
        default=None,
        description="既存の変分推論結果ファイルパス"
    )
    inference_config: InferenceConfig = Field(
        description="新しい変分推論を実行する場合の設定"
    )


class DataProcessingConfig(BaseModel):
    """データ処理設定。

    概要:
    - データ処理のパフォーマンス設定を管理します。
    """
    parallel_workers: int = Field(
        description="CSV処理の並列化設定（CPUコア数に応じて調整）",
        examples=[1, 4, 8]
    )
    memory_limit_mb: int = Field(
        description="メモリ制限（MB）",
        examples=[1024, 2048, 4096]
    )
    data_type: str = Field(
        description="データ型（メモリ使用量を削減したい場合は'float32'に変更）",
        examples=["float64", "float32"]
    )


class OptionalDataConfig(BaseModel):
    """オプションデータ設定。

    概要:
    - オプションのデータファイルパスを管理します。
    """
    mapping_file: Optional[str] = Field(
        default=None,
        description="マッピングファイルパス（デフォルト値を使用する場合はnull）"
    )




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
    
    # 分析期間設定
    analysis_period: AnalysisPeriodConfig = Field(
        description="分析期間設定"
    )
    
    # 出力設定
    output: OutputConfig = Field(
        description="出力設定"
    )
    
    # 変分推論設定
    variational_inference: VariationalInferenceConfig = Field(
        description="変分推論設定"
    )
    
    # データ処理設定
    data_processing: DataProcessingConfig = Field(
        description="データ処理設定"
    )
    
    # ログ設定
    log_level: str = Field(
        description="ログレベル（DEBUG/INFO/WARNING/ERROR/CRITICAL）",
        examples=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    
    # オプションデータ設定
    optional_data: OptionalDataConfig = Field(
        description="オプションデータ設定"
    )
    
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
    
    def validate_paths(self) -> Dict[str, bool]:
        """パスの存在を検証。

        概要:
        - 出力ディレクトリ、マッピングファイルの存在確認
        - ディレクトリ作成可能性の検証

        Returns:
            各パスの検証結果辞書（キー: パス種別、値: 存在フラグ）
        """
        results = {}
        
        # 出力ディレクトリの確認
        results['output_directory'] = self.output.directory.exists() or self.output.directory.parent.exists()
        
        # マッピングファイルの確認
        if self.optional_data.mapping_file:
            mapping_path = Path(self.optional_data.mapping_file)
            results['mapping_file'] = mapping_path.exists()
        else:
            results['mapping_file'] = True  # オプショナルなので存在しなくてもOK
        
        # 既存の変分推論結果ファイルの確認
        if self.variational_inference.existing_result_path:
            existing_path = Path(self.variational_inference.existing_result_path)
            results['existing_result_file'] = existing_path.exists()
        else:
            results['existing_result_file'] = True  # オプショナルなので存在しなくてもOK
        
        return results
    
    def setup_logging(self) -> None:
        """ログ設定を適用。

        概要:
        - gppmロガーの設定を適用
        - コンソールにログ出力
        - 他のライブラリのログレベルを制御

        特徴:
        - フォーマッターによる統一されたログ形式
        - ログレベルの動的設定
        """
        level = getattr(logging, self.log_level, logging.INFO)
        
        # フォーマッター
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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
        
        # 他のライブラリのログレベルを制御
        logging.getLogger().setLevel(logging.WARNING)
        logger.propagate = False


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
                    try:
                        self._config = GPPMConfig()
                        self._config.setup_logging()
                    except (FileNotFoundError, ValueError, ValidationError) as e:
                        print(f"設定エラー: {e}")
                        raise
        return self._config        
    
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
            if config.analysis_period.end < config.analysis_period.start:
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