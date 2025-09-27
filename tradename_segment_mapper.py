"""
Tradename Segment Mapper
========================

目的
- proto.pyの処理をCLI内に実装
- gppm_configのmapping_df.pklを生成
- GPU環境での処理を前提とした実装

主要コンポーネント
- `TradenameSegmentMapper`: メイン処理クラス
- `TradenameSegmentMapperConfig`: 設定管理用Pydanticモデル
- `TradenameSegmentMapperResult`: 結果管理用Pydanticモデル

パフォーマンス設計
- GPU環境での処理を前提
- 大規模データセットに対応
- メモリ効率的な処理

使用例
    from gppm.analysis.tradename_segment_mapper import TradenameSegmentMapper

    mapper = TradenameSegmentMapper()
    result = mapper.generate_mapping_df(
        output_path="/path/to/mapping_df.pkl"
    )
"""

from __future__ import annotations

import os
import re
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp

import pandas as pd
import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    field_serializer,
    computed_field,
    model_validator,
)
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DataFrameLoader
from tqdm import tqdm
import faiss

from gppm.core.config_manager import get_logger
from data_providers.sources.factset_rbics.provider import RBICSProvider
from data_providers.sources.factset_tradenames.provider import FactSetTradenameProvider
from data_providers.sources.factset_rbics.query_params import RBICSQueryParams
from data_providers.sources.factset_tradenames.query_params import FactSetTradenameQueryParams
from wolf_period import WolfPeriod

logger = get_logger(__name__)


class TradenameSegmentMapperConfig(BaseModel):
    """Tradename Segment Mapperの設定管理。

    目的:
    - Tradename Segment Mapperに必要な設定を型安全に管理
    - GPU環境での処理を前提とした設定
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # モデル設定
    model_path: str = Field(
        default="/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202411/BAAI-bge-m3-langchain",
        description="埋め込みモデルのパス"
    )
    chunk_size: int = Field(
        default=100000,
        description="ドキュメント分割サイズ"
    )
    max_items_per_entity: int = Field(
        default=100,
        description="企業あたりの最大商品数"
    )

    # 検索・マッピング設定
    initial_search_ratio: float = Field(
        default=0.05,
        description="初期検索候補数（総データ数の割合）"
    )
    search_expansion_factor: int = Field(
        default=2,
        description="検索候補増加倍率"
    )
    max_search_ratio: float = Field(
        default=0.95,
        description="最大検索候補数（総データ数の割合）"
    )
    search_timeout: int = Field(
        default=3600,
        description="検索タイムアウト（秒）"
    )
    batch_size: int = Field(
        default=5000,
        description="バッチサイズ"
    )
    cpu_workers: int = Field(
        default=4,
        description="CPU並列ワーカー数"
    )
    gpu_batch_size: int = Field(
        default=1000,
        description="GPU並列バッチサイズ"
    )

    # 出力設定
    output_path: str = Field(
        description="出力ファイルパス"
    )

    @field_validator("model_path")
    @classmethod
    def _validate_model_path(cls, v: str) -> str:
        """モデルパスの存在確認。"""
        if not os.path.exists(v):
            raise ValueError(f"モデルパスが存在しません: {v}")
        return v

    @field_validator("output_path")
    @classmethod
    def _validate_output_path(cls, v: str) -> str:
        """出力パスの親ディレクトリの存在確認。"""
        parent_dir = Path(v).parent
        if not parent_dir.exists():
            parent_dir.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("chunk_size", "batch_size", "max_items_per_entity", "search_expansion_factor", "search_timeout", "cpu_workers", "gpu_batch_size")
    @classmethod
    def _validate_positive_integers(cls, v: int) -> int:
        """正の整数の検証。"""
        if v <= 0:
            raise ValueError("値は正の数である必要があります")
        return v

    @field_validator("initial_search_ratio", "max_search_ratio")
    @classmethod
    def _validate_ratios(cls, v: float) -> float:
        """割合の検証。"""
        if not 0 < v <= 1:
            raise ValueError("割合は0より大きく1以下である必要があります")
        return v

    @model_validator(mode="after")
    def _validate_config_consistency(self) -> "TradenameSegmentMapperConfig":
        """設定の一貫性検証。"""
        if self.chunk_size < self.batch_size:
            raise ValueError("chunk_sizeはbatch_size以上である必要があります")
        
        if self.initial_search_ratio > self.max_search_ratio:
            raise ValueError("initial_search_ratioはmax_search_ratio以下である必要があります")
        
        if self.search_expansion_factor <= 1:
            raise ValueError("search_expansion_factorは1より大きい必要があります")
        
        return self


class TradenameSegmentMapperResult(BaseModel):
    """Tradename Segment Mapper結果の管理。

    目的:
    - 生成結果の型安全な管理
    - 検証とシリアライゼーション
    """

    model_config = ConfigDict(
        frozen=True,
        str_strip_whitespace=True,
        validate_assignment=True,
        extra="forbid",
    )

    # 基本情報
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="生成日時"
    )
    total_mappings: int = Field(
        description="総マッピング数"
    )
    total_entities: int = Field(
        description="総企業数"
    )

    # データ統計
    sector_records_count: int = Field(
        description="セクターレコード数"
    )
    tradename_records_count: int = Field(
        description="商品名レコード数"
    )

    # ファイル情報
    output_path: str = Field(
        description="出力ファイルパス"
    )
    file_size_mb: Optional[float] = Field(
        default=None,
        description="ファイルサイズ（MB）"
    )

    @computed_field
    @property
    def success_rate(self) -> float:
        """マッピング成功率の計算。"""
        if self.total_entities == 0:
            return 0.0
        return self.total_mappings / self.total_entities

    @field_validator("total_mappings", "total_entities", "sector_records_count", "tradename_records_count")
    @classmethod
    def _validate_non_negative_integers(cls, v: int) -> int:
        """非負整数の検証。"""
        if v < 0:
            raise ValueError("値は0以上である必要があります")
        return v

    @field_validator("file_size_mb")
    @classmethod
    def _validate_file_size(cls, v: Optional[float]) -> Optional[float]:
        """ファイルサイズの検証。"""
        if v is not None and v < 0:
            raise ValueError("ファイルサイズは0以上である必要があります")
        return v

    @model_validator(mode="after")
    def _validate_result_consistency(self) -> "TradenameSegmentMapperResult":
        """結果の一貫性検証。"""
        if self.total_mappings > self.total_entities:
            raise ValueError("マッピング数は企業数を超えることはできません")
        
        if self.success_rate > 1.0:
            raise ValueError("成功率は1.0を超えることはできません")
        
        return self

    @field_serializer("generated_at")
    def _serialize_datetime(self, dt: datetime) -> str:
        """日時をシリアライズ。"""
        return dt.isoformat()


class TradenameSegmentMapper:
    """Tradename Segment Mapperのメインクラス。

    概要
    ----
    proto.pyの処理をCLI内に実装し、mapping_df.pklを生成します。
    GPU環境での処理を前提とし、大規模データセットに対応します。

    主要機能
    --------
    - セグメントデータの取得と処理
    - 商品名データの取得と処理
    - 埋め込みベクトルの生成
    - ベクトル検索によるマッピング生成
    - 結果の保存と検証

    パフォーマンス最適化
    ------------------
    - GPU環境での高速処理
    - バッチ処理による効率化
    - メモリ効率的なデータ処理
    - 並列処理による高速化

    主要メソッド
    ------------
    - generate_mapping_df(): メイン処理
    - _check_gpu_environment(): GPU環境チェック
    - _load_embedding_model(): 埋め込みモデル読み込み
    - _process_sector_data(): セクターデータ処理
    - _process_tradename_data(): 商品名データ処理
    - _create_vectorstore(): ベクトルストア作成
    - _generate_mappings(): マッピング生成
    - _save_results(): 結果保存

    制約事項
    --------
    - GPU環境での実行が必要
    - 十分なメモリ容量が必要
    - モデルファイルの存在が必要

    例外処理
    --------
    - RuntimeError: GPU環境が利用できない場合
    - FileNotFoundError: 必要なファイルが見つからない場合
    - MemoryError: メモリ不足の場合
    - ValueError: 設定が不正な場合
    """

    def __init__(self, config: Optional[TradenameSegmentMapperConfig] = None):
        """コンストラクタ。

        Args:
            config: 設定オブジェクト（省略時はデフォルト設定）
        """
        self.config = config or TradenameSegmentMapperConfig(
            output_path="/tmp/mapping_df.pkl"
        )
        self._embedding_model: Optional[HuggingFaceEmbeddings] = None
        self._sentence_transformer: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=self.config.cpu_workers)
        self._process_executor = ProcessPoolExecutor(max_workers=min(4, mp.cpu_count()))
        self._gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._mapping_cache: Dict[Tuple[str, str], Tuple[str, str]] = {}
        self._performance_stats = {
            "total_queries": 0,
            "successful_mappings": 0,
            "failed_mappings": 0,
            "processing_time": 0.0
        }
        
        logger.info(f"TradenameSegmentMapper初期化完了 - GPU: {torch.cuda.is_available()}, "
                   f"CPU並列ワーカー: {self.config.cpu_workers}, "
                   f"バッチサイズ: {self.config.batch_size}")

    def __del__(self):
        """リソースクリーンアップ。"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
        if hasattr(self, '_process_executor'):
            self._process_executor.shutdown(wait=False)

    def generate_mapping_df(self, output_path: Optional[str] = None) -> TradenameSegmentMapperResult:
        """Mapping DataFrame生成のメイン処理。

        Args:
            output_path: 出力ファイルパス（省略時は設定値を使用）

        Returns:
            生成結果の情報

        Raises:
            RuntimeError: GPU環境が利用できない場合
            FileNotFoundError: 必要なファイルが見つからない場合
            MemoryError: メモリ不足の場合
        """
        if output_path:
            self.config = self.config.model_copy(update={"output_path": output_path})

        logger.info("Tradename Segment Mapping生成開始")
        
        # 既存のmapping_file設定を確認
        try:
            from gppm.core.config_manager import ConfigManager
            config_manager = ConfigManager()
            config = config_manager.get_config()
            if config.optional_data.mapping_file:
                logger.info(f"既存のマッピングファイルが設定されています: {config.optional_data.mapping_file}")
                logger.info("新しくマッピングファイルを生成します")
        except Exception as e:
            logger.debug(f"設定ファイルの読み込みエラー: {e}")

        # GPU環境チェック
        self._check_gpu_environment()

        # 埋め込みモデル読み込み
        self._load_embedding_model()

        # セクターデータ処理
        sector_df = self._process_sector_data()
        logger.info(f"セクターデータ処理完了: {len(sector_df)} 件")

        # 商品名データ処理
        tradename_df = self._process_tradename_data()
        logger.info(f"商品名データ処理完了: {len(tradename_df)} 件")

        # ベクトルストア作成
        vectorstore_search, vectorstore_searched = self._create_vectorstore(sector_df, tradename_df)
        logger.info("ベクトルストア作成完了")

        # マッピング生成
        mapping_df = self._generate_mappings(vectorstore_search, vectorstore_searched)
        logger.info(f"マッピング生成完了: {len(mapping_df)} 件")

        # 結果保存
        result = self._save_results(mapping_df, sector_df, tradename_df)
        logger.info(f"結果保存完了: {self.config.output_path}")

        return result

    def _check_gpu_environment(self) -> None:
        """GPU環境のチェック。

        Raises:
            RuntimeError: GPU環境が利用できない場合
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU環境が利用できません。CUDAが有効な環境で実行してください。"
            )
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU環境確認完了: {gpu_name}, メモリ: {gpu_memory:.1f}GB")

    def _load_embedding_model(self) -> None:
        """埋め込みモデルの読み込み。"""
        logger.info(f"埋め込みモデル読み込み開始: {self.config.model_path}")
        
        try:
            # SentenceTransformerモデルの読み込み
#            self._sentence_transformer = SentenceTransformer(self.config.model_path)
            
            # HuggingFaceEmbeddingsの設定（最新仕様に合わせて修正）
            self._embedding_model = HuggingFaceEmbeddings(
                
                model_name=self.config.model_path,
                model_kwargs={"device": "cuda"}
            )
            
            logger.info("埋め込みモデル読み込み完了")
            
        except Exception as e:
            logger.error(f"埋め込みモデル読み込みエラー: {e}")
            raise

    def _process_sector_data(self) -> pd.DataFrame:
        """セクターデータの処理。

        Returns:
            処理済みセクターデータのDataFrame
        """
        logger.info("セクターデータ処理開始")

        # RBICSプロバイダーを使用してセグメントデータを取得
        rbics_provider = RBICSProvider()
        
        # 現在の日付でクエリパラメータを作成
        current_date = datetime.now(timezone.utc).date()
        query_params = RBICSQueryParams(
            period=WolfPeriod.from_day(current_date)
        )

        try:
            # セグメントデータの取得（生データ）
            raw_sector_df = rbics_provider._query_revenue_segment_data(query_params)
            
            if raw_sector_df.empty:
                raise ValueError("セグメントデータが取得できませんでした")

            # RBICSマスターデータの取得（REVENUE_L6_NAME, REVENUE_DESCR用）
            rbics_master = rbics_provider._query_structure_data(query_params)
            
            if rbics_master.empty:
                raise ValueError("RBICSマスターデータが取得できませんでした")
                
        except Exception as e:
            logger.error(f"データ取得エラー: {e}")
            raise

        # データの前処理（処理済みデータ）
        processed_sector_df = self._preprocess_sector_data(raw_sector_df, rbics_master)
        
        return processed_sector_df

    def _preprocess_sector_data(self, df: pd.DataFrame, rbics_master: pd.DataFrame) -> pd.DataFrame:
        """セクターデータの前処理。

        Args:
            df: 生のセクターデータ
            rbics_master: RBICSマスターデータ

        Returns:
            前処理済みのセクターデータ
        """
        df = df.copy()

        # セグメント名の正規化
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.strip()
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.replace(r'[一ー]', '-', regex=True)
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.replace(r'([^\w\-])', '-', regex=True)
        df["SEGMENT_NAME"] = df["SEGMENT_NAME"].str.replace(r'(\w)\s', r'\1 - ', regex=True)

        # 売上比率の計算
        df["SALES_RATIO"] = df["SEG_SHARE"].astype(float) / 100

        # 会計年度の計算
        df["FISCAL_YEAR"] = df["PERIOD_END_DATE"].map(self._get_fiscal_year)

        # 売上比率の合計計算
        df["SEG_SHARE_SUM"] = df.groupby(["FACTSET_ENTITY_ID", "FISCAL_YEAR"])["SEG_SHARE"].transform("sum")

        # ソート
        df.sort_values(by=["FACTSET_ENTITY_ID", "FISCAL_YEAR", "SALES_RATIO"], inplace=True)

        # 重複行の削除
        df = df.groupby(["FACTSET_ENTITY_ID", "FISCAL_YEAR"]).apply(self._remove_redundant_rows).reset_index(drop=True)

        # RBICSマスターデータとの結合（REVENUE_L6_NAME, REVENUE_DESCRを追加）
        df = df.merge(
            rbics_master[["L6_ID", "L6_NAME", "L6_DESCR"]].rename(
                columns={"L6_ID": "REVENUE_L6_ID", "L6_NAME": "REVENUE_L6_NAME", "L6_DESCR": "REVENUE_DESCR"}
            ),
            on="REVENUE_L6_ID",
            how="left"
        )

        # テキスト前処理（カンマをハイフンに置換）
        if 'REVENUE_L6_NAME' in df.columns:
            df['REVENUE_L6_NAME'] = df['REVENUE_L6_NAME'].str.replace(',', '-')

        return df

    def _remove_redundant_rows(self, group: pd.DataFrame) -> pd.DataFrame:
        """重複行の削除処理。"""
        flag = False
        if group["SEG_SHARE"].sum() != 101:
            flag = True
        indices_to_remove = []

        for idx, seg in group["SEGMENT_NAME"].items():
            search_str = seg + " "
            if not flag:
                if any(other_seg.startswith(search_str) for j, other_seg in group["SEGMENT_NAME"].items() if j != idx):
                    indices_to_remove.append(idx)

        for j, other_seg in group["SEGMENT_NAME"].items():
            if other_seg.startswith(search_str):
                if not other_seg.startswith(seg + ' - '):
                    updated_seg = seg + " - " + other_seg[len(seg):].lstrip()
                    group.at[j, 'SEGMENT_NAME'] = updated_seg

        return group.drop(indices_to_remove)

    def _get_fiscal_year(self, date_val) -> int:
        """日付から会計年度を取得。"""
        if pd.isna(date_val):
            return None
        date_obj = pd.to_datetime(date_val)
        if date_obj.month >= 4:
            return date_obj.year
        else:
            return date_obj.year - 1

    def _process_tradename_data(self) -> pd.DataFrame:
        """商品名データの処理。

        Returns:
            処理済み商品名データのDataFrame
        """
        logger.info("商品名データ処理開始")

        # FactSet Tradenameプロバイダーを使用
        tradename_provider = FactSetTradenameProvider()
        
        # クエリパラメータの作成
        query_params = FactSetTradenameQueryParams(
            active_only=True,
            batch_size=self.config.batch_size,
            max_items_per_entity=self.config.max_items_per_entity
        )

        # 商品名データの取得
        tradename_records = tradename_provider.get_tradename_records(query_params)
        
        if not tradename_records:
            raise ValueError("商品名データが取得できませんでした")

        # レコードをDataFrameに変換
        tradename_data = []
        for record in tradename_records:
            tradename_data.append({
                "FACTSET_ENTITY_ID": record.factset_entity_id,
                "PRODUCT_NAME": record.product_name,
                "L6_ID": record.l6_id,
                "L6_NAME": record.l6_name,
                "SEARCH_TEXT": record.search_text
            })

        tradename_df = pd.DataFrame(tradename_data)
        
        # データの前処理
        tradename_df = self._preprocess_tradename_data(tradename_df)
        
        return tradename_df

    def _preprocess_tradename_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """商品名データの前処理。

        Args:
            df: 生の商品名データ

        Returns:
            前処理済みの商品名データ
        """
        df = df.copy()

        # 商品名の正規化
        df['PRODUCT_NAME'] = df['PRODUCT_NAME'].apply(self._extract_right_side)

        # 企業・L6・L6_NAMEでグループ化して商品名を結合
        df = (df
            .groupby(['FACTSET_ENTITY_ID', 'L6_ID', 'L6_NAME'])['PRODUCT_NAME']
            .apply(lambda x: ' , '.join(x.head(self.config.max_items_per_entity)))
            .to_frame("PRODUCT_NAME")
            .sort_values(by="PRODUCT_NAME", key=lambda x: x.str.len(), ascending=False)
            .reset_index(drop=False))

        # 検索テキストの生成
        df['SEARCH_TEXT'] = df['L6_NAME'] + ' : ' + df['PRODUCT_NAME']

        # ソート処理
        df = self._sort_tradename_data(df)

        return df

    def _extract_right_side(self, text: str) -> str:
        """テキストから右側の要素を抽出。"""
        match = re.split(r'\s*-\s*', text, maxsplit=1)
        return match[-1] if len(match) > 1 else text

    def _sort_tradename_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """商品名データのソート処理。"""
        sorted_idx = df['SEARCH_TEXT'].str.len().sort_values().index
        sorted_idx_list = sorted_idx.tolist()
        
        left = 0
        right = len(sorted_idx_list) - 1
        new_index = []

        while left <= right:
            if left == right:
                new_index.append(sorted_idx_list[left])
                break
            new_index.append(sorted_idx_list[left])
            new_index.append(sorted_idx_list[right])
            left += 1
            right -= 1

        return df.loc[new_index].reset_index(drop=True)

    def _create_vectorstore(self, sector_df: pd.DataFrame, tradename_df: pd.DataFrame) -> FAISS:
        """ベクトルストアの作成。

        Args:
            sector_df: セクターデータ
            tradename_df: 商品名データ

        Returns:
            作成されたベクトルストア
        """
        logger.info("ベクトルストア作成開始")
        start_time = time.time()

        # セクターデータの処理
        sector_df_processed = sector_df[["FACTSET_ENTITY_ID", "SEG_SHARE", "SEGMENT_NAME", "REVENUE_L6_ID", 
                                         "REVENUE_L6_NAME", "REVENUE_DESCR"]].copy()
        sector_df_processed["SEARCHED_TEXT"] = sector_df_processed["REVENUE_L6_NAME"] + " , " + sector_df_processed["REVENUE_DESCR"]
        sector_df_processed.drop_duplicates(subset=["FACTSET_ENTITY_ID", "SEARCHED_TEXT"], inplace=True)
        sector_df_processed.query("SEARCHED_TEXT.notna()", inplace=True)

        # 不正文字の削除
        sector_df_processed['SEGMENT_NAME'] = sector_df_processed['SEGMENT_NAME'].apply(self._remove_illegal_characters)

        # ドキュメントローダーの作成
        loader1 = DataFrameLoader(sector_df_processed.dropna(subset=["SEARCHED_TEXT"]), page_content_column="SEARCHED_TEXT")
        loader2 = DataFrameLoader(tradename_df, page_content_column="SEARCH_TEXT")

        documents1 = loader1.load()
        documents2 = loader2.load()

        # メモリ効率的なドキュメント分割
        def split_documents_optimized(documents, chunk_size):
            for start in range(0, len(documents), chunk_size):
                yield documents[start:start + chunk_size]

        chunks1 = list(split_documents_optimized(documents1, self.config.chunk_size))
        chunks2 = list(split_documents_optimized(documents2, self.config.chunk_size))

        # 並列処理でベクトルストア作成
        def create_vectorstore_chunk(chunk, embedding_model):
            return FAISS.from_documents(chunk, embedding_model)

        # セクターデータのベクトルストア作成（検索先）
        logger.info("セクターデータベクトルストア作成開始")
        vectorstore_searched = None
        for i, chunk in enumerate(tqdm(chunks1, desc="セクターデータ処理")):
            try:
                if vectorstore_searched is None:
                    vectorstore_searched = create_vectorstore_chunk(chunk, self._embedding_model)
                else:
                    temp_vectorstore = create_vectorstore_chunk(chunk, self._embedding_model)
                    vectorstore_searched.merge_from(temp_vectorstore)
                    
                # メモリクリーンアップ
                if 'temp_vectorstore' in locals():
                    del temp_vectorstore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"セクターデータチャンク{i}処理エラー: {e}")
                continue

        # 商品名データのベクトルストア作成（検索元）
        logger.info("商品名データベクトルストア作成開始")
        vectorstore_search = None
        for i, chunk in enumerate(tqdm(chunks2, desc="商品名データ処理")):
            try:
                if vectorstore_search is None:
                    vectorstore_search = create_vectorstore_chunk(chunk, self._embedding_model)
                else:
                    temp_vectorstore = create_vectorstore_chunk(chunk, self._embedding_model)
                    vectorstore_search.merge_from(temp_vectorstore)
                    
                # メモリクリーンアップ
                if 'temp_vectorstore' in locals():
                    del temp_vectorstore
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"商品名データチャンク{i}処理エラー: {e}")
                continue

        elapsed_time = time.time() - start_time
        logger.info(f"ベクトルストア作成完了: 検索先={vectorstore_searched.index.ntotal}件, 検索元={vectorstore_search.index.ntotal}件 (処理時間: {elapsed_time:.2f}秒)")
        return vectorstore_search, vectorstore_searched

    def _remove_illegal_characters(self, value: str) -> str:
        """不正な制御文字を削除。"""
        if isinstance(value, str):
            return re.sub(r'[\x00-\x1F\x7F-\x9F]', '', value)
        return value

    def _generate_mappings(self, vectorstore_search: FAISS, vectorstore_searched: FAISS) -> pd.DataFrame:
        """マッピングの生成。

        Args:
            vectorstore_search: 検索元ベクトルストア（商品名データ）
            vectorstore_searched: 検索先ベクトルストア（セクターデータ）

        Returns:
            生成されたマッピングDataFrame
        """
        logger.info("マッピング生成開始")
        start_time = time.time()

        # クエリ準備（検索元ベクトルストアから）
        doc_id_to_faiss_id = {v: k for k, v in vectorstore_search.index_to_docstore_id.items()}
        query_ids = list(vectorstore_search.docstore._dict.keys())

        queries_embeddings = []
        query_metadata_list = []

        for doc_id in query_ids:
            faiss_id = doc_id_to_faiss_id[doc_id]
            emb = vectorstore_search.index.reconstruct(faiss_id)
            queries_embeddings.append(emb)

            doc = vectorstore_search.docstore.search(doc_id)
            query_metadata_list.append((doc.metadata["FACTSET_ENTITY_ID"], doc.metadata["L6_ID"]))

        queries_embeddings = np.array(queries_embeddings)

        # マッピング生成
        mapping = {}
        faiss_index = vectorstore_searched.index
        N = len(queries_embeddings)
        total_docs = vectorstore_searched.index.ntotal
        unmapped_indices = set(range(N))
        
        # パーセンテージベースの検索設定
        initial_search_count = int(total_docs * self.config.initial_search_ratio)
        max_search_count = int(total_docs * self.config.max_search_ratio)
        search_expansion_factor = self.config.search_expansion_factor
        batch_size = self.config.batch_size
        search_timeout = self.config.search_timeout

        # 最適化されたバッチ処理（キャッシュ機能付き）
        def process_batch_optimized(batch_indices, query_batch, batch_metadata, k_val):
            batch_mapping = {}
            batch_unmapped = set()
            
            try:
                # キャッシュチェック
                cache_hits = 0
                for i, (factset_id_query, l6_id) in enumerate(batch_metadata):
                    cache_key = (factset_id_query, l6_id)
                    if cache_key in self._mapping_cache:
                        batch_mapping[cache_key] = self._mapping_cache[cache_key]
                        cache_hits += 1
                    else:
                        batch_unmapped.add(batch_indices[i])
                
                if cache_hits > 0:
                    logger.debug(f"キャッシュヒット: {cache_hits}/{len(batch_metadata)}")
                
                if not batch_unmapped:
                    return batch_mapping, set()
                
                # GPU並列検索
                distances, indices = faiss_index.search(query_batch, k_val)
                candidate_idxs = set(indices.flatten())
                candidate_idxs.discard(-1)
                
                if candidate_idxs:
                    # 候補ドキュメントの並列取得
                    candidate_doc_ids = {idx: vectorstore_searched.index_to_docstore_id[idx] for idx in candidate_idxs}
                    candidate_docs = {
                        doc_id: vectorstore_searched.docstore.search(doc_id)
                        for doc_id in candidate_doc_ids.values()
                    }

                    # 効率的なマッピング検索
                    for i, idx_list in enumerate(indices):
                        global_i = batch_indices[i]
                        factset_id_query, l6_id = batch_metadata[i]
                        
                        # キャッシュチェック
                        cache_key = (factset_id_query, l6_id)
                        if cache_key in self._mapping_cache:
                            batch_mapping[cache_key] = self._mapping_cache[cache_key]
                            continue
                        
                        # 早期終了のための最適化
                        found = False
                        for idx in idx_list:
                            if idx == -1:
                                continue

                            doc_id = candidate_doc_ids.get(idx)
                            doc = candidate_docs.get(doc_id)

                            if doc is not None and doc.metadata.get("FACTSET_ENTITY_ID") == factset_id_query:
                                revere_l6_id = doc.metadata.get("REVENUE_L6_ID")
                                mapping_result = (factset_id_query, revere_l6_id)
                                batch_mapping[cache_key] = mapping_result
                                self._mapping_cache[cache_key] = mapping_result
                                found = True
                                break
                        
                        if not found:
                            batch_unmapped.add(global_i)
                else:
                    batch_unmapped = set(batch_indices)
                    
            except Exception as e:
                logger.error(f"バッチ処理エラー: {e}")
                batch_unmapped = set(batch_indices)
            
            return batch_mapping, batch_unmapped

        iteration = 0
        current_search_count = initial_search_count
        
        with tqdm(total=N, desc="マッピング生成", unit="件", 
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]') as pbar:
            while unmapped_indices:
                iteration += 1
                batch_start_time = time.time()
                
                logger.info(f"検索実行: 検索候補数={current_search_count}, 未マッピング={len(unmapped_indices)}, 反復={iteration}")
                
                unmapped_list = list(unmapped_indices)
                batch_mappings = {}
                new_unmapped = set()
                
                # 並列処理でバッチを処理
                futures = []
                for start_idx in range(0, len(unmapped_list), batch_size):
                    batch_indices = unmapped_list[start_idx : start_idx + batch_size]
                    query_batch = queries_embeddings[batch_indices]
                    batch_metadata = [query_metadata_list[i] for i in batch_indices]
                    
                    future = self._executor.submit(process_batch_optimized, batch_indices, query_batch, batch_metadata, current_search_count)
                    futures.append(future)
                
                # 結果を収集
                successful_batches = 0
                for future in futures:
                    try:
                        batch_mapping, batch_unmapped = future.result(timeout=300)
                        batch_mappings.update(batch_mapping)
                        new_unmapped.update(batch_unmapped)
                        successful_batches += 1
                    except Exception as e:
                        logger.error(f"バッチ処理エラー: {e}")
                        continue
                
                # マッピング結果を統合
                mapping.update(batch_mappings)
                unmapped_indices = new_unmapped
                
                # パフォーマンス統計更新
                self._performance_stats["total_queries"] += len(unmapped_list)
                self._performance_stats["successful_mappings"] += len(batch_mappings)
                self._performance_stats["failed_mappings"] += len(new_unmapped)
                
                # プログレス更新
                mapped_count = len(mapping)
                pbar.update(mapped_count - pbar.n)
                
                # 詳細な進捗情報
                batch_time = time.time() - batch_start_time
                success_rate = len(batch_mappings) / len(unmapped_list) if unmapped_list else 0
                logger.info(f"バッチ処理完了: 成功={len(batch_mappings)}, 失敗={len(new_unmapped)}, "
                           f"成功率={success_rate:.2%}, 処理時間={batch_time:.2f}秒")
                
                # 早期終了条件
                if not unmapped_indices:
                    break
                    
                if current_search_count >= max_search_count:
                    logger.warning(f"最大検索候補数({max_search_count})に到達しました。ループを終了します。")
                    break
                    
                current_search_count = min(current_search_count * search_expansion_factor, max_search_count)
                
                # 時間制限チェック
                elapsed_time = time.time() - start_time
                if elapsed_time > search_timeout:
                    logger.warning(f"検索タイムアウト({search_timeout}秒)に達しました。処理を終了します。")
                    break

        elapsed_time = time.time() - start_time
        self._performance_stats["processing_time"] = elapsed_time
        
        # 最終統計情報
        final_success_rate = len(mapping) / N if N > 0 else 0
        logger.info(f"マッピング完了: {len(mapping)} 件 / {N} 件 (処理時間: {elapsed_time:.2f}秒, 成功率: {final_success_rate:.2%})")
        logger.info(f"パフォーマンス統計: 総クエリ={self._performance_stats['total_queries']}, "
                   f"成功マッピング={self._performance_stats['successful_mappings']}, "
                   f"失敗マッピング={self._performance_stats['failed_mappings']}")

        # DataFrameに変換（エラーハンドリング付き）
        try:
            mapping_df = pd.DataFrame.from_dict(mapping, orient='index').reset_index()
            mapping_df[['FACTSET_ENTITY_ID', 'PRODUCT_L6_ID']] = pd.DataFrame(mapping_df['index'].tolist(), index=mapping_df.index)
            mapping_df['RELABEL_L6_ID'] = mapping_df[1]
            mapping_df = mapping_df.drop(columns=['index', 0])[['FACTSET_ENTITY_ID', 'PRODUCT_L6_ID', 'RELABEL_L6_ID']]
            
            # データ検証
            if mapping_df.empty:
                logger.warning("マッピング結果が空です")
            else:
                logger.info(f"DataFrame作成完了: {len(mapping_df)} 行")
                
        except Exception as e:
            logger.error(f"DataFrame変換エラー: {e}")
            # 空のDataFrameを返す
            mapping_df = pd.DataFrame(columns=['FACTSET_ENTITY_ID', 'PRODUCT_L6_ID', 'RELABEL_L6_ID'])

        return mapping_df

    def _save_results(self, mapping_df: pd.DataFrame, sector_df: pd.DataFrame, tradename_df: pd.DataFrame) -> TradenameSegmentMapperResult:
        """結果の保存。

        Args:
            mapping_df: マッピングDataFrame
            sector_df: セクターデータ
            tradename_df: 商品名データ

        Returns:
            保存結果の情報
        """
        logger.info("結果保存開始")

        # マッピングDataFrameの保存
        mapping_df.to_pickle(self.config.output_path)

        # ファイルサイズの計算
        file_size_mb = os.path.getsize(self.config.output_path) / (1024 * 1024)

        # 結果の作成
        result = TradenameSegmentMapperResult(
            total_mappings=len(mapping_df),
            total_entities=len(mapping_df['FACTSET_ENTITY_ID'].unique()),
            sector_records_count=len(sector_df),
            tradename_records_count=len(tradename_df),
            output_path=self.config.output_path,
            file_size_mb=file_size_mb
        )

        logger.info(f"結果保存完了: {result.total_mappings} 件のマッピング")
        return result


__all__ = [
    "TradenameSegmentMapperConfig",
    "TradenameSegmentMapperResult", 
    "TradenameSegmentMapper",
]
