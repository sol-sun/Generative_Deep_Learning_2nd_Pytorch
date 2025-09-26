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
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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

    # 検索設定
    initial_k: int = Field(
        default=100000,
        description="初期検索結果数"
    )
    increment_factor: int = Field(
        default=2,
        description="k増加倍率"
    )
    batch_size: int = Field(
        default=10000,
        description="バッチサイズ"
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

    @field_validator("chunk_size", "batch_size", "max_items_per_entity", "initial_k", "increment_factor")
    @classmethod
    def _validate_positive_integers(cls, v: int) -> int:
        """正の整数の検証。"""
        if v <= 0:
            raise ValueError("値は正の数である必要があります")
        return v

    @model_validator(mode="after")
    def _validate_config_consistency(self) -> "TradenameSegmentMapperConfig":
        """設定の一貫性検証。"""
        if self.chunk_size < self.batch_size:
            raise ValueError("chunk_sizeはbatch_size以上である必要があります")
        
        if self.initial_k < self.batch_size:
            raise ValueError("initial_kはbatch_size以上である必要があります")
        
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
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("TradenameSegmentMapper初期化完了")

    def __del__(self):
        """リソースクリーンアップ。"""
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)

    def generate_mapping_df(self, output_path: Optional[str] = None) -> MappingResult:
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
        
        logger.info(f"GPU環境確認完了: {torch.cuda.get_device_name(0)}")

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

        # ドキュメントの分割
        def split_documents(documents, chunk_size):
            for start in range(0, len(documents), chunk_size):
                yield documents[start:start + chunk_size]

        chunks1 = list(split_documents(documents1, self.config.chunk_size))
        chunks2 = list(split_documents(documents2, self.config.chunk_size))

        # セクターデータのベクトルストア作成（検索先）
        vectorstore_searched = None
        for chunk in chunks1:
            if vectorstore_searched is None:
                vectorstore_searched = FAISS.from_documents(chunk, self._embedding_model)
            else:
                temp_vectorstore = FAISS.from_documents(chunk, self._embedding_model)
                vectorstore_searched.merge_from(temp_vectorstore)

        # 商品名データのベクトルストア作成（検索元）
        vectorstore_search = None
        for chunk in chunks2:
            if vectorstore_search is None:
                vectorstore_search = FAISS.from_documents(chunk, self._embedding_model)
            else:
                temp_vectorstore = FAISS.from_documents(chunk, self._embedding_model)
                vectorstore_search.merge_from(temp_vectorstore)

        logger.info(f"ベクトルストア作成完了: 検索先={vectorstore_searched.index.ntotal}件, 検索元={vectorstore_search.index.ntotal}件")
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
        faiss_index = vectorstore_searched.index  # 検索先のインデックスを使用
        N = len(queries_embeddings)
        unmapped_indices = set(range(N))
        k = self.config.initial_k
        increment_factor = self.config.increment_factor
        batch_size = self.config.batch_size
        max_k = vectorstore_searched.index.ntotal

        while unmapped_indices:
            logger.info(f"検索実行: k={k}, 未マッピング={len(unmapped_indices)}")
            
            unmapped_list = list(unmapped_indices)
            
            for start_idx in range(0, len(unmapped_list), batch_size):
                batch_indices = unmapped_list[start_idx : start_idx + batch_size]
                query_batch = queries_embeddings[batch_indices]
                batch_metadata = [query_metadata_list[i] for i in batch_indices]
                
                distances, indices = faiss_index.search(query_batch, k)
                candidate_idxs = set(indices.flatten())
                candidate_idxs.discard(-1)
                
                candidate_doc_ids = {idx: vectorstore_searched.index_to_docstore_id[idx] for idx in candidate_idxs}
                candidate_docs = {
                    doc_id: vectorstore_searched.docstore.search(doc_id)
                    for doc_id in candidate_doc_ids.values()
                }

                for i, idx_list in enumerate(indices):
                    global_i = batch_indices[i]
                    factset_id_query, l6_id = batch_metadata[i]

                    for idx in idx_list:
                        if idx == -1:
                            continue

                        doc_id = candidate_doc_ids.get(idx)
                        doc = candidate_docs.get(doc_id)

                        if doc is not None and doc.metadata.get("FACTSET_ENTITY_ID") == factset_id_query:
                            revere_l6_id = doc.metadata.get("REVENUE_L6_ID")
                            mapping[(factset_id_query, l6_id)] = (factset_id_query, revere_l6_id)
                            unmapped_indices.remove(global_i)
                            break

            if unmapped_indices:
                if k > max_k:
                    logger.warning("最大kに到達しました。ループを終了します。")
                    break
                k *= increment_factor

        logger.info(f"マッピング完了: {len(mapping)} 件 / {N} 件")

        # DataFrameに変換
        mapping_df = pd.DataFrame.from_dict(mapping, orient='index').reset_index()
        mapping_df[['FACTSET_ENTITY_ID', 'PRODUCT_L6_ID']] = pd.DataFrame(mapping_df['index'].tolist(), index=mapping_df.index)
        mapping_df['RELABEL_L6_ID'] = mapping_df[1]
        mapping_df = mapping_df.drop(columns=['index', 0])[['FACTSET_ENTITY_ID', 'PRODUCT_L6_ID', 'RELABEL_L6_ID']]

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
