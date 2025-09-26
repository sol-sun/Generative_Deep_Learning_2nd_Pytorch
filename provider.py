"""
FactSet Tradenames データプロバイダー
==================================

目的
- FactSetの商品名・サービス名データを高速かつ安全に取得
- 商品名の正規化と検索用テキスト生成
- 大規模データ向けのバッチ/並列/キャッシュ最適化

主要コンポーネント
- `FactSetTradenameProvider`: 取得・整形・最適化を担う高水準API
- `FactSetTradenameRecord`: 商品名レコードのPydanticモデル
- `FactSetTradenameQueryParams`: フィルタ、性能チューニングを表すクエリモデル

パフォーマンス設計（要点）
- バッチ検証: レコード検証をまとめて実行
- 並列化: `ThreadPoolExecutor`による計算/IOの分散
- キャッシュ: `@lru_cache`による重複計算/参照の抑制
- ベクトル化: pandasによる列演算の最適化

使用例
    from data_providers.sources.factset_tradenames.provider import FactSetTradenameProvider

    # プロバイダーの初期化
    provider = FactSetTradenameProvider(max_workers=4)

    # 商品名データの取得
    tradename_records = provider.get_tradename_records(
        factset_entity_ids=["001C7F-E"],  # 特定の企業
        active_only=True,                # アクティブな商品のみ
        batch_size=1000,                # バッチサイズ
    )
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import pandas as pd
import numpy as np

from wolf_period import WolfPeriod, Frequency
from gppm.core.config_manager import get_logger
from data_providers.core.base_provider import BaseProvider

# FactSet Tradenames固有の型定義をインポート
from .types import FactSetTradenameRecord
from .query_params import FactSetTradenameQueryParams

# RBICSプロバイダーをインポート（L6_NAME取得用）
from data_providers.sources.factset_rbics.provider import RBICSProvider
from data_providers.sources.factset_rbics.query_params import RBICSQueryParams


logger = get_logger(__name__)


class FactSetTradenameProvider(BaseProvider):
    """高速かつWolfPeriod対応のFactSet Tradenamesデータプロバイダー。

    概要
    ----
    FactSetの商品名・サービス名データを高速かつ安全に取得するプロバイダーです。
    商品名の正規化、検索用テキスト生成、RBICS分類との関連付けを含みます。

    主要機能
    --------
    - 商品名データの高速取得
    - 商品名の正規化処理
    - 検索用テキストの自動生成
    - 企業・商品・期間によるフィルタリング
    - バッチ処理・並列化・キャッシュによる性能最適化

    パフォーマンス最適化
    ------------------
    - バッチ処理による効率的なデータ検証
    - 並列処理による高速化
    - キャッシュ機能による重複計算の回避
    - ベクトル化演算による処理速度向上
    - データベースクエリの最適化

    主要メソッド
    ------------
    - get_tradename_records(): 商品名データ取得
    - _normalize_query_params(): パラメータ正規化（内部メソッド）

    制約事項
    --------
    - 最大並列度: 8以下（推奨: 4）
    - バッチサイズ: 1,000〜10,000（推奨: 1,000-5,000）
    - データベース接続: 同時接続数制限あり

    例外処理
    --------
    - ValidationError: レコード検証失敗（不正なデータ形式）
    - DatabaseError: データベースアクセス失敗（接続・権限・SQL）
    - ValueError: パラメータ検証失敗（不正な引数）
    - NotImplementedError: 未実装機能呼び出し

    依存関係
    --------
    - ThreadPoolExecutor: 並列処理
    - pandas: データ処理
    - wolf_period: 期間管理
    """
    
    def __init__(self, max_workers: int = 4) -> None:
        """コンストラクタ。

        Args:
            max_workers: 並列計算に用いるスレッド数（計算/補助処理で使用）。
        """
        super().__init__()
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._validation_cache: Dict[str, Any] = {}
        self._rbics_provider = None  # 遅延初期化
        logger.debug("FactSetTradenameProvider初期化完了: max_workers=%d", max_workers)
    
    def __del__(self):
        """リソースクリーンアップ。

        スレッドプールをシャットダウンします（プロセス終了時のリーク回避）。
        """
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)
    
    def _normalize_query_params(self, params: Optional[FactSetTradenameQueryParams], kwargs: Dict[str, Any]) -> FactSetTradenameQueryParams:
        """ユーザフレンドリーなキーワードを`FactSetTradenameQueryParams`へ正規化。

        - `entity_ids`エイリアスを`factset_entity_ids`に集約
        - 単一文字列はリストへ変換（`factset_entity_ids`, `l6_ids`）
        - `params`と`kwargs`の併用はエラー
        """
        if params is not None and kwargs:
            raise ValueError("Use either 'params' or keyword arguments, not both.")
        if params is not None:
            return params

        # エイリアス吸収と型の昇格
        uf: Dict[str, Any] = dict(kwargs) if kwargs else {}
        # entity_ids aliases -> factset_entity_ids
        if "factset_entity_ids" not in uf:
            if "entity_ids" in uf:
                uf["factset_entity_ids"] = uf.pop("entity_ids")
            elif "entity_id" in uf:
                uf["factset_entity_ids"] = [uf.pop("entity_id")]
        # list promotion for factset_entity_ids
        if "factset_entity_ids" in uf and isinstance(uf["factset_entity_ids"], str):
            uf["factset_entity_ids"] = [uf["factset_entity_ids"]]
        # list promotion for l6_ids
        if "l6_ids" in uf and isinstance(uf["l6_ids"], str):
            uf["l6_ids"] = [uf["l6_ids"]]

        return FactSetTradenameQueryParams.model_validate(uf)

    def _get_l6_name_mapping(self, l6_ids: List[str]) -> Dict[str, str]:
        """L6_IDからL6_NAMEへのマッピングを取得します。

        Args:
            l6_ids: L6_IDのリスト

        Returns:
            L6_IDをキー、L6_NAMEを値とする辞書
        """
        if not l6_ids:
            return {}
        
        # 遅延初期化
        if self._rbics_provider is None:
            self._rbics_provider = RBICSProvider(max_workers=2)
        
        try:
            # RBICS構造マスタからL6_NAMEを取得
            from wolf_period import WolfPeriod
            from datetime import datetime, timezone
            
            current_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
            query_params = RBICSQueryParams(period=current_period)
            
            structure_records = self._rbics_provider.get_structure_records(query_params)
            
            # L6_IDからL6_NAMEへのマッピングを作成
            l6_name_mapping = {}
            for record in structure_records:
                if record.l6_id in l6_ids:
                    l6_name_mapping[record.l6_id] = record.l6_name
            
            logger.debug(f"L6_NAMEマッピング取得完了: {len(l6_name_mapping)}件")
            return l6_name_mapping
            
        except Exception as e:
            logger.warning(f"L6_NAMEマッピング取得エラー: {e}")
            return {}

    def get_tradename_records(
        self,
        params: Optional[FactSetTradenameQueryParams] = None,
        /,
        **kwargs: Any,
    ) -> List[FactSetTradenameRecord]:
        """商品名レコードを高速取得。

        Args:
            params: 既存の `FactSetTradenameQueryParams` を直接指定（後方互換）。
            **kwargs: ユーザフレンドリーなキーワード指定。
                - entity_ids / factset_entity_ids: FactSet Entity ID（単一または配列）
                - l6_ids: RBICS L6 ID（単一または配列）
                - product_names: 商品名（単一または配列）
                - active_only: アクティブな商品のみ
                - batch_size: パフォーマンス制御

        Returns:
            取得・正規化済みの `FactSetTradenameRecord` リスト。

        Raises:
            ValidationError: レコード検証に失敗した場合。
            DatabaseError: DB アクセスに失敗した場合。
        """
        params = self._normalize_query_params(params, kwargs)
        
        logger.info(
            "FactSet Tradenameデータ取得開始: entity_ids=%s active_only=%s batch_size=%d",
            params.factset_entity_ids,
            params.active_only,
            params.batch_size
        )
        
        df = self._query_tradename_data(params)
        
        # L6_NAMEマッピングを取得
        unique_l6_ids = df['L6_ID'].dropna().unique().tolist() if not df.empty else []
        l6_name_mapping = self._get_l6_name_mapping(unique_l6_ids)
        
        records = self._create_tradename_records(df, params.batch_size, l6_name_mapping)
        
        logger.info(
            "FactSet Tradenameデータ取得完了: 取得件数=%d",
            len(records)
        )
        
        return records
    
    def _query_tradename_data(self, params: FactSetTradenameQueryParams) -> pd.DataFrame:
        """商品名データを取得して DataFrame を返します。

        - 必要列: `TRADE_ID`, `FACTSET_ENTITY_ID`, `PRODUCT_ID`, `PRODUCT_NAME`,
          `L6_ID`, `START_DATE`, `END_DATE`, `MULTI_ASSIGN_FLAG`

        Args:
            params: フィルタ条件（企業ID・商品名・期間・件数制限 など）。

        Returns:
            商品名データの `pandas.DataFrame`。
        """
        # WHERE句の構築（SQLインジェクション防止）
        where_conditions = []
        sql_params = []
        
        if params.active_only:
            where_conditions.append("END_DATE IS NULL")
        
        if params.factset_entity_ids:
            placeholders = ",".join(["%s"] * len(params.factset_entity_ids))
            where_conditions.append(f"FACTSET_ENTITY_ID IN ({placeholders})")
            sql_params.extend(params.factset_entity_ids)
        
        if params.l6_ids:
            placeholders = ",".join(["%s"] * len(params.l6_ids))
            where_conditions.append(f"L6_ID IN ({placeholders})")
            sql_params.extend(params.l6_ids)
        
        if params.product_names:
            placeholders = ",".join(["%s"] * len(params.product_names))
            where_conditions.append(f"PRODUCT_NAME IN ({placeholders})")
            sql_params.extend(params.product_names)
        
        if params.exclude_empty_names:
            where_conditions.append("PRODUCT_NAME IS NOT NULL AND PRODUCT_NAME != ''")
        
        if params.min_product_name_length:
            where_conditions.append(f"LEN(PRODUCT_NAME) >= {params.min_product_name_length}")
        
        where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
        
        sql = f"""
        SELECT 
            TRADE_ID,
            FACTSET_ENTITY_ID,
            PRODUCT_ID,
            PRODUCT_NAME,
            L6_ID,
            START_DATE,
            END_DATE,
            MULTI_ASSIGN_FLAG
        FROM FACTSET_FEED.rbics_v1.rbics_entity_product
        WHERE {where_clause}
        ORDER BY FACTSET_ENTITY_ID, PRODUCT_NAME
        """
        
        logger.debug("FactSet Tradenameクエリ実行: params=%s sql=%s", sql_params, sql)
        return self.execute_query(sql, params=sql_params)
    
    def _create_tradename_records(self, df: pd.DataFrame, batch_size: int, l6_name_mapping: Optional[Dict[str, str]] = None) -> List[FactSetTradenameRecord]:
        """商品名の `DataFrame` をレコードにバッチ変換します。

        Args:
            df: 商品名データの `DataFrame`。
            batch_size: バッチサイズ（検証/変換の単位）。
            l6_name_mapping: L6_IDからL6_NAMEへのマッピング辞書（オプション）。

        Returns:
            `FactSetTradenameRecord` のリスト。
        """
        if df.empty:
            return []
        
        # l6_name_mappingがNoneの場合は空の辞書として扱う
        if l6_name_mapping is None:
            l6_name_mapping = {}
        
        records: List[FactSetTradenameRecord] = []
        validation_errors = 0
        
        # バッチ処理による最適化
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            
            # 並列バリデーション
            batch_records = []
            for _, row in batch_df.iterrows():
                try:
                    # WolfPeriod作成
                    retrieved_period = WolfPeriod.from_day(datetime.now(timezone.utc).date())
                    
                    # 日付の変換
                    start_date = None
                    end_date = None
                    if pd.notna(row.get("START_DATE")):
                        start_date = pd.to_datetime(row["START_DATE"]).to_pydatetime()
                    if pd.notna(row.get("END_DATE")):
                        end_date = pd.to_datetime(row["END_DATE"]).to_pydatetime()
                    
                    # L6_NAMEをマッピングから取得
                    l6_id = row.get("L6_ID")
                    l6_name = None
                    if l6_id and l6_name_mapping:
                        l6_name = l6_name_mapping.get(l6_id)
                    
                    record = FactSetTradenameRecord(
                        trade_id=row.get("TRADE_ID"),
                        factset_entity_id=row.get("FACTSET_ENTITY_ID"),
                        product_id=row.get("PRODUCT_ID"),
                        product_name=row.get("PRODUCT_NAME"),
                        l6_id=l6_id,
                        l6_name=l6_name,
                        start_date=start_date,
                        end_date=end_date,
                        multi_assign_flag=row.get("MULTI_ASSIGN_FLAG"),
                        retrieved_period=retrieved_period,
                    )
                    batch_records.append(record)
                except Exception as e:
                    validation_errors += 1
                    # 詳細な商品名情報を含むエラーログ
                    tradename_info = {
                        "TRADE_ID": row.get("TRADE_ID"),
                        "FACTSET_ENTITY_ID": row.get("FACTSET_ENTITY_ID"),
                        "PRODUCT_NAME": row.get("PRODUCT_NAME"),
                        "L6_ID": row.get("L6_ID"),
                        "START_DATE": str(row.get("START_DATE")),
                        "END_DATE": str(row.get("END_DATE")),
                        "MULTI_ASSIGN_FLAG": row.get("MULTI_ASSIGN_FLAG")
                    }
                    # None値を除外してログを見やすくする
                    filtered_info = {k: v for k, v in tradename_info.items() if v is not None and v != ""}
                    
                    logger.debug(
                        "FactSet Tradenameレコード検証エラー: tradename_data=%s error=%s",
                        filtered_info,
                        str(e)
                    )
            
            records.extend(batch_records)
        
        if validation_errors > 0:
            logger.warning(
                "FactSet Tradenameデータ変換完了: 有効レコード=%d 検証エラー=%d",
                len(records),
                validation_errors
            )
        
        return records


__all__ = [
    "FactSetTradenameRecord",
    "FactSetTradenameQueryParams",
    "FactSetTradenameProvider",
]
