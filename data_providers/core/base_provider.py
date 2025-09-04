"""
データプロバイダー基底クラス
==========================

データプロバイダー間で共通して使用されるデータ処理機能を提供する基底クラス。

主な機能:
- データベース接続機能 (GibDB継承)
- テキスト正規化処理
- 重複データフィルタリング
- 会計期間調整
- 重複期間除去

使用例:
    from data_providers.core.base_provider import BaseProvider
    
    # 基底クラスとして継承
    class MyProvider(BaseProvider):
        def __init__(self):
            super().__init__()  # データベース接続を初期化
            # 独自の処理を追加
    
    # スタティックメソッドとして直接使用
    normalized_text = BaseProvider.normalize_text("Sample Text")
    
    # 会計期間調整
    df_adjusted = BaseProvider.adjust_fiscal_term(df, date_col="DATE")
    
    # 重複期間除去
    df_clean = BaseProvider.remove_duplicate_periods(df)

設計思想:
- シンプルで理解しやすい実装
- 元のコードの機能を完全に保持
- 拡張性よりも安定性を重視
- データプロバイダー間の共通処理を集約

注意事項:
- このクラスは継承して使用することを前提としています
- データベース接続は GibDB を使用します
- 会計期間の調整は日本の会計年度に基づきます
"""

import pandas as pd
import unicodedata
from data_providers.database.db_connections import GibDB


class BaseProvider(GibDB):
    """
    データプロバイダー基底クラス
    
    概要:
    データプロバイダー間で共通して使用されるデータ処理機能を提供する基底クラス。
    データベース接続、テキスト正規化、重複データ処理、会計期間調整などの
    共通機能を集約しています。
    
    特徴:
    - GibDBを継承したデータベース接続機能
    - テキスト正規化機能 (Unicode正規化)
    - 重複データフィルタリング機能
    - 会計期間調整機能 (日本基準)
    - 重複期間除去機能
    
    継承方法:
        class MyDataProvider(BaseProvider):
            def __init__(self):
                super().__init__()  # データベース接続を初期化
                # 独自の処理を追加
    
    スタティックメソッド:
        - normalize_text(): テキスト正規化
        - filter_func(): 重複FSYM_IDフィルタリング
        - adjust_fiscal_term(): 会計期間調整
        - remove_duplicate_periods(): 重複期間除去
    """
    
    def __init__(self):
        """
        データプロセッサーの初期化
        
        GibDBを継承しているため、データベース接続が自動的に初期化されます。
        継承クラスでは super().__init__() を呼び出してから
        独自の初期化処理を実装してください。
        """
        super().__init__()
    
    @staticmethod
    def normalize_text(s):
        """
        テキストの正規化処理
        
        Unicode正規化 (NFKC) と小文字化を適用してテキストを正規化します。
        データベース検索や比較処理で使用する際の一貫性を保つために使用します。
        
        Args:
            s: 正規化対象の文字列
            
        Returns:
            正規化された文字列 (入力が文字列でない場合はそのまま返却)
            
        Example:
            >>> DataProcessor.normalize_text("Ａｐｐｌｅ Inc.")
            'apple inc.'
        """
        if isinstance(s, str):
            s_lower = s.lower()
            return unicodedata.normalize('NFKC', s_lower)
        return s
    
    @staticmethod
    def filter_func(group):
        """
        重複FSYM_IDのフィルタリング処理
        
        同一グループ内でFSYM_IDが複数存在する場合、
        PRIMARY_EQUITY_FLAG=1のレコードのみを残します。
        データの重複を解決するために使用します。
        
        Args:
            group: pandas DataFrame のグループ (groupby操作の結果)
            
        Returns:
            フィルタリングされたDataFrame
            
        Note:
            PRIMARY_EQUITY_FLAG=1は主力証券を示すフラグです。
        """
        if group["FSYM_ID"].nunique() > 1:
            return group[group["PRIMARY_EQUITY_FLAG"] == 1]
        else:
            return group
    
    @staticmethod
    def adjust_fiscal_term(df: pd.DataFrame, date_col: str = "DATE") -> pd.DataFrame:
        """
        会計期間の調整処理
        
        日付データから会計期間 (FTERM_2) と会計年度 (FISCAL_YEAR) を計算します。
        日本の会計年度基準 (4月開始) に基づいて処理を行います。
        
        Args:
            df: 処理対象のDataFrame
            date_col: 日付カラム名 (デフォルト: "DATE")
            
        Returns:
            会計期間情報が追加されたDataFrame
            
        Note:
            - FTERM_2: YYYYMM形式の会計期間
            - FISCAL_YEAR: 会計年度
            - 日本の会計年度基準 (4月開始) を使用
        """
        _MONTH_MAP = {1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6, 7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12}
        df = df.copy()
        df["FTERM"] = pd.to_datetime(df[date_col])
        df["FTERM_2"] = (
            df["FTERM"].dt.year.astype(str) + df["FTERM"].dt.month.map(lambda m: "{:02}".format(_MONTH_MAP[m]))
        ).astype(int)
        from wolf_period import WolfPeriod, Frequency
        
        def get_fiscal_year_from_yyyymm(yyyymm):
            """
            YYYYMM形式から会計年度を取得
            
            WolfPeriodライブラリを使用してYYYYMM形式の期間から
            日本の会計年度基準 (4月開始) に基づく会計年度を計算します。
            
            Args:
                yyyymm: YYYYMM形式の期間 (例: 202303)
                
            Returns:
                会計年度 (例: 2023)
            """
            # 月次期間を作成してfiscal_yearプロパティを直接使用
            period = WolfPeriod.from_yyyymm(yyyymm, freq=Frequency.M)
            return period.fiscal_year
        
        df["FISCAL_YEAR"] = df["FTERM_2"].map(get_fiscal_year_from_yyyymm)
        return df
    
    @staticmethod
    def remove_duplicate_periods(df: pd.DataFrame, 
                               group_cols: list = None, 
                               date_col: str = "DATE",
                               period_col: str = "FTERM_2") -> pd.DataFrame:
        """
        重複期間データの除去処理
        
        指定されたグループカラムと期間カラムで重複するデータを検出し、
        最新の日付のレコードのみを残します。データの整合性を保つために使用します。
        
        Args:
            df: 処理対象のDataFrame
            group_cols: グループ化するカラムのリスト (デフォルト: ["FSYM_ID", "FACTSET_ENTITY_ID"])
            date_col: 日付カラム名 (デフォルト: "DATE")
            period_col: 期間カラム名 (デフォルト: "FTERM_2")
            
        Returns:
            重複が除去されたDataFrame
            
        Note:
            - 重複データが見つからない場合は元のDataFrameを返却
            - グループ化カラムが存在しない場合は警告を出力
            - 除去されたレコード数がコンソールに出力されます
        """
        if group_cols is None:
            group_cols = ["FSYM_ID", "FACTSET_ENTITY_ID"]
        available_group_cols = [col for col in group_cols if col in df.columns]
        if not available_group_cols:
            print("警告: 指定されたグループカラムが見つかりません。FSYM_IDのみを使用します。")
            available_group_cols = ["FSYM_ID"] if "FSYM_ID" in df.columns else []
        if not available_group_cols:
            print("エラー: グループ化できるカラムがありません。")
            return df
        df_result = df.copy()
        df_result[date_col] = pd.to_datetime(df_result[date_col])
        duplicate_groups = (
            df_result.groupby(available_group_cols + [period_col])
            .size()
            .reset_index(name='count')
            .query('count > 1')
        )
        if len(duplicate_groups) == 0:
            print("重複する会計期間データは見つかりませんでした。")
            return df_result
        print(f"重複データが見つかりました: {len(duplicate_groups)} グループ")
        df_result = (
            df_result.sort_values(available_group_cols + [period_col, date_col])
            .groupby(available_group_cols + [period_col], group_keys=False)
            .tail(1)
            .reset_index(drop=True)
        )
        removed_count = len(df) - len(df_result)
        print(f"重複除去完了: {removed_count} 件のレコードを除去しました。")
        return df_result