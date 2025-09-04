"""
データベース接続共通モジュール(Microsoft SQL Server)

Microsoft SQL Serverへ接続して、SQL結果をDataFrameで返す。

主な機能：
- SQL Serverへの接続と切断
- クエリ実行結果のDataFrame変換
- Decimal型データの自動数値変換（NUMERIC, DECIMAL, MONEY型等に対応）

使用例：
    # 通常の使用（Decimal型は自動的にfloat/intに変換される）
    df = db.execute_query("SELECT * FROM table")
    
    # Decimal型の変換を無効化する場合
    df = db.execute_query("SELECT * FROM table", convert_decimal=False)
"""

import traceback
from abc import ABCMeta
from decimal import Decimal
from typing import Any, Optional, List

import pandas as pd
import pymssql
import numpy as np

from gppm.core.config_manager import get_logger


class SQLServer(metaclass=ABCMeta):
    """ Microsoft SQL Serverクラス

    :param str host: host
    :param str username: ユーザ名
    :param str password: パスワード
    """

    def __init__(self, host: str, port: str, username: str = None, password: str = None):
        """ コンストラクタ """
        if type(self) == SQLServer:
            raise TypeError("Can't instantiate abstract class.")

        try:
            # connection params
            self._host = host
            self._port = port
            self._uid = username
            self._pwd = password

        except Exception as _ex:
            traceback.print_exc()
            raise _ex
        finally:
            pass

        # ロガー設定
        self.logger = get_logger(__name__)

    def __del__(self):
        """ デストラクタ """
        pass

    def _connect(self):
        """ コネクション接続 """
        try:
            _conn = pymssql.connect(
                host=self._host,
                port=self._port,
                user=self._uid,
                password=self._pwd,
            )            
        except Exception as _ex:
            traceback.print_exc()
            raise _ex
        return _conn
    
    def _disconnect(self, cursor, conn):
        """ コネクション切断 """
        if cursor:
            cursor.close()
        if conn:
            conn.close()

    def _convert_decimal_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decimal型のカラムを数値型に変換
        
        SQL Serverから返されるDecimal型データを適切な数値型（float64）に変換します。
        NUMERIC, DECIMAL, MONEY型などがPythonのdecimal.Decimal型として返される場合に対応。
        
        Args:
            df: 変換対象のDataFrame
            
        Returns:
            変換後のDataFrame
        """
        if df.empty:
            return df
        
        # 各カラムをチェックして変換
        for col in df.columns:
            # カラムにDecimal型が含まれているかチェック
            has_decimal = df[col].dropna().map(lambda x: isinstance(x, Decimal)).any()
            
            # Decimal型が検出された場合は変換
            if has_decimal:
                try:
                    # Decimal型のみをfloatに変換し、その他の値は保持
                    df[col] = df[col].apply(
                        lambda x: float(x) if isinstance(x, Decimal) else x
                    )
                    
                    # 変換後のカラムを数値型に変換（可能な場合のみ）
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                        
                except Exception as e:
                    # 変換に失敗した場合はそのまま残す
                    self.logger.warning(f"Could not convert column '{col}' from Decimal: {e}")
                        
        return df

    def _format_sql_for_debug(self, sql: str, params: Optional[List[Any]] = None) -> str:
        """
        デバッグ用にSQLクエリにパラメータを反映した文字列を生成
        
        Args:
            sql: 元のSQL文（%sプレースホルダー含む）
            params: SQLパラメータのリスト
            
        Returns:
            パラメータが反映されたSQL文字列
            
        Note:
            あくまでデバッグ用のため、完全なSQL文法の正確性は保証しません
        """
        if params is None or len(params) == 0:
            return sql
        
        debug_sql = sql
        try:
            # パラメータを文字列として置換
            formatted_params = []
            for param in params:
                if param is None:
                    formatted_params.append('NULL')
                elif isinstance(param, str):
                    # 文字列の場合はシングルクォートで囲む
                    escaped_param = param.replace("'", "''")
                    formatted_params.append(f"'{escaped_param}'")
                elif isinstance(param, (int, float, Decimal)):
                    formatted_params.append(str(param))
                else:
                    # その他の型は文字列化してクォートで囲む
                    escaped_param = str(param).replace("'", "''")
                    formatted_params.append(f"'{escaped_param}'")
            
            # %sプレースホルダーを順番に置換
            debug_sql = debug_sql % tuple(formatted_params)
            
        except Exception as e:
            # 置換に失敗した場合は元のSQLとパラメータを別々に表示
            params_str = ', '.join([str(p) for p in params])
            debug_sql = f"{sql}\n-- Parameters: {params_str}"
            
        return debug_sql

    def execute_query(self, sql: str, params: Optional[List[Any]] = None, convert_decimal: bool = True) -> pd.DataFrame:
        """
        検索の実行
        
        Args:
            sql: 実行するSQL文
            params: SQLパラメータのリスト（プレースホルダーを使用する場合）
            convert_decimal: Decimal型を数値型に自動変換するか（デフォルト: True）
            
        Returns:
            クエリ結果のDataFrame
            
        Example:
            # パラメータなしのクエリ
            df = db.execute_query("SELECT * FROM users")
            
            # パラメータ付きのクエリ
            df = db.execute_query("SELECT * FROM users WHERE age > %s", [18])
        """
        _df = pd.DataFrame()

        # 接続
        _conn = self._connect()
        _cursor = None
        
        # クエリ発行
        try:
            _cursor = _conn.cursor()
            
            # パラメータ反映後のSQLクエリをログ出力用に生成
            debug_sql = self._format_sql_for_debug(sql, params)
            self.logger.debug(f"Executing SQL query:\n{debug_sql}")
            
            if params is not None and len(params) > 0:
                _cursor.execute(sql, tuple(params))
            else:
                _cursor.execute(sql)
            _rows = _cursor.fetchall()
            _cols = [i[0].upper() for i in _cursor.description]

            # DataFrame化
            if len(_rows):
                _df = pd.DataFrame(_rows, columns=_cols)
                
                # Decimal型の自動変換（デフォルトで有効）
                if convert_decimal:
                    _df = self._convert_decimal_to_numeric(_df)

        except Exception as _ex:
            self.logger.error(f"Failed to execute query:\n{sql}")
            traceback.print_exc()
            raise _ex
        
        # 切断
        finally:
            self._disconnect(_cursor, _conn)

        return _df