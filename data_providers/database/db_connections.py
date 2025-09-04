"""
データベース接続管理
==================

複数のデータベース接続を管理するモジュール。
現在はGIBデータベース接続を提供し、将来的に他のデータベース接続も追加予定。

使用例:
    from data_providers.database.db_connections import GibDB
    
    # GIBデータベース接続
    db = GibDB()
    df = db.execute_query("SELECT * FROM table")
"""

from data_providers.database.sqlserver import SQLServer


class GibDB(SQLServer):
    """GIBデータベース接続（読み取り専用）"""
    
    def __init__(self):
        self.host = '172.22.200.25'
        self.port = '1433'
        self.username = 'READ_MASTER'
        self.password = 'MASTER_READ123'
        super().__init__(host=self.host, port=self.port, username=self.username, password=self.password)


# 将来的な拡張例:
# class ProductionDB(SQLServer):
#     """本番環境データベース接続"""
#     def __init__(self):
#         # 本番環境の設定
#         pass
#
# class AnalyticsDB(SQLServer):
#     """分析用データベース接続"""
#     def __init__(self):
#         # 分析用の設定
#         pass
