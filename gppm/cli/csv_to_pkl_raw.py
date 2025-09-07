"""
元のCSVからPKL生成コード（raw版）
巨大なCSVファイルからPKLファイルを生成するための元の実装
"""

import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import logging

# ログ設定
logger = logging.getLogger(__name__)


def task(cols, csv_path):
    """CSVの列を処理して統計値を計算するタスク"""
    chunksize = 200
    my_data = pd.read_csv(
        csv_path,
        usecols=cols,
        comment='#',
        skip_blank_lines=True,
        chunksize=chunksize,
        engine='c',
        dtype=float
    )
    col_num = 0
    out = pd.DataFrame()
    for n in my_data:
        col_num += chunksize
        logger.debug('[ %s ]' % col_num)
        out = pd.concat([out, n], axis=1)
    median_df = out.median(axis=1)
    std_df = out.std(axis=1).add_prefix('std_')

    return pd.concat([median_df, std_df], axis=0)


def read_csv_to_dataframe(hoge, p, csv_path):
    """並列処理でCSVを読み込んでDataFrameに変換"""
    sms_multi = pd.Series([], dtype='float64')
    with ProcessPoolExecutor() as e:
        futures = []
        for i in hoge:
            future = e.submit(task, i, csv_path)
            futures.append(future)
        for r in futures:
            sms_multi = pd.concat([sms_multi, r.result()], axis=0)
    return sms_multi


def read_item_roic_result(csv_path: str, out_path: str, regex: str = None):
    """メイン処理：CSVからPKLファイルを生成"""
    test = next(pd.read_csv(
        csv_path,
        comment='#',
        skip_blank_lines=True,
        chunksize=1,
        engine='c',
        dtype=float
    ))
    target_cols = test.columns.tolist()  # [s for s in test.columns.tolist() if e.match(regex, s)]

    hoge = np.array_split(target_cols, 51)
    sms_multi = read_csv_to_dataframe(hoge, p=51, csv_path=csv_path)
    sms_multi.to_pickle(out_path)

    print(sms_multi)


if __name__ == '__main__':
    csv_path = '/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202505/item_roic_rbics/result/Item_ROIC_Item_Beta-20250627033115.csv'
    out_path = '/home/tmiyahara/repos/Neumann-Notebook/tmiyahara/202505/stan_result_15_usd2.pkl'
    regex = "^.*$"

    read_item_roic_result(csv_path, out_path, regex)
