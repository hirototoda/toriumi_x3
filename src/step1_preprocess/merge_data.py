"""
Step 1: 前処理 — 3ファイル結合

公式TSV 3ファイル(notes, ratings, noteStatusHistory)を
noteId をキーとして結合し、分析用の統合DataFrameを作成する。
"""

import pandas as pd


def merge_tsv_files(
    notes_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    3つのDataFrameをnoteIdで結合する。

    Returns
    -------
    pd.DataFrame
        ratings をベースに notes と history を left join したもの
    """
    merged = ratings_df.merge(
        notes_df, on="noteId", how="left", suffixes=("", "_note")
    )
    merged = merged.merge(history_df, on="noteId", how="left")
    print(f"    merged: {len(merged):,} rows")
    return merged
