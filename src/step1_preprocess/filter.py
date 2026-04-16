"""
Step 1: フィルタリング

総評価数が一定数以上のノートのみを分析対象として絞り込む。
"""

import pandas as pd


def filter_by_rating_count(
    df: pd.DataFrame,
    min_count: int = 20,
) -> pd.DataFrame:
    """
    総評価数がmin_count以上のノートのみに絞り込む。
    """
    counts = df.groupby("noteId").size()
    valid_notes = counts[counts >= min_count].index
    filtered = df[df["noteId"].isin(valid_notes)]
    print(f"    filter >= {min_count} ratings: {len(valid_notes):,} notes, {len(filtered):,} rows")
    return filtered
