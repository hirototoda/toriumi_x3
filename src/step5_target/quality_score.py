"""
Step 5: 品質スコア Q

helpfulness投票とは独立な品質指標:
    Q = URL数(正規化) + 文字数(正規化)

ドメイン信頼性は notes.tsv だけでは取れないため、最低限実装ではURL数+文字数。
"""

import re

import numpy as np
import pandas as pd


def compute_quality_score(notes_df: pd.DataFrame) -> pd.Series:
    """
    各ノートの品質スコアQを計算する。

    Parameters
    ----------
    notes_df : pd.DataFrame
        noteId, summary カラムを含む

    Returns
    -------
    pd.Series
        noteId をインデックスとする品質スコアQ (0~1)
    """
    df = notes_df.copy()
    text = df["summary"].fillna("")

    # URL数
    url_count = text.apply(lambda t: len(re.findall(r"https?://", t)))
    # 文字数
    char_count = text.str.len()

    # 正規化 (min-max)
    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    q = (_norm(url_count) + _norm(char_count)) / 2.0
    q.index = df["noteId"]
    q.name = "quality"
    print(f"    quality score: mean={q.mean():.3f}, std={q.std():.3f}")
    return q
