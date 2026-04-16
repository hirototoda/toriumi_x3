"""
Step 5: 「正しいのに消えた」ノートの特定

ターゲット集合:
    {Q上位25%} ∩ {TypeAバーストあり} ∩ {最終ステータス ≠ Helpful}
"""

import pandas as pd


def extract_target_notes(
    df: pd.DataFrame,
    top_percent: int = 25,
) -> pd.DataFrame:
    """
    「正しいのに消えた」ノートのターゲット集合を抽出する。

    Parameters
    ----------
    df : pd.DataFrame
        columns: noteId, quality, type_a, deleted
    top_percent : int
        品質スコア上位N%

    Returns
    -------
    pd.DataFrame
        条件を満たすノート
    """
    q_threshold = df["quality"].quantile(1 - top_percent / 100)
    targets = df[
        (df["quality"] >= q_threshold) &
        (df["type_a"] == 1) &
        (df["deleted"] == 1)
    ].copy()

    print(f"    target notes: {len(targets)} / {len(df)}")
    print(f"      Q >= {q_threshold:.3f} (top {top_percent}%)")
    print(f"      + TypeA burst + not Helpful")
    return targets
