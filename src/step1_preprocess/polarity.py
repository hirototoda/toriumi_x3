"""
Step 1: polarity計算

評価行列にSVDを適用し、各評価者のpolarity vector（2次元）を計算する。
循環論法回避のため、各評価者の最初のfirst_n件の評価でpolarityを固定する。
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


def compute_polarity(
    ratings_df: pd.DataFrame,
    first_n: int = 50,
) -> pd.DataFrame:
    """
    各評価者のpolarity（2Dベクトル）を計算する。

    Parameters
    ----------
    ratings_df : pd.DataFrame
        columns: noteId, raterParticipantId, createdAtMillis, helpfulnessLevel
    first_n : int
        各評価者の最初のN件のみ使用

    Returns
    -------
    pd.DataFrame
        columns: raterParticipantId, polarity_x, polarity_y
    """
    level_map = {
        "HELPFUL": 1.0,
        "SOMEWHAT_HELPFUL": 0.0,
        "NOT_HELPFUL": -1.0,
    }
    df = ratings_df.copy()
    df["score"] = df["helpfulnessLevel"].map(level_map)
    df = df.dropna(subset=["score"])

    # 各評価者の最初の first_n 件に絞る
    df = df.sort_values("createdAtMillis")
    df["rank"] = df.groupby("raterParticipantId").cumcount()
    df = df[df["rank"] < first_n].drop(columns=["rank"])

    # 評価数が少なすぎる評価者を除外（最低5件）
    counts = df.groupby("raterParticipantId").size()
    valid_raters = counts[counts >= 5].index
    df = df[df["raterParticipantId"].isin(valid_raters)]

    # 疎行列構築
    rater_ids = sorted(df["raterParticipantId"].unique())
    note_ids = sorted(df["noteId"].unique())
    rater_idx = {r: i for i, r in enumerate(rater_ids)}
    note_idx = {n: i for i, n in enumerate(note_ids)}

    rows = df["raterParticipantId"].map(rater_idx).values
    cols = df["noteId"].map(note_idx).values
    vals = df["score"].values
    mat = csr_matrix((vals, (rows, cols)), shape=(len(rater_ids), len(note_ids)))

    # 列中心化 → SVD
    col_mean = np.array(mat.mean(axis=0)).flatten()
    mat_centered = csr_matrix(mat.toarray() - col_mean[np.newaxis, :])

    k = min(2, min(mat_centered.shape) - 1)
    u, s, _ = svds(mat_centered, k=k)
    order = np.argsort(-s)
    coords = u[:, order] * s[order][np.newaxis, :]

    result = pd.DataFrame({
        "raterParticipantId": rater_ids,
        "polarity_x": coords[:, 0],
        "polarity_y": coords[:, 1] if k == 2 else 0.0,
    })
    print(f"    polarity computed for {len(result):,} raters")
    return result
