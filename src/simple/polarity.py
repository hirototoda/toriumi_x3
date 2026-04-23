"""
Polarity 計算 (TruncatedSVD)

各評価者を 2 次元ベクトル (polarity_x, polarity_y) で表現する。
直感: 似た note 群を Helpful/Not Helpful 評価する人は近くに来る。

実装の素直さ:
  scipy.sparse + sklearn.decomposition.TruncatedSVD で完結。
  matrix.toarray() を呼ばないので、サンプリング後でも安全に動く。

循環論法回避:
  各 rater の最初の first_n=50 件だけで polarity を固定する。
  これがないと「ノートが Not Helpful になった後の評価」も polarity に
  混ざってしまい、後段の TypeA/B 判定で目的変数を見ていることになる。
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

LEVEL_MAP = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.0, "NOT_HELPFUL": -1.0}


def compute_polarity(ratings: pd.DataFrame, first_n: int = 50, seed: int = 42) -> pd.DataFrame:
    df = ratings.copy()
    df["score"] = df["helpfulnessLevel"].map(LEVEL_MAP)
    df = df.dropna(subset=["score"])

    # 各 rater の最初の first_n 件のみ採用 (循環論法回避)
    df = df.sort_values("createdAtMillis")
    df["rank"] = df.groupby("raterParticipantId").cumcount()
    df = df[df["rank"] < first_n].drop(columns=["rank"])

    # 評価が少なすぎる rater (< 5 件) は polarity が決まらないので除外
    counts = df.groupby("raterParticipantId").size()
    valid_raters = counts[counts >= 5].index
    df = df[df["raterParticipantId"].isin(valid_raters)]

    if df.empty:
        print("[polarity] no valid raters")
        return pd.DataFrame(columns=["raterParticipantId", "polarity_x", "polarity_y"])

    # rater × note の疎行列を作る
    rater_ids = sorted(df["raterParticipantId"].unique())
    note_ids  = sorted(df["noteId"].unique())
    rater_idx = {r: i for i, r in enumerate(rater_ids)}
    note_idx  = {n: i for i, n in enumerate(note_ids)}

    rows = df["raterParticipantId"].map(rater_idx).values
    cols = df["noteId"].map(note_idx).values
    vals = df["score"].values
    matrix = csr_matrix((vals, (rows, cols)), shape=(len(rater_ids), len(note_ids)))

    # TruncatedSVD で次元 2 に圧縮
    n_comp = min(2, min(matrix.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=seed)
    coords = svd.fit_transform(matrix)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 1))])

    out = pd.DataFrame({
        "raterParticipantId": rater_ids,
        "polarity_x": coords[:, 0],
        "polarity_y": coords[:, 1],
    })
    print(f"[polarity] {len(out):,} raters, explained_variance_ratio={svd.explained_variance_ratio_.round(3).tolist()}")
    return out
