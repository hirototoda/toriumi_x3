"""
Step 4: 特徴量計算

ロジスティック回帰の説明変数を構築する。
"""

import numpy as np
import pandas as pd


def compute_trend(ratings_df: pd.DataFrame, min_evals: int = 4) -> pd.Series:
    """
    各ノートの評価推移（Trend）を計算する。
    初期の平均スコアと後期の平均スコアの差。
    正の値 → 後半で評価が上がった、負の値 → 後半で評価が下がった。
    """
    level_map = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.0, "NOT_HELPFUL": -1.0}
    df = ratings_df.copy()
    df["score"] = df["helpfulnessLevel"].map(level_map)
    df = df.dropna(subset=["score"])
    df = df.sort_values("createdAtMillis")

    _min = min_evals

    def _trend(g):
        n = len(g)
        if n < _min:
            return 0.0
        half = n // 2
        return g["score"].iloc[half:].mean() - g["score"].iloc[:half].mean()

    trend = df.groupby("noteId").apply(_trend, include_groups=False)
    trend.name = "trend"
    print(f"    trend computed for {len(trend):,} notes")
    return trend


def compute_features_for_regression(
    ratings_df: pd.DataFrame,
    burst_df: pd.DataFrame,
    history_df: pd.DataFrame,
    quality: pd.Series,
    trend_min_evals: int = 4,
) -> pd.DataFrame:
    """
    ロジスティック回帰用の特徴量DataFrameを構築する。

    Returns
    -------
    pd.DataFrame
        columns: noteId, deleted, type_a, type_b, trend, quality
    """
    # ノート一覧
    note_ids = ratings_df["noteId"].unique()

    # 被説明変数: currentStatus が Helpful でないなら deleted=1
    status = history_df.drop_duplicates("noteId").set_index("noteId")["currentStatus"]

    # バーストタイプフラグ
    if not burst_df.empty:
        burst_flags = burst_df.groupby("noteId")["burst_type"].first()
    else:
        burst_flags = pd.Series(dtype=str, name="burst_type")

    # Trend
    trend = compute_trend(ratings_df, min_evals=trend_min_evals)

    # NEEDS_MORE_RATINGS / UNKNOWN は回帰から除外（判定済みのみ使用）
    valid_statuses = {"CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"}

    rows = []
    for nid in note_ids:
        s = status.get(nid, "UNKNOWN")
        if s not in valid_statuses:
            continue
        deleted = 0 if s == "CURRENTLY_RATED_HELPFUL" else 1
        ta = 1 if burst_flags.get(nid) == "A" else 0
        tb = 1 if burst_flags.get(nid) == "B" else 0
        t = trend.get(nid, 0.0)
        q = quality.get(nid, 0.0) if quality is not None else 0.0
        rows.append({
            "noteId": nid,
            "deleted": deleted,
            "type_a": ta,
            "type_b": tb,
            "trend": t,
            "quality": q,
        })

    feat_df = pd.DataFrame(rows)
    if feat_df.empty:
        feat_df = pd.DataFrame(columns=["noteId", "deleted", "type_a", "type_b", "trend", "quality"])
        print(f"    features: 0 notes (no notes with definitive status)")
    else:
        print(f"    features: {len(feat_df):,} notes, deleted={feat_df['deleted'].sum()}, helpful={(feat_df['deleted']==0).sum()}")
    return feat_df
