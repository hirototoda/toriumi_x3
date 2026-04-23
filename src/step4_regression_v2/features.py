"""
Step 4 v2: 特徴量計算 (control 拡張版)

既存 src/step4_regression/features.py に対して以下を追加:
  - ratings_count   : 各ノートの評価総数 (人気度の control)
  - bridging_score  : 評価者 polarity 多様性 (CN MF noteScore の自前近似)

bridging_score は polarity_df から compute_bridging_score() で計算する。
評価者 1 人未満で polarity 計算不能なノートは NaN のままにする
(M2 の dropna でそのノートだけ落ちる)。
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.step4_regression.features import compute_trend
from src.step4_regression_v2.bridging import compute_bridging_score


def _ratings_count_per_note(ratings_df: pd.DataFrame) -> pd.Series:
    return ratings_df.groupby("noteId").size().rename("ratings_count")


def compute_features_for_regression_v2(
    ratings_df: pd.DataFrame,
    burst_df: pd.DataFrame,
    history_df: pd.DataFrame,
    quality: pd.Series,
    polarity_df: Optional[pd.DataFrame] = None,
    trend_min_evals: int = 4,
) -> pd.DataFrame:
    """v2: ロジスティック回帰用の特徴量 DataFrame を構築する。

    Returns
    -------
    pd.DataFrame
        columns: noteId, deleted, type_a, type_b, trend, quality,
                 ratings_count, bridging_score
    """
    note_ids = ratings_df["noteId"].unique()

    status = history_df.drop_duplicates("noteId").set_index("noteId")["currentStatus"]

    if not burst_df.empty:
        burst_flags = burst_df.groupby("noteId")["burst_type"].first()
    else:
        burst_flags = pd.Series(dtype=str, name="burst_type")

    trend = compute_trend(ratings_df, min_evals=trend_min_evals)
    rcount = _ratings_count_per_note(ratings_df)

    if polarity_df is not None and not polarity_df.empty:
        bridging = compute_bridging_score(ratings_df, polarity_df)
    else:
        bridging = pd.Series(dtype=float, name="bridging_score")

    valid_statuses = {"CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"}

    rows = []
    for nid in note_ids:
        s = status.get(nid, "UNKNOWN")
        if s not in valid_statuses:
            continue
        rows.append({
            "noteId": nid,
            "deleted": 0 if s == "CURRENTLY_RATED_HELPFUL" else 1,
            "type_a": 1 if burst_flags.get(nid) == "A" else 0,
            "type_b": 1 if burst_flags.get(nid) == "B" else 0,
            "trend": trend.get(nid, 0.0),
            "quality": quality.get(nid, 0.0) if quality is not None else 0.0,
            "ratings_count": int(rcount.get(nid, 0)),
            "bridging_score": bridging.get(nid, np.nan) if not bridging.empty else np.nan,
        })

    cols = ["noteId", "deleted", "type_a", "type_b", "trend", "quality",
            "ratings_count", "bridging_score"]
    if not rows:
        feat_df = pd.DataFrame(columns=cols)
        print("    features_v2: 0 notes (no notes with definitive status)")
        return feat_df

    feat_df = pd.DataFrame(rows, columns=cols)
    n_bridge = feat_df["bridging_score"].notna().sum()
    print(
        f"    features_v2: {len(feat_df):,} notes, "
        f"deleted={feat_df['deleted'].sum()}, helpful={(feat_df['deleted']==0).sum()}, "
        f"bridging available={n_bridge:,}"
    )
    return feat_df
