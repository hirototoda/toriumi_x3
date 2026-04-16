"""
Step 3: バースト分類（TypeA / TypeB）

バースト内の評価者のpolarity分散で分類する。
- Type A（陣営反応）: polarity分散が小さい → 同じ陣営が集中
- Type B（自然拡散）: polarity分散が大きい → 多様な立場が自然に集まる

分散の定義: Var(polarity_x) + Var(polarity_y)  （共分散行列のトレース）
"""

import numpy as np
import pandas as pd


def classify_burst_type(
    burst_df: pd.DataFrame,
    polarity_df: pd.DataFrame,
    threshold: float | None = None,
) -> pd.DataFrame:
    """
    バーストをTypeA/TypeBに分類する。

    Parameters
    ----------
    burst_df : pd.DataFrame
        detect_bursts() の出力。burst_raters カラムを含む。
    polarity_df : pd.DataFrame
        compute_polarity() の出力。raterParticipantId, polarity_x, polarity_y。
    threshold : float or None
        分散の閾値。None なら中央値で自動決定。

    Returns
    -------
    pd.DataFrame
        burst_type ("A" or "B") と polarity_variance カラムを追加
    """
    if burst_df.empty:
        burst_df = burst_df.copy()
        burst_df["polarity_variance"] = pd.Series(dtype=float)
        burst_df["burst_type"] = pd.Series(dtype=str)
        return burst_df

    pol_map_x = polarity_df.set_index("raterParticipantId")["polarity_x"]
    pol_map_y = polarity_df.set_index("raterParticipantId")["polarity_y"]

    variances = []
    for _, row in burst_df.iterrows():
        raters = row["burst_raters"]
        px = [pol_map_x.get(r) for r in raters]
        py = [pol_map_y.get(r) for r in raters]
        # polarity が不明な評価者は除外
        px = [v for v in px if v is not None and not np.isnan(v)]
        py = [v for v in py if v is not None and not np.isnan(v)]

        if len(px) >= 2:
            var = np.var(px) + np.var(py)
        else:
            var = np.nan
        variances.append(var)

    result = burst_df.copy()
    result["polarity_variance"] = variances
    valid = result["polarity_variance"].dropna()

    if threshold is None and len(valid) > 0:
        threshold = valid.median()

    result["burst_type"] = np.where(
        result["polarity_variance"].isna(), "B",
        np.where(result["polarity_variance"] <= threshold, "A", "B"),
    )

    n_a = (result["burst_type"] == "A").sum()
    n_b = (result["burst_type"] == "B").sum()
    print(f"    TypeA: {n_a}, TypeB: {n_b} (threshold={threshold:.4f})")
    return result
