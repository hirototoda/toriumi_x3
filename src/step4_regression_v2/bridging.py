"""
Step 4 v2: Bridging score 計算

X 公開 MF アルゴリズム (Community Notes scorer) の noteScore = `coreNoteIntercept`
は配布されていないため、その本質である「polarity が異なる評価者からも Helpful を
もらえているか (bridging)」を、既に持っている polarity_df から自前で近似する。

定義 (シンプル版):
    bridging_score(note) = Var(polarity_x) + Var(polarity_y)
                           over the raters who rated the note

これは src/step3_burst/classify_burst.py の polarity_variance と同じ式だが、
向こうは「バースト中の評価者」だけを見るのに対し、こちらはノートに対する
**全評価者** を対象にするのが違い。

直感:
  - bridging_score 大 → 多様な polarity の評価者が集まっている (= MF的にも intercept が高くなりやすい)
  - bridging_score 小 → 同じ polarity の評価者が集中 (= MF的にも intercept が伸びにくい)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_bridging_score(
    ratings_df: pd.DataFrame,
    polarity_df: pd.DataFrame,
) -> pd.Series:
    """各ノートの bridging_score を計算する。

    Parameters
    ----------
    ratings_df : pd.DataFrame
        columns: noteId, raterParticipantId
    polarity_df : pd.DataFrame
        columns: raterParticipantId, polarity_x, polarity_y

    Returns
    -------
    pd.Series
        index=noteId, name='bridging_score'
        polarity 不明な評価者しかいないノートは NaN になる (回帰側で dropna)
    """
    if ratings_df.empty or polarity_df.empty:
        return pd.Series(dtype=float, name="bridging_score")

    pol = polarity_df[["raterParticipantId", "polarity_x", "polarity_y"]].drop_duplicates(
        "raterParticipantId"
    )
    merged = ratings_df[["noteId", "raterParticipantId"]].merge(
        pol, on="raterParticipantId", how="inner",
    )
    if merged.empty:
        return pd.Series(dtype=float, name="bridging_score")

    grouped = merged.groupby("noteId")
    var_x = grouped["polarity_x"].var(ddof=0)
    var_y = grouped["polarity_y"].var(ddof=0)
    n = grouped.size()

    # 評価者 1 人だと var=NaN になる。サンプル不足ノートは NaN のままにしておく。
    bridging = var_x + var_y
    bridging[n < 2] = np.nan
    bridging.name = "bridging_score"

    n_valid = bridging.notna().sum()
    print(f"    bridging_score computed for {len(bridging):,} notes "
          f"({n_valid:,} valid, mean={bridging.mean():.4f})")
    return bridging
