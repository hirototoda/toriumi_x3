"""
Step 5: 品質スコアの精度検証

200件を人手/LLMで「高品質/低品質」にラベル付けし、
品質スコアQとの一致率を確認する。
"""

import pandas as pd


def compute_agreement_rate(
    q_scores: pd.Series,
    labels: pd.Series,
) -> float:
    """
    品質スコアQと人手ラベルの一致率を計算する。

    Parameters
    ----------
    q_scores : pd.Series
        品質スコアQ（float）
    labels : pd.Series
        人手ラベル（1=高品質, 0=低品質）

    Returns
    -------
    float
        一致率（0.0〜1.0）
    """
    median_q = q_scores.median()
    predicted = (q_scores >= median_q).astype(int)
    common = predicted.index.intersection(labels.index)
    if len(common) == 0:
        return 0.0
    agreement = (predicted[common] == labels[common]).mean()
    return float(agreement)
