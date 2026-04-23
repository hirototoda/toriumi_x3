"""
Step 5: 品質モデル用の特徴量抽出

学習と推論で同じ関数を使うことで特徴量ドリフトを防ぐ。
"""

import re

import pandas as pd

from src.step5_target.domain_trust import URL_RE, domain_trust_score

FEATURE_COLUMNS = ["url_count", "char_count", "domain_trust"]


def extract_quality_features(notes_df: pd.DataFrame) -> pd.DataFrame:
    """
    ノート DataFrame から品質モデル用の特徴量を抽出する。

    Parameters
    ----------
    notes_df : pd.DataFrame
        noteId, summary を含む DataFrame

    Returns
    -------
    pd.DataFrame
        index=noteId, columns=[url_count, char_count, domain_trust]
    """
    df = notes_df.copy()
    text = df["summary"].fillna("").astype(str)

    url_count = text.apply(lambda t: len(URL_RE.findall(t))).astype(int)
    char_count = text.str.len().astype(int)
    domain_trust = text.apply(domain_trust_score).astype(float)

    feats = pd.DataFrame({
        "url_count": url_count.values,
        "char_count": char_count.values,
        "domain_trust": domain_trust.values,
    }, index=df["noteId"].values)
    feats.index.name = "noteId"
    return feats
