"""
Step 2: トピック分類

政治トピックのノートのみを抽出する。
キーワードベースのフィルタリング（APIコスト$0）。
"""

import pandas as pd

# 政治系キーワード（英語）— 最低限のリスト
POLITICAL_KEYWORDS = [
    "trump", "biden", "democrat", "republican", "gop",
    "congress", "senate", "election", "vote", "ballot",
    "liberal", "conservative", "left-wing", "right-wing",
    "immigration", "border", "refugee", "asylum",
    "abortion", "gun control", "second amendment", "2nd amendment",
    "climate change", "global warming",
    "vaccine", "covid", "pandemic", "mask mandate",
    "blm", "police", "protest", "riot",
    "supreme court", "scotus",
    "nato", "ukraine", "russia", "china", "israel", "palestine", "gaza",
    "woke", "dei", "crt", "critical race",
    "transgender", "lgbtq", "gender",
    "censorship", "free speech", "misinformation",
    "government", "policy", "legislation", "partisan",
]


def classify_political_topics(df: pd.DataFrame) -> pd.DataFrame:
    """
    政治トピックのノートのみを抽出する。

    Parameters
    ----------
    df : pd.DataFrame
        summary カラムを含むDataFrame

    Returns
    -------
    pd.DataFrame
        is_political カラムが追加されたDataFrame
    """
    if "summary" not in df.columns:
        print("    WARNING: summary column not found, skipping topic filter")
        df = df.copy()
        df["is_political"] = True
        return df

    text = df["summary"].fillna("").str.lower()
    pattern = "|".join(POLITICAL_KEYWORDS)
    df = df.copy()
    df["is_political"] = text.str.contains(pattern, regex=True)

    n_political = df["is_political"].sum()
    n_total = len(df)
    print(f"    political topics: {n_political:,} / {n_total:,} notes ({n_political/max(n_total,1)*100:.1f}%)")
    return df
