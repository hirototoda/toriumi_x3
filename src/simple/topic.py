"""
政治トピック判定 (substring → 単語境界版)

旧 src/step2_topic/classify.py は \\b なしで substring マッチしていたため、
"vote" → "voted" "voter" "devote" 等が誤マッチしていた。
ここでは \\b で囲った正規表現で誤マッチを抑える。

入力: notes_unique (noteId, summary)
出力: notes_unique のうち summary が政治キーワードを含む行
"""

import re

import pandas as pd

POLITICAL_KEYWORDS = [
    # 米政治
    "trump", "biden", "democrat", "republican", "gop",
    "congress", "senate", "election", "vote", "ballot",
    "liberal", "conservative",
    # 政策
    "immigration", "border", "refugee", "asylum",
    "abortion", "gun control", "second amendment", "2nd amendment",
    "climate change", "global warming",
    "vaccine", "covid", "pandemic", "mask mandate",
    # 社会運動
    "blm", "police", "protest", "riot",
    "supreme court", "scotus",
    # 国際
    "nato", "ukraine", "russia", "china", "israel", "palestine", "gaza",
    # カルチャーウォー
    "woke", "dei", "crt", "critical race",
    "transgender", "lgbtq", "gender",
    "censorship", "free speech", "misinformation",
    "government", "policy", "legislation", "partisan",
]


def _build_pattern(keywords: list[str]) -> re.Pattern:
    # 単語境界 \b で囲み、複数単語キーワード ("gun control" 等) はそのまま
    escaped = [re.escape(k) for k in keywords]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)


def filter_political_notes(notes: pd.DataFrame) -> pd.DataFrame:
    if "summary" not in notes.columns:
        raise KeyError("notes に summary 列がありません")
    pattern = _build_pattern(POLITICAL_KEYWORDS)
    text = notes["summary"].fillna("").astype(str)
    is_political = text.str.contains(pattern)
    out = notes[is_political].copy()
    print(f"[topic] political: {len(out):,} / {len(notes):,} notes ({len(out)/max(len(notes),1)*100:.1f}%)")
    return out
