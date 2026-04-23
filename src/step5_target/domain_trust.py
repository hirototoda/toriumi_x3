"""
ドメイン信頼度スコア

ノート本文内の URL を抽出し、ドメインの信頼性に基づいて [0, 1] のスコアを返す。
外部 API は使わず、静的ホワイトリスト / ブラックリスト / TLD ルールのみ。
"""

import re
from urllib.parse import urlparse

TRUSTED_TLDS = {"gov", "edu", "int", "mil"}

TRUSTED_DOMAINS = {
    # 国際通信社・主要報道
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk",
    "nytimes.com", "washingtonpost.com", "wsj.com", "ft.com",
    "economist.com", "theguardian.com", "bloomberg.com",
    # 学術・科学
    "nature.com", "science.org", "sciencemag.org",
    "ncbi.nlm.nih.gov", "pubmed.ncbi.nlm.nih.gov",
    # 国際機関・行政系（.int / .gov は TLD で拾うが補強）
    "who.int", "un.org", "imf.org", "worldbank.org",
    # 日本の主要メディア
    "nhk.or.jp", "asahi.com", "mainichi.jp", "yomiuri.co.jp",
    "japantimes.co.jp", "nikkei.com",
    # ファクトチェック
    "snopes.com", "factcheck.org", "politifact.com",
    # 百科
    "wikipedia.org", "en.wikipedia.org", "ja.wikipedia.org",
}

LOW_TRUST_DOMAINS = {
    # 短縮URL（転送先が不明）
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly",
    # 自費出版・個人メディア
    "medium.com", "substack.com",
    # 動画・SNS（出典になりにくい）
    "youtube.com", "youtu.be", "x.com", "twitter.com",
    "tiktok.com", "instagram.com", "facebook.com",
}

URL_RE = re.compile(r"https?://[^\s)<>\"']+", re.IGNORECASE)


def _score_domain(host: str) -> float:
    host = host.lower().lstrip(".")
    if host.startswith("www."):
        host = host[4:]

    tld = host.rsplit(".", 1)[-1]
    if tld in TRUSTED_TLDS:
        return 1.0

    if host in TRUSTED_DOMAINS:
        return 1.0
    # サブドメインも許容（例: news.bbc.co.uk）
    for d in TRUSTED_DOMAINS:
        if host.endswith("." + d):
            return 1.0

    if host in LOW_TRUST_DOMAINS:
        return 0.2
    for d in LOW_TRUST_DOMAINS:
        if host.endswith("." + d):
            return 0.2

    return 0.5


def domain_trust_score(text: str) -> float:
    """
    テキスト中の全URLから最大信頼度スコアを返す。
    URL 無し → 0.0、1 本でも信頼できる出典があれば報いる設計。
    """
    if not isinstance(text, str) or not text:
        return 0.0
    urls = URL_RE.findall(text)
    if not urls:
        return 0.0
    scores = []
    for u in urls:
        try:
            host = urlparse(u).hostname or ""
        except Exception:
            continue
        if host:
            scores.append(_score_domain(host))
    return max(scores) if scores else 0.0
