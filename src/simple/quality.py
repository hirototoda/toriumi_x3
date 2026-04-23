"""
Quality スコア (LLM ラベル学習済みの重みをハードコード)

200 件のノートを Claude が {0,1} ラベル付け → ロジスティック回帰を学習
(scripts/train_quality_model.py 参照, CV AUC = 0.894)

ここでは学習済 joblib をブラックボックスで呼ばず、係数を定数として
書き写しておく。再学習しない限り重みは変わらないし、何が起きているか
コードを読むだけで分かるのが利点。

数式 (標準化後の線形和 → sigmoid):
    z = INTERCEPT + Σ COEF[k] * (feature[k] - MEAN[k]) / SCALE[k]
    quality = 1 / (1 + exp(-z))   ∈ [0, 1]

寄与度 (係数の絶対値の比):
    domain_trust : 44%   ← 最大
    char_count   : 33%
    url_count    : 23%

→ 「信頼できるドメインの URL があるか」が一番効くという
   LLM ラベルから読み取れる事前知識と整合する。

再学習方法:
    python scripts/train_quality_model.py train
    上で出た `coef:` `scaler.mean_:` `scaler.scale_:` をこの定数に書き写す。
"""

import numpy as np
import pandas as pd

from src.step5_target.domain_trust import URL_RE, domain_trust_score

# === LLM ラベル学習で得た係数 (n_train=200, CV AUC=0.894) =========
INTERCEPT = 0.5191
COEF  = {"url_count": 0.6538, "char_count": 0.9624, "domain_trust": 1.2845}
MEAN  = {"url_count": 1.325,  "char_count": 274.46, "domain_trust": 0.4355}
SCALE = {"url_count": 1.2039, "char_count": 183.40, "domain_trust": 0.3228}
# =================================================================


def _features(text: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({
        "url_count":    text.apply(lambda t: len(URL_RE.findall(t))).astype(float),
        "char_count":   text.str.len().astype(float),
        "domain_trust": text.apply(domain_trust_score).astype(float),
    })


def quality_score(notes: pd.DataFrame) -> pd.Series:
    """notes (noteId, summary) → quality スコア (Series, index=noteId, value ∈ [0,1])"""
    text  = notes["summary"].fillna("").astype(str)
    feats = _features(text)

    z = INTERCEPT + sum(
        COEF[k] * (feats[k] - MEAN[k]) / SCALE[k] for k in COEF
    )
    q = 1.0 / (1.0 + np.exp(-z))

    out = pd.Series(q.values, index=notes["noteId"].values, name="quality")
    print(f"[quality] mean={out.mean():.3f}, std={out.std():.3f} ({len(out):,} notes)")
    return out
