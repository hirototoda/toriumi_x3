"""
品質モデルの smoke test。
合成データで sample / train / save / load / predict の round-trip を確認する。
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.step5_target.domain_trust import domain_trust_score
from src.step5_target.quality_features import FEATURE_COLUMNS, extract_quality_features
from src.step5_target.quality_model import (
    load_model,
    predict_quality,
    save_model,
    train_quality_model,
)
from src.step5_target.quality_score import (
    _heuristic_quality_score,
    compute_quality_score,
)


def _make_synthetic(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """低品質 n/2 + 高品質 n/2 の合成ノート"""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n // 2):
        rows.append({
            "noteId": f"lo_{i}",
            "summary": rng.choice([
                "short.",
                "just an opinion",
                "fake news see https://bit.ly/abc",
                "I disagree",
                "check https://x.com/someone",
            ]),
            "label": 0,
        })
    for i in range(n // 2):
        rows.append({
            "noteId": f"hi_{i}",
            "summary": (
                "This claim is incorrect. According to a 2023 study published "
                "in https://www.nature.com/articles/abc, the data show the opposite. "
                "See also https://www.reuters.com/world/xyz for context."
            ),
            "label": 1,
        })
    return pd.DataFrame(rows)


def test_domain_trust_score():
    assert domain_trust_score("no urls here") == 0.0
    assert domain_trust_score("see https://www.reuters.com/x") == 1.0
    assert domain_trust_score("see https://nih.gov/page") == 1.0
    assert domain_trust_score("see https://news.bbc.co.uk/foo") == 1.0
    assert domain_trust_score("see https://bit.ly/abc") == 0.2
    # max aggregation: 1 本でも信頼できるなら高い
    assert domain_trust_score(
        "see https://bit.ly/abc and https://www.apnews.com/x"
    ) == 1.0
    assert domain_trust_score("see https://example.com/page") == 0.5


def test_extract_quality_features():
    df = _make_synthetic(4)
    feats = extract_quality_features(df)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats.index.name == "noteId"
    assert len(feats) == 4
    assert (feats["char_count"] >= 0).all()
    assert (feats["domain_trust"].between(0.0, 1.0)).all()


def test_train_and_predict_roundtrip(tmp_path: Path):
    df = _make_synthetic(20)
    feats = extract_quality_features(df[["noteId", "summary"]])
    labels = df.set_index("noteId")["label"]

    bundle = train_quality_model(feats, labels)
    assert set(bundle.keys()) >= {
        "scaler", "clf", "feature_names", "trained_at", "n_train", "cv_auc"
    }
    assert bundle["n_train"] == 20
    assert bundle["feature_names"] == FEATURE_COLUMNS

    out = tmp_path / "model.joblib"
    save_model(bundle, out)
    assert out.exists()

    loaded = load_model(out)
    q = predict_quality(df[["noteId", "summary"]], loaded)
    assert q.index.name == "noteId"
    assert q.between(0.0, 1.0).all()
    assert len(q) == len(df)

    # 合成データは明確に分離されているので hi_* の平均は lo_* より高いはず
    hi_mean = q[q.index.str.startswith("hi_")].mean()
    lo_mean = q[q.index.str.startswith("lo_")].mean()
    assert hi_mean > lo_mean


def test_compute_quality_score_uses_model(tmp_path: Path):
    df = _make_synthetic(20)
    feats = extract_quality_features(df[["noteId", "summary"]])
    labels = df.set_index("noteId")["label"]
    bundle = train_quality_model(feats, labels)
    path = tmp_path / "m.joblib"
    save_model(bundle, path)

    q = compute_quality_score(df[["noteId", "summary"]], model_path=path)
    assert q.between(0.0, 1.0).all()


def test_compute_quality_score_missing_explicit_raises(tmp_path: Path):
    df = _make_synthetic(4)
    with pytest.raises(FileNotFoundError):
        compute_quality_score(df[["noteId", "summary"]], model_path=tmp_path / "nope.joblib")


def test_compute_quality_score_missing_default_falls_back(monkeypatch, tmp_path: Path):
    from src.step5_target import quality_score as qs_mod
    monkeypatch.setattr(qs_mod, "DEFAULT_MODEL_PATH", tmp_path / "missing.joblib")

    df = _make_synthetic(4)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        q = compute_quality_score(df[["noteId", "summary"]])
        assert any("falling back to heuristic" in str(x.message) for x in w)
    assert len(q) == len(df)
    assert q.between(0.0, 1.0).all()


def test_heuristic_still_works():
    df = _make_synthetic(6)
    q = _heuristic_quality_score(df[["noteId", "summary"]])
    assert len(q) == 6
    assert q.between(0.0, 1.0).all()
