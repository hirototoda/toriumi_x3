"""
Step 5: 品質モデル（LLM ラベルから学習する 2値ロジスティック回帰）

入力: 200 件程度のラベル付きノート（label ∈ {0, 1}）
出力: ノート → 高品質確率 [0, 1] の予測器

bundle 構造:
    {
        "scaler": StandardScaler,
        "clf": LogisticRegression,
        "feature_names": [...],
        "trained_at": iso string,
        "n_train": int,
        "cv_auc": float,
    }
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from src.step5_target.quality_features import FEATURE_COLUMNS, extract_quality_features


def train_quality_model(
    features_df: pd.DataFrame,
    labels: pd.Series,
    cv_folds: int = 5,
) -> dict:
    """
    ラベル付き特徴量から品質モデルを学習する。

    Parameters
    ----------
    features_df : pd.DataFrame
        extract_quality_features の出力（index=noteId）
    labels : pd.Series
        index=noteId, 値={0, 1}
    cv_folds : int
        CV 分割数（ラベル数が少ない場合は自動で縮小）

    Returns
    -------
    dict
        モデル bundle
    """
    common = features_df.index.intersection(labels.index)
    X = features_df.loc[common, FEATURE_COLUMNS].values
    y = labels.loc[common].astype(int).values

    if len(y) < 10:
        raise ValueError(f"ラベル数が少なすぎます: {len(y)} 件 (最低 10 件必要)")
    if len(set(y)) < 2:
        raise ValueError("ラベルに 2 値の両方が含まれていません（全 0 または全 1）")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")

    min_class = min((y == 0).sum(), (y == 1).sum())
    folds = max(2, min(cv_folds, min_class))
    try:
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
        cv_auc = float(cross_val_score(clf, X_scaled, y, cv=cv, scoring="roc_auc").mean())
    except Exception as e:
        print(f"  WARNING: CV AUC を計算できませんでした ({e})")
        cv_auc = float("nan")

    clf.fit(X_scaled, y)

    return {
        "scaler": scaler,
        "clf": clf,
        "feature_names": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_train": int(len(y)),
        "cv_auc": cv_auc,
    }


def save_model(bundle: dict, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)


def load_model(path: Union[str, Path]) -> dict:
    return joblib.load(Path(path))


def predict_quality(notes_df: pd.DataFrame, bundle: dict) -> pd.Series:
    """
    ノートに対して高品質確率 [0, 1] を返す。
    """
    feats = extract_quality_features(notes_df)
    cols = bundle["feature_names"]
    X = bundle["scaler"].transform(feats[cols].values)
    proba = bundle["clf"].predict_proba(X)[:, 1]
    q = pd.Series(proba, index=feats.index, name="quality")
    return q
