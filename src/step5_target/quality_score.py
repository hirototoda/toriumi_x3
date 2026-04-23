"""
Step 5: 品質スコア Q

優先: LLM ラベル学習済みモデル (models/quality_model.joblib)
代替: URL 数 + 文字数のヒューリスティック（モデル不在時のフォールバック）
"""

import re
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = ROOT / "models" / "quality_model.joblib"


def compute_quality_score(
    notes_df: pd.DataFrame,
    model_path: Optional[Union[str, Path]] = None,
) -> pd.Series:
    """
    各ノートの品質スコアQを計算する。

    model_path=None の場合、デフォルトパス (models/quality_model.joblib) を試す。
    - 存在: ML モデルで予測（高品質確率 [0, 1]）
    - 不在: 警告を出し、旧ヒューリスティックにフォールバック
    model_path を明示指定して不在の場合は FileNotFoundError。
    """
    explicit = model_path is not None
    path = Path(model_path) if explicit else DEFAULT_MODEL_PATH

    if path.exists():
        # lazy import: joblib/sklearn が無くてもヒューリスティックは動く
        from src.step5_target.quality_model import load_model, predict_quality
        bundle = load_model(path)
        q = predict_quality(notes_df, bundle)
        print(
            f"    quality (ML): mean={q.mean():.3f}, std={q.std():.3f} "
            f"[model: n_train={bundle.get('n_train')}, cv_auc={bundle.get('cv_auc'):.3f}]"
        )
        return q

    if explicit:
        raise FileNotFoundError(f"quality model not found: {path}")

    warnings.warn(
        f"quality model not found at {path}; falling back to heuristic (URL数+文字数).",
        stacklevel=2,
    )
    return _heuristic_quality_score(notes_df)


def _heuristic_quality_score(notes_df: pd.DataFrame) -> pd.Series:
    """
    旧ヒューリスティック: URL 数と文字数の min-max 正規化平均。
    """
    df = notes_df.copy()
    text = df["summary"].fillna("")

    url_count = text.apply(lambda t: len(re.findall(r"https?://", t)))
    char_count = text.str.len()

    def _norm(s):
        mn, mx = s.min(), s.max()
        if mx == mn:
            return pd.Series(0.5, index=s.index)
        return (s - mn) / (mx - mn)

    q = (_norm(url_count) + _norm(char_count)) / 2.0
    q.index = df["noteId"]
    q.name = "quality"
    print(f"    quality (heuristic): mean={q.mean():.3f}, std={q.std():.3f}")
    return q
