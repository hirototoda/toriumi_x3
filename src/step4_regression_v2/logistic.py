"""
Step 4 v2: ロジスティック回帰 (control 拡張版)

旧式:
    log p/(1-p) = β0 + β1·TypeA + β2·TypeB + β3·Trend + β4·Quality

v2 はこれに加えて
    + β5·log1p(ratings_count)   人気度の control
    + β6·bridging_score          評価者 polarity 多様性 (CN MF noteScore の自前近似)

3 モデルを比較表示する:
  M0: 旧式 (再掲、ベースライン)
  M1: + log1p(ratings_count)
  M2: + log1p(ratings_count) + bridging_score   ← 一番効くモデル

bridging_score の欠損ノートは M2 から除外される (列 dropna)。

解釈:
  M2 で β_TypeA が消えれば → CN アルゴリズム的「polarity 多様性不足」で説明可能
  M2 で β_TypeA が残れば   → 多様性を揃えてもなお陣営攻撃の効果がある (アルゴ穴)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import statsmodels.api as sm


BASE_COLS = ["type_a", "type_b", "trend", "quality"]


@dataclass
class ModelFit:
    name: str
    n: int
    coef: dict[str, float] = field(default_factory=dict)
    pval: dict[str, float] = field(default_factory=dict)
    note: str = ""
    result: object = None


def _safe_fit(df: pd.DataFrame, x_cols: list[str], name: str) -> ModelFit:
    fit = ModelFit(name=name, n=len(df))
    if len(df) < 10:
        fit.note = f"skip: too few notes ({len(df)})"
        return fit
    if df["deleted"].nunique() < 2:
        fit.note = "skip: no variance in deleted"
        return fit
    for c in x_cols:
        if c not in df.columns:
            fit.note = f"skip: missing column {c}"
            return fit

    sub = df[["deleted"] + x_cols].dropna()
    fit.n = len(sub)
    if fit.n < 10 or sub["deleted"].nunique() < 2:
        fit.note = f"skip: too few notes after dropna ({fit.n})"
        return fit
    if sub["type_a"].nunique() < 2:
        fit.note = "skip: no variance in type_a"
        return fit

    y = sub["deleted"]
    X = sm.add_constant(sub[x_cols])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        except Exception as e:
            fit.note = f"error: {e}"
            return fit

    summary = res.summary2().tables[1]
    for idx in summary.index:
        fit.coef[idx] = float(summary.loc[idx, "Coef."])
        fit.pval[idx] = float(summary.loc[idx, "P>|z|"])
    if abs(fit.coef.get("type_a", 0.0)) > 100:
        fit.note = f"warn: large β (perfect separation suspected) β_typeA={fit.coef['type_a']:.1f}"
    fit.result = res
    return fit


def _prep(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ratings_count" in out.columns:
        out["log_ratings_count"] = np.log1p(out["ratings_count"].astype(float))
    return out


def fit_logistic_regression_v2(
    feat_df: pd.DataFrame,
    *,
    print_results: bool = True,
) -> dict[str, ModelFit]:
    """3 モデル (M0/M1/M2) を順に推定する。"""
    df = _prep(feat_df)

    fits: dict[str, ModelFit] = {}
    fits["M0_base"]               = _safe_fit(df, BASE_COLS, "M0_base")
    fits["M1_with_count"]         = _safe_fit(df, BASE_COLS + ["log_ratings_count"], "M1_with_count")
    if "bridging_score" in df.columns and df["bridging_score"].notna().any():
        fits["M2_with_count_bridge"] = _safe_fit(
            df, BASE_COLS + ["log_ratings_count", "bridging_score"], "M2_with_count_bridge",
        )
    else:
        fits["M2_with_count_bridge"] = ModelFit(
            name="M2_with_count_bridge", n=0,
            note="skip: bridging_score 列が無い or 全 NaN (polarity_df 未投入)",
        )

    if print_results:
        _print_fits(fits)
    return fits


def _print_fits(fits: dict[str, ModelFit]) -> None:
    print("\n" + "=" * 72)
    print("  Logistic Regression v2 — Model comparison")
    print("=" * 72)
    for name, fit in fits.items():
        print(f"\n[{name}]  n={fit.n}")
        if fit.note:
            print(f"  {fit.note}")
        if not fit.coef:
            continue
        for var, beta in fit.coef.items():
            p = fit.pval[var]
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {var:<22} β={beta:+.4f}  p={p:.4f} {sig}")
    print("=" * 72)
    b0 = fits.get("M0_base", ModelFit("", 0)).coef.get("type_a")
    b2 = fits.get("M2_with_count_bridge", ModelFit("", 0)).coef.get("type_a")
    if b0 is not None and b2 is not None:
        print(f"\n  β_typeA: M0={b0:+.4f}  →  M2={b2:+.4f}  "
              f"(変化={b2-b0:+.4f})")
        print("  M2 で残れば「polarity 多様性を揃えてもなお陣営攻撃の効果あり (アルゴ穴)」")
        print("  M2 で消えれば「polarity 多様性不足で説明できる (CN アルゴ仕様の帰結)」")
    print("=" * 72)


def fits_to_rows(fits: dict[str, ModelFit], topic: Optional[str] = None) -> list[dict]:
    """topic 比較表に流し込みやすい形に整形する。"""
    rows = []
    for name, fit in fits.items():
        row = {
            "model": name,
            "n": fit.n,
            "note": fit.note,
        }
        if topic is not None:
            row = {"topic": topic, **row}
        for var in ["type_a", "type_b", "trend", "quality",
                    "log_ratings_count", "bridging_score", "const"]:
            row[f"beta_{var}"] = fit.coef.get(var)
            row[f"p_{var}"] = fit.pval.get(var)
        p = fit.pval.get("type_a")
        if p is None:
            row["sig_typeA"] = ""
        elif p < 0.001:
            row["sig_typeA"] = "***"
        elif p < 0.01:
            row["sig_typeA"] = "**"
        elif p < 0.05:
            row["sig_typeA"] = "*"
        else:
            row["sig_typeA"] = ""
        rows.append(row)
    return rows
