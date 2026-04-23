"""
特徴量構築 + ロジスティック回帰

回帰式:
    log p(deleted=1) / (1-p) = β0 + β1·type_a + β2·type_b
                                  + β3·quality + β4·log_ratings_count

trend を削った理由:
    trend は (後半 score 平均 - 前半 score 平均) で、目的変数 deleted と
    同じ生データ (helpfulnessLevel) から作られた量。これを control に
    入れると目的変数を一部 control する形になり、β_typeA を不当に
    押し下げる (bad control)。

説明変数間の相関:
    type_a × type_b           : 排他 (1 ノート 1 バーストなので両方 1 にはならない)
    type_a × log_ratings_count: バースト判定が min_count 件以上を要求するため
                                 構造的な正相関 → log_ratings_count の control 必須
    quality × log_ratings_count: 弱い正相関 (長い note は注目を集めやすい)

→ run() で相関行列と VIF を必ず print する.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

VALID_STATUSES = {"CURRENTLY_RATED_HELPFUL", "CURRENTLY_RATED_NOT_HELPFUL"}
X_COLS = ["type_a", "type_b", "quality", "log_ratings_count"]


def build_features(
    ratings: pd.DataFrame,
    bursts: pd.DataFrame,
    history: pd.DataFrame,
    quality: pd.Series,
) -> pd.DataFrame:
    """各 note 1 行の特徴量 DataFrame を作る."""
    # 評価数 (人気度の control)
    rcount = ratings.groupby("noteId").size().rename("ratings_count")

    # ノート status を辞書化
    status = history.drop_duplicates("noteId").set_index("noteId")["currentStatus"]

    # バーストタイプ (note 単位、A or B or NaN)
    if not bursts.empty:
        burst_flag = bursts.set_index("noteId")["burst_type"]
    else:
        burst_flag = pd.Series(dtype=object)

    note_ids = ratings["noteId"].unique()
    rows = []
    for nid in note_ids:
        s = status.get(nid)
        if s not in VALID_STATUSES:
            continue
        bt = burst_flag.get(nid)
        # bt は None (バースト無し) / "A" / "B" / pd.NA (バーストはあるが分類不能).
        # NA は run_logit の dropna() で落とすために type_a/type_b を NaN にする
        # (旧実装は `1 if bt == "A" else 0` で NA 比較が TypeError を起こしていた)
        if bt is not None and pd.isna(bt):
            type_a = np.nan
            type_b = np.nan
        else:
            type_a = 1 if bt == "A" else 0
            type_b = 1 if bt == "B" else 0
        rows.append({
            "noteId":            nid,
            "deleted":           0 if s == "CURRENTLY_RATED_HELPFUL" else 1,
            "type_a":            type_a,
            "type_b":            type_b,
            "quality":           float(quality.get(nid, np.nan)),
            "ratings_count":     int(rcount.get(nid, 0)),
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        print("[features] 0 notes (no notes with definitive status)")
        return feat

    feat["log_ratings_count"] = np.log1p(feat["ratings_count"])
    print(
        f"[features] {len(feat):,} notes, "
        f"deleted={int(feat['deleted'].sum())}, "
        f"helpful={int((feat['deleted']==0).sum())}, "
        f"typeA={int(feat['type_a'].sum())}, typeB={int(feat['type_b'].sum())}"
    )
    return feat


def _print_corr_and_vif(X: pd.DataFrame) -> None:
    print("\n[diag] correlation matrix:")
    print(X.corr().round(3).to_string())

    print("\n[diag] VIF (Variance Inflation Factor):")
    Xc = sm.add_constant(X)
    for i, col in enumerate(Xc.columns):
        vif = variance_inflation_factor(Xc.values, i)
        flag = "OK" if vif < 5 else "注意" if vif < 10 else "多重共線あり"
        print(f"  {col:<20} VIF={vif:6.2f}  [{flag}]")


def run_logit(feat: pd.DataFrame) -> sm.iolib.summary2.Summary:
    """ロジスティック回帰を 1 本走らせて summary を返す."""
    sub = feat[["deleted"] + X_COLS].dropna()
    if len(sub) < 10:
        raise ValueError(f"too few notes after dropna: {len(sub)}")
    if sub["deleted"].nunique() < 2:
        raise ValueError("deleted has no variance")
    if sub["type_a"].nunique() < 2:
        raise ValueError("type_a has no variance")

    X = sub[X_COLS]
    y = sub["deleted"]

    _print_corr_and_vif(X)

    Xc = sm.add_constant(X)
    res = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()

    print("\n" + "=" * 64)
    print("  Logistic Regression  (deleted ~ type_a + type_b + quality + log_ratings_count)")
    print("=" * 64)
    summary = res.summary2().tables[1]
    for var in summary.index:
        beta = summary.loc[var, "Coef."]
        p    = summary.loc[var, "P>|z|"]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:<22} β={beta:+.4f}  p={p:.4f} {sig}")
    print("=" * 64)

    p_a = summary.loc["type_a", "P>|z|"] if "type_a" in summary.index else 1.0
    p_b = summary.loc["type_b", "P>|z|"] if "type_b" in summary.index else 1.0
    if   p_a < 0.05 and p_b >= 0.05: verdict = "TypeA のみ有意 → 仮説支持 (陣営反応で潰されている)"
    elif p_a < 0.05 and p_b < 0.05:  verdict = "TypeA/B 両方有意 → 自然拡散も寄与 (仮説部分支持)"
    elif p_a >= 0.05 and p_b < 0.05: verdict = "TypeB のみ有意 → 仮説不支持"
    else:                            verdict = "どちらも非有意 → 仮説不支持 / サンプル不足"
    print(f"\n  → {verdict}")
    print("=" * 64)

    return res
