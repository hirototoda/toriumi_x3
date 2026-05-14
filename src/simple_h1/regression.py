"""
H1 用の特徴量構築 + ロジスティック回帰

simple.regression との差分:
  * 変数: type_a, type_b → type_a_helpful, type_a_nothelp, type_b_helpful, type_b_nothelp
  * quality, log_ratings_count は同じ
  * VIF / 相関行列の print 処理は共通 → simple.regression._print_corr_and_vif を import

回帰式:
    log p(deleted=1)/(1-p)
      = β0 + β1·type_a_helpful + β2·type_a_nothelp
           + β3·type_b_helpful + β4·type_b_nothelp
           + β5·quality + β6·log_ratings_count

H1 の判定指針 (本モジュール末尾で自動 print):
  * β(type_a_nothelp) > 0 で有意 → 「NOT_HELPFUL 一斉で潰されている」を支持
  * β(type_a_helpful) < 0 で有意 → 「HELPFUL 一斉で擁護されている」が負係数の正体
  * 両者の符号・有意性をまとめて verdict として出す
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.simple.regression import VALID_STATUSES, _print_corr_and_vif

X_COLS_H1 = [
    "type_a_helpful", "type_a_nothelp",
    "type_b_helpful", "type_b_nothelp",
    "quality", "log_ratings_count",
]


def build_features_h1(
    ratings: pd.DataFrame,
    bursts:  pd.DataFrame,  # simple_h1.burst.detect_bursts_with_direction + classify_burst_type 済み
    history: pd.DataFrame,
    quality: pd.Series,
) -> pd.DataFrame:
    """TypeA/B × {helpful, nothelp} の 4 ダミーを各 note 1 行で返す."""
    rcount = ratings.groupby("noteId").size().rename("ratings_count")
    status = history.drop_duplicates("noteId").set_index("noteId")["currentStatus"]

    if not bursts.empty:
        bt_map  = bursts.set_index("noteId")["burst_type"]        # "A" / "B"
        dir_map = bursts.set_index("noteId")["burst_direction"]   # "helpful" / "nothelp"
    else:
        bt_map  = pd.Series(dtype=object)
        dir_map = pd.Series(dtype=object)

    rows = []
    for nid in ratings["noteId"].unique():
        s = status.get(nid)
        if s not in VALID_STATUSES:
            continue
        bt = bt_map.get(nid)    # None / "A" / "B"
        dr = dir_map.get(nid)   # None / "helpful" / "nothelp"
        rows.append({
            "noteId":            nid,
            "deleted":           0 if s == "CURRENTLY_RATED_HELPFUL" else 1,
            "type_a_helpful":    1 if bt == "A" and dr == "helpful" else 0,
            "type_a_nothelp":    1 if bt == "A" and dr == "nothelp" else 0,
            "type_b_helpful":    1 if bt == "B" and dr == "helpful" else 0,
            "type_b_nothelp":    1 if bt == "B" and dr == "nothelp" else 0,
            "quality":           float(quality.get(nid, np.nan)),
            "ratings_count":     int(rcount.get(nid, 0)),
        })

    feat = pd.DataFrame(rows)
    if feat.empty:
        print("[features-h1] 0 notes (no notes with definitive status)")
        return feat

    feat["log_ratings_count"] = np.log1p(feat["ratings_count"])
    print(
        f"[features-h1] {len(feat):,} notes, "
        f"deleted={int(feat['deleted'].sum())}, "
        f"helpful={int((feat['deleted']==0).sum())}, "
        f"A+helpful={int(feat['type_a_helpful'].sum())}, "
        f"A+nothelp={int(feat['type_a_nothelp'].sum())}, "
        f"B+helpful={int(feat['type_b_helpful'].sum())}, "
        f"B+nothelp={int(feat['type_b_nothelp'].sum())}"
    )
    return feat


def _verdict_h1(summary: pd.DataFrame) -> str:
    """H1 の仮説支持 / 不支持を 4 つの係数の符号・有意性で判定."""
    def ok(var: str) -> tuple[float, float, bool]:
        if var not in summary.index:
            return (0.0, 1.0, False)
        b = summary.loc[var, "Coef."]
        p = summary.loc[var, "P>|z|"]
        return (b, p, p < 0.05)

    b_ah, p_ah, s_ah = ok("type_a_helpful")
    b_an, p_an, s_an = ok("type_a_nothelp")
    b_bh, p_bh, s_bh = ok("type_b_helpful")
    b_bn, p_bn, s_bn = ok("type_b_nothelp")

    lines = []
    if s_an and b_an > 0:
        lines.append("TypeA+nothelp が β>0 で有意 → 『陣営反応で潰されている』仮説を支持")
    elif s_an and b_an < 0:
        lines.append("TypeA+nothelp が β<0 で有意 → 元仮説と逆 (NOT_HELPFUL 一斉でも採用側に寄る)")
    else:
        lines.append("TypeA+nothelp は非有意 → 『潰し』方向の効果は検出されず")

    if s_ah and b_ah < 0:
        lines.append("TypeA+helpful が β<0 で有意 → 元 TypeA 負係数の正体は『擁護一斉』")
    elif s_ah and b_ah > 0:
        lines.append("TypeA+helpful が β>0 で有意 → 擁護一斉も潰し方向")
    else:
        lines.append("TypeA+helpful は非有意")

    return "\n    ".join(lines)


def run_logit_h1(feat: pd.DataFrame) -> sm.iolib.summary2.Summary:
    """H1 用 4 ダミー版ロジット."""
    sub = feat[["deleted"] + X_COLS_H1].dropna()
    if len(sub) < 10:
        raise ValueError(f"too few notes after dropna: {len(sub)}")
    if sub["deleted"].nunique() < 2:
        raise ValueError("deleted has no variance")

    # 4 ダミーのうち全部 0 の変数があったら VIF/相関で落ちるので事前警告
    for col in ("type_a_helpful", "type_a_nothelp", "type_b_helpful", "type_b_nothelp"):
        if sub[col].nunique() < 2:
            print(f"[warn] {col} has no variance (all {sub[col].iloc[0]}) — 係数は推定不能")

    X = sub[X_COLS_H1]
    y = sub["deleted"]

    _print_corr_and_vif(X)

    Xc = sm.add_constant(X)
    res = sm.GLM(y, Xc, family=sm.families.Binomial()).fit()

    print("\n" + "=" * 72)
    print("  Logistic Regression (H1)")
    print("  deleted ~ type_a_helpful + type_a_nothelp + type_b_helpful + type_b_nothelp")
    print("          + quality + log_ratings_count")
    print("=" * 72)
    summary = res.summary2().tables[1]
    for var in summary.index:
        beta = summary.loc[var, "Coef."]
        p    = summary.loc[var, "P>|z|"]
        sig  = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {var:<22} β={beta:+.4f}  p={p:.4f} {sig}")
    print("=" * 72)

    print(f"\n  → H1 判定:\n    {_verdict_h1(summary)}")
    print("=" * 72)

    return res
