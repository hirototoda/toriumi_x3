"""
TypeA/B バースト区間内の helpfulnessLevel 平均を集計する追加分析

目的:
  本番回帰で β_typeA < 0 となった結果 (バーストがあると Not Helpful 化しにくい)
  の方向性を helpfulnessLevel そのもので確認する。
    平均 > 0 → HELPFUL 多数 (= 防衛 rally 仮説と整合)
    平均 < 0 → NOT_HELPFUL 多数 (= 攻撃 rally 仮説と整合)

入力:
  data/processed/simple_bursts.csv (noteId, burst_start, burst_end, burst_type)
  data/raw/ratings-*.tsv

出力:
  data/processed/simple_burst_helpfulness.txt
  (TypeA/B ごとの mean / std / N / Helpful 率 / NotHelpful 率)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.simple.load import _CHUNK, _find, RATINGS_COLS  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RAW       = ROOT / "data" / "raw"

SCORE_MAP = {
    "HELPFUL":          +1,
    "SOMEWHAT_HELPFUL":  0,
    "NOT_HELPFUL":      -1,
}


def load_bursts() -> pd.DataFrame:
    path = PROCESSED / "simple_bursts.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} が無い。先に scripts/run_simple.py を実行してください。")
    b = pd.read_csv(path, dtype={"noteId": str})
    need = {"noteId", "burst_start", "burst_end", "burst_type"}
    missing = need - set(b.columns)
    if missing:
        raise ValueError(f"simple_bursts.csv に必要列が不足: {missing}")
    print(f"[in] bursts: {len(b):,}  (A={int((b.burst_type=='A').sum())}, B={int((b.burst_type=='B').sum())})")
    return b


def load_ratings_in_bursts(burst_note_ids: set[str]) -> pd.DataFrame:
    """ratings をチャンク読みしながら burst 対象 noteId だけ拾う"""
    paths = _find(RAW, "ratings")
    kept = []
    for p in paths:
        print(f"[load]   scanning {p.name} ...")
        for chunk in pd.read_csv(
            p, sep="\t", usecols=RATINGS_COLS,
            dtype={"noteId": str, "raterParticipantId": str},
            chunksize=_CHUNK,
        ):
            kept.append(chunk[chunk["noteId"].isin(burst_note_ids)])
    r = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=RATINGS_COLS)
    print(f"[load] ratings (in burst notes): {len(r):,}")
    return r


def aggregate(bursts: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """各バーストの区間 [burst_start, burst_end] 内の評価の helpfulness 統計を集める"""
    ratings = ratings.copy()
    ratings["score"] = ratings["helpfulnessLevel"].map(SCORE_MAP)

    by_note = {nid: g for nid, g in ratings.groupby("noteId")}

    rows = []
    for _, row in bursts.iterrows():
        nid = row["noteId"]
        if nid not in by_note:
            continue
        g = by_note[nid]
        m = (g["createdAtMillis"] >= row["burst_start"]) & (g["createdAtMillis"] <= row["burst_end"])
        sub = g.loc[m]
        if sub.empty:
            continue
        rows.append({
            "noteId":       nid,
            "burst_type":   row["burst_type"],
            "n_in_burst":   len(sub),
            "mean_score":   sub["score"].mean(),
            "frac_helpful":     float((sub["helpfulnessLevel"] == "HELPFUL").mean()),
            "frac_somewhat":    float((sub["helpfulnessLevel"] == "SOMEWHAT_HELPFUL").mean()),
            "frac_not_helpful": float((sub["helpfulnessLevel"] == "NOT_HELPFUL").mean()),
        })
    return pd.DataFrame(rows)


def summarize(per_burst: pd.DataFrame) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Burst-internal helpfulness summary")
    lines.append("=" * 70)
    lines.append(f"bursts analyzed: {len(per_burst):,}")
    lines.append("")

    for tp, g in per_burst.groupby("burst_type"):
        lines.append(f"[Type{tp}]  N = {len(g):,} bursts")
        lines.append(f"  mean_score        : {g['mean_score'].mean():+.4f}  (std {g['mean_score'].std():.4f})")
        lines.append(f"  frac_helpful      : {g['frac_helpful'].mean():.4f}")
        lines.append(f"  frac_somewhat     : {g['frac_somewhat'].mean():.4f}")
        lines.append(f"  frac_not_helpful  : {g['frac_not_helpful'].mean():.4f}")
        lines.append("")

    if {"A", "B"}.issubset(set(per_burst["burst_type"].unique())):
        a = per_burst.loc[per_burst.burst_type == "A", "mean_score"]
        b = per_burst.loc[per_burst.burst_type == "B", "mean_score"]
        from scipy import stats
        t, p = stats.ttest_ind(a, b, equal_var=False)
        lines.append(f"Welch t-test  TypeA vs TypeB mean_score: t={t:+.3f}, p={p:.3g}")
        lines.append(f"  diff (A - B) = {a.mean() - b.mean():+.4f}")
        lines.append("")

    lines.append("Interpretation guide:")
    lines.append("  mean_score >  0  ->  HELPFUL 多数 (= 防衛 rally 仮説と整合)")
    lines.append("  mean_score <  0  ->  NOT_HELPFUL 多数 (= 攻撃 rally 仮説と整合)")
    lines.append("  TypeA が TypeB より明確に + 寄り  -> 同陣営集中 = 防衛が主流")
    return "\n".join(lines)


def main() -> None:
    bursts = load_bursts()
    note_ids = set(bursts["noteId"].astype(str))
    ratings = load_ratings_in_bursts(note_ids)
    if ratings.empty:
        print("ERROR: ratings が空。data/raw/ratings-*.tsv の有無を確認してください。", file=sys.stderr)
        sys.exit(1)

    per_burst = aggregate(bursts, ratings)
    out_csv = PROCESSED / "simple_burst_helpfulness.csv"
    per_burst.to_csv(out_csv, index=False)
    print(f"[out] {out_csv} ({len(per_burst):,} rows)")

    report = summarize(per_burst)
    print("\n" + report)
    (PROCESSED / "simple_burst_helpfulness.txt").write_text(report)
    print(f"\n[out] {PROCESSED / 'simple_burst_helpfulness.txt'}")


if __name__ == "__main__":
    main()
