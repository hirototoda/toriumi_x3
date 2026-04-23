"""
案A: トピック別にパイプラインを実行し、比較表を出力する。

使い方:
  python scripts/run_by_topic.py                    # 全データ
  python scripts/run_by_topic.py --nrows 500000     # ratings 先頭50万行
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"

from src.io.cache import (
    load_ratings_cached,
    load_notes_cached,
    load_status_history_cached,
    compute_polarity_cached,
    compute_quality_cached,
    ratings_cache_tag,
    notes_cache_tag,
)
from src.step1_preprocess.filter import filter_by_rating_count
from src.step3_burst.detect import detect_bursts
from src.step3_burst.classify_burst import classify_burst_type
from src.step4_regression.features import compute_features_for_regression

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# ── トピック定義 ─────────────────────────────────────
TOPICS = {
    "vaccine_covid": [
        "vaccine", "covid", "pandemic", "mask", "booster",
        "pfizer", "moderna", "antivax", "lockdown", "fauci",
    ],
    "israel_palestine": [
        "israel", "palestine", "gaza", "hamas", "idf",
        "netanyahu", "ceasefire", "zionist", "hezbollah",
    ],
    "trump": [
        "trump", "maga", "indictment", "mar-a-lago",
        "january 6", "j6", "impeach",
    ],
    "immigration": [
        "immigration", "border", "migrant", "asylum",
        "deportation", "illegal alien", "refugee", "caravan",
    ],
    "gun_control": [
        "gun control", "second amendment", "2nd amendment",
        "shooting", "nra", "firearm", "gun violence",
    ],
    "ALL_POLITICAL": [
        "trump", "biden", "democrat", "republican", "gop",
        "congress", "senate", "election", "vote", "ballot",
        "liberal", "conservative", "immigration", "border",
        "abortion", "gun control", "vaccine", "covid",
        "climate change", "supreme court",
        "israel", "palestine", "gaza", "ukraine", "russia",
        "woke", "dei", "transgender", "lgbtq", "censorship",
        "government", "policy", "partisan",
    ],
}


def filter_by_topic(ratings_df, notes_df, keywords):
    """notes の summary にキーワードを含むノートだけに ratings を絞る"""
    notes_unique = notes_df.drop_duplicates("noteId")
    text = notes_unique["summary"].fillna("").str.lower()
    pattern = "|".join(keywords)
    mask = text.str.contains(pattern, regex=True)
    matched_notes = notes_unique.loc[mask, "noteId"]
    return ratings_df[ratings_df["noteId"].isin(matched_notes)]


def run_regression(feat_df):
    """回帰を実行し、TypeA の係数とp値を返す"""
    if feat_df.empty:
        print(f"      skip: empty feature table")
        return None, None, 0, 0, 0

    n_a = int(feat_df["type_a"].sum())
    n_b = int(feat_df["type_b"].sum())
    n = len(feat_df)

    # 回帰が成立するための最低条件
    if n < 10:
        print(f"      skip: too few notes ({n})")
        return None, None, n, n_a, n_b
    if feat_df["deleted"].nunique() < 2:
        print(f"      skip: no variance in deleted (helpful={( feat_df['deleted']==0).sum()}, not={feat_df['deleted'].sum()})")
        return None, None, n, n_a, n_b
    if feat_df["type_a"].nunique() < 2:
        print(f"      skip: no variance in type_a")
        return None, None, n, n_a, n_b

    y = feat_df["deleted"]
    X = feat_df[["type_a", "type_b", "trend", "quality"]]
    X = sm.add_constant(X)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            result = sm.GLM(y, X, family=sm.families.Binomial()).fit()
            summary = result.summary2().tables[1]
            beta1 = summary.loc["type_a", "Coef."]
            p1 = summary.loc["type_a", "P>|z|"]
            # 係数が異常に大きい場合は完全分離 → 信頼できない
            if abs(beta1) > 100:
                print(f"      skip: perfect separation (β={beta1:.1f})")
                return None, None, n, n_a, n_b
            return beta1, p1, n, n_a, n_b
        except Exception as e:
            print(f"      regression error: {e}")
            return None, None, n, n_a, n_b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-rating-files", type=int, default=None,
                        help="ratings ファイルの読み込み数 (default: 全ファイル)")
    parser.add_argument("--file-offset", type=int, default=0,
                        help="ratings ファイルの読み込み開始位置。先頭から N 個飛ばす (default: 0)")
    parser.add_argument("--skip-rows", type=int, default=0,
                        help="ratings の先頭から何行をスキップするか (default: 0)")
    parser.add_argument("--chunk-suffix", type=str, default="",
                        help="出力ファイル名の末尾に付ける識別子 (例: '_f0_r0')")
    parser.add_argument("--min-ratings", type=int, default=10,
                        help="ノートの最低評価件数 (default: 10)")
    parser.add_argument("--topics-json", type=str, default=None,
                        help="トピック定義JSONファイルのパス（省略時はデフォルトを使用）")
    parser.add_argument("--polarity-first-n", type=int, default=50,
                        help="polarity 計算に使う評価者ごとの最初の評価数 (default: 50)")
    parser.add_argument("--burst-speed-multiplier", type=float, default=3.0,
                        help="平均速度の何倍でバーストとみなすか (default: 3.0)")
    parser.add_argument("--burst-min-count", type=int, default=5,
                        help="バースト判定に必要な最小評価数 (default: 5)")
    parser.add_argument("--burst-threshold", type=float, default=None,
                        help="TypeA/TypeB 分類の極性分散閾値 (default: 中央値で自動)")
    parser.add_argument("--trend-min-evals", type=int, default=4,
                        help="トレンド計算に必要な最小評価数 (default: 4)")
    parser.add_argument("--quality-model-path", type=str, default=None,
                        help="品質モデル (joblib) のパス。未指定時は models/quality_model.joblib を試し、"
                             "無ければヒューリスティックにフォールバック")
    args = parser.parse_args()

    # トピック定義の上書き
    global TOPICS
    if args.topics_json:
        import json
        with open(args.topics_json) as f:
            TOPICS = json.load(f)
        print(f"  Topics loaded from {args.topics_json}: {list(TOPICS.keys())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ── データ読み込み ────────────────────────────────
    t0 = time.time()
    print("[Loading data]")
    ratings_df = load_ratings_cached(
        RAW_DIR,
        nrows=args.nrows,
        max_files=args.max_rating_files,
        file_offset=args.file_offset,
        skip_rows=args.skip_rows,
    )
    r_tag = ratings_cache_tag(
        RAW_DIR,
        nrows=args.nrows, max_files=args.max_rating_files,
        file_offset=args.file_offset, skip_rows=args.skip_rows,
    )
    notes_df = load_notes_cached(RAW_DIR)
    n_tag = notes_cache_tag(RAW_DIR)
    history_df = load_status_history_cached(RAW_DIR)

    print(f"  ⏱ Loading: {_fmt(time.time() - t0)}")

    # ── Step 1: Polarity ──────────────────────────────
    t1 = time.time()
    print("\n[Step 1] Computing polarity...")
    polarity_df = compute_polarity_cached(
        ratings_df, first_n=args.polarity_first_n, ratings_tag=r_tag,
    )

    # ── フィルタ（20件以上） ──────────────────────────
    print(f"\n[Step 1] Filtering (>= {args.min_ratings} ratings)...")
    ratings_filtered = filter_by_rating_count(ratings_df, min_count=args.min_ratings)

    print(f"  ⏱ Step 1: {_fmt(time.time() - t1)}")

    # ── 品質スコア ────────────────────────────────────
    print("\n[Step 5] Computing quality scores...")
    notes_unique = notes_df.drop_duplicates("noteId")
    quality = compute_quality_cached(
        notes_unique, model_path=args.quality_model_path, notes_tag=n_tag,
    )

    # ── トピック別実行 ────────────────────────────────
    results = []

    for topic_name, keywords in TOPICS.items():
        t_topic = time.time()
        print(f"\n{'='*60}")
        print(f"  Topic: {topic_name}")
        print(f"{'='*60}")

        # トピックフィルタ
        ratings_topic = filter_by_topic(ratings_filtered, notes_df, keywords)
        n_notes = ratings_topic["noteId"].nunique()
        print(f"  notes: {n_notes}, ratings: {len(ratings_topic):,}")

        if n_notes < 10:
            print("  → too few notes, skipping")
            results.append({
                "topic": topic_name, "n_notes": n_notes,
                "n_typeA": 0, "n_typeB": 0,
                "beta_typeA": None, "p_typeA": None, "sig": "",
            })
            continue

        # バースト検出・分類
        burst_df = detect_bursts(
            ratings_topic,
            speed_multiplier=args.burst_speed_multiplier,
            min_count=args.burst_min_count,
        )
        if burst_df.empty:
            print("  → no bursts, skipping")
            results.append({
                "topic": topic_name, "n_notes": n_notes,
                "n_typeA": 0, "n_typeB": 0,
                "beta_typeA": None, "p_typeA": None, "sig": "",
            })
            continue

        burst_classified = classify_burst_type(burst_df, polarity_df, threshold=args.burst_threshold)

        # 特徴量 & 回帰
        feat_df = compute_features_for_regression(
            ratings_topic, burst_classified, history_df, quality,
            trend_min_evals=args.trend_min_evals,
        )
        beta1, p1, n, n_a, n_b = run_regression(feat_df)

        sig = ""
        if p1 is not None:
            if p1 < 0.001: sig = "***"
            elif p1 < 0.01: sig = "**"
            elif p1 < 0.05: sig = "*"

        print(f"  ⏱ {topic_name}: {_fmt(time.time() - t_topic)}")
        results.append({
            "topic": topic_name,
            "n_notes": n,
            "n_typeA": n_a,
            "n_typeB": n_b,
            "beta_typeA": beta1,
            "p_typeA": p1,
            "sig": sig,
        })

    # ── 比較表出力 ────────────────────────────────────
    res_df = pd.DataFrame(results)
    suffix = args.chunk_suffix
    res_df.to_csv(OUT_DIR / f"topic_comparison{suffix}.csv", index=False)

    print("\n")
    print("=" * 72)
    print("  TOPIC COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'Topic':<20} {'Notes':>6} {'TypeA':>6} {'TypeB':>6} {'β(TypeA)':>10} {'p-value':>10} {'Sig':>4}")
    print("-" * 72)
    for _, row in res_df.iterrows():
        b = f"{row['beta_typeA']:+.4f}" if row["beta_typeA"] is not None else "   N/A"
        p = f"{row['p_typeA']:.4f}" if row["p_typeA"] is not None else "   N/A"
        print(f"  {row['topic']:<20} {row['n_notes']:>6} {row['n_typeA']:>6} {row['n_typeB']:>6} {b:>10} {p:>10} {row['sig']:>4}")
    print("=" * 72)
    print(f"\n  Total time: {_fmt(time.time() - t_total)}")
    print(f"  Output: {OUT_DIR / f'topic_comparison{suffix}.csv'}")


if __name__ == "__main__":
    main()
