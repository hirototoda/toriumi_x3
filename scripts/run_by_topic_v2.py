"""
v2: トピック別パイプライン (control 拡張版回帰)

run_by_topic.py との違い:
  - 特徴量に ratings_count, bridging_score を追加
  - トピック × モデル(M0/M1/M2) の比較表を出力
  - features_by_topic_v2*.csv に新 control 入りの特徴量を保存

bridging_score は polarity_df から自前計算するので追加データ不要。
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

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
from src.step4_regression_v2.features import compute_features_for_regression_v2
from src.step4_regression_v2.logistic import fit_logistic_regression_v2, fits_to_rows


RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

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
    notes_unique = notes_df.drop_duplicates("noteId")
    text = notes_unique["summary"].fillna("").str.lower()
    pattern = "|".join(keywords)
    mask = text.str.contains(pattern, regex=True)
    matched_notes = notes_unique.loc[mask, "noteId"]
    return ratings_df[ratings_df["noteId"].isin(matched_notes)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-rating-files", type=int, default=None)
    parser.add_argument("--file-offset", type=int, default=0)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--chunk-suffix", type=str, default="")
    parser.add_argument("--min-ratings", type=int, default=10)
    parser.add_argument("--topics-json", type=str, default=None)
    parser.add_argument("--polarity-first-n", type=int, default=50)
    parser.add_argument("--burst-speed-multiplier", type=float, default=3.0)
    parser.add_argument("--burst-min-count", type=int, default=5)
    parser.add_argument("--burst-threshold", type=float, default=None)
    parser.add_argument("--trend-min-evals", type=int, default=4)
    parser.add_argument("--quality-model-path", type=str, default=None)
    args = parser.parse_args()

    global TOPICS
    if args.topics_json:
        import json
        with open(args.topics_json) as f:
            TOPICS = json.load(f)
        print(f"  Topics loaded from {args.topics_json}: {list(TOPICS.keys())}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.chunk_suffix
    t_total = time.time()

    # ── データ読み込み ───────────────────────────────
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

    # ── Step 1: Polarity + フィルタ ──────────────────
    t1 = time.time()
    print("\n[Step 1] Computing polarity...")
    polarity_df = compute_polarity_cached(
        ratings_df, first_n=args.polarity_first_n, ratings_tag=r_tag,
    )
    print(f"\n[Step 1] Filtering (>= {args.min_ratings} ratings)...")
    ratings_filtered = filter_by_rating_count(ratings_df, min_count=args.min_ratings)
    print(f"  ⏱ Step 1: {_fmt(time.time() - t1)}")

    # ── 品質スコア ───────────────────────────────────
    print("\n[Step 5] Computing quality scores...")
    notes_unique = notes_df.drop_duplicates("noteId")
    quality = compute_quality_cached(
        notes_unique, model_path=args.quality_model_path, notes_tag=n_tag,
    )

    # ── トピック別実行 ───────────────────────────────
    all_model_rows: list[dict] = []
    feat_dfs = []

    for topic_name, keywords in TOPICS.items():
        t_topic = time.time()
        print(f"\n{'='*60}\n  Topic: {topic_name}\n{'='*60}")

        ratings_topic = filter_by_topic(ratings_filtered, notes_df, keywords)
        n_notes = ratings_topic["noteId"].nunique()
        print(f"  notes: {n_notes}, ratings: {len(ratings_topic):,}")

        if n_notes < 10:
            print("  → too few notes, skipping")
            all_model_rows.append({"topic": topic_name, "model": "M0_base", "n": n_notes,
                                   "note": "skip: too few notes"})
            continue

        burst_df = detect_bursts(
            ratings_topic,
            speed_multiplier=args.burst_speed_multiplier,
            min_count=args.burst_min_count,
        )
        if burst_df.empty:
            print("  → no bursts, skipping")
            all_model_rows.append({"topic": topic_name, "model": "M0_base", "n": n_notes,
                                   "note": "skip: no bursts"})
            continue

        burst_classified = classify_burst_type(burst_df, polarity_df, threshold=args.burst_threshold)

        feat_df = compute_features_for_regression_v2(
            ratings_topic, burst_classified, history_df, quality,
            polarity_df=polarity_df,
            trend_min_evals=args.trend_min_evals,
        )

        if not feat_df.empty:
            feat_with_topic = feat_df.copy()
            feat_with_topic.insert(0, "topic", topic_name)
            feat_dfs.append(feat_with_topic)

        if feat_df.empty or feat_df["type_a"].sum() == 0:
            print("  → no type_a in features, skipping regression")
            all_model_rows.append({"topic": topic_name, "model": "M0_base",
                                   "n": len(feat_df), "note": "skip: no type_a"})
            continue

        fits = fit_logistic_regression_v2(feat_df, print_results=True)
        all_model_rows.extend(fits_to_rows(fits, topic=topic_name))
        print(f"  ⏱ {topic_name}: {_fmt(time.time() - t_topic)}")

    # ── 出力 ────────────────────────────────────────
    res_df = pd.DataFrame(all_model_rows)
    res_df.to_csv(OUT_DIR / f"topic_models_v2{suffix}.csv", index=False)

    if feat_dfs:
        all_feats = pd.concat(feat_dfs, ignore_index=True)
        all_feats.to_csv(OUT_DIR / f"features_by_topic_v2{suffix}.csv", index=False)
        print(f"\n  features_by_topic_v2{suffix}.csv: {len(all_feats):,} rows "
              f"({all_feats['topic'].nunique()} topics)")

    # ── 比較表 (β_typeA) ─────────────────────────────
    print("\n" + "=" * 90)
    print("  TOPIC × MODEL  COMPARISON (β_typeA)")
    print("=" * 90)
    print(f"  {'Topic':<20} {'Model':<24} {'n':>6} {'β_typeA':>10} {'p':>10} {'sig':>5}")
    print("-" * 90)
    for _, row in res_df.iterrows():
        beta = row.get("beta_type_a")
        p = row.get("p_type_a")
        sig = row.get("sig_typeA", "")
        b = f"{beta:+.4f}" if isinstance(beta, (int, float)) and pd.notna(beta) else "   N/A"
        ps = f"{p:.4f}" if isinstance(p, (int, float)) and pd.notna(p) else "   N/A"
        print(f"  {str(row.get('topic','')):<20} {str(row.get('model','')):<24} "
              f"{int(row.get('n', 0)):>6} {b:>10} {ps:>10} {sig:>5}")
    print("=" * 90)
    print(f"\n  Total time: {_fmt(time.time() - t_total)}")
    print(f"  Output: {OUT_DIR / f'topic_models_v2{suffix}.csv'}")


if __name__ == "__main__":
    main()
