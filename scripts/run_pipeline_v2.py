"""
パイプライン統合スクリプト v2: control 拡張版の回帰を流す。

run_pipeline.py との違い:
  - 特徴量に ratings_count と bridging_score (polarity 多様性 = CN MF noteScore の自前近似) を追加
  - M0/M1/M2 の 3 モデル比較を出力 (data/processed/regression_models_v2*.csv)

bridging_score は polarity_df から自前計算するので追加データ不要。

使い方:
  python scripts/run_pipeline_v2.py --nrows 200000     # 動作確認
  python scripts/run_pipeline_v2.py                    # 全データ (チャンク前提)
"""

import argparse
import sys
import time
from pathlib import Path

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
from src.step2_topic.classify import classify_political_topics
from src.step3_burst.detect import detect_bursts
from src.step3_burst.classify_burst import classify_burst_type
from src.step4_regression_v2.features import compute_features_for_regression_v2
from src.step4_regression_v2.logistic import fit_logistic_regression_v2, fits_to_rows
from src.step5_target.target_extraction import extract_target_notes


RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Community Notes pipeline v2 (with bridging_score control)")
    parser.add_argument("--nrows", type=int, default=None)
    parser.add_argument("--max-rating-files", type=int, default=None)
    parser.add_argument("--file-offset", type=int, default=0)
    parser.add_argument("--skip-rows", type=int, default=0)
    parser.add_argument("--chunk-suffix", type=str, default="")
    parser.add_argument("--polarity-first-n", type=int, default=50)
    parser.add_argument("--min-rating-count", type=int, default=20)
    parser.add_argument("--burst-speed-multiplier", type=float, default=3.0)
    parser.add_argument("--burst-min-count", type=int, default=5)
    parser.add_argument("--burst-threshold", type=float, default=None)
    parser.add_argument("--trend-min-evals", type=int, default=4)
    parser.add_argument("--target-top-percent", type=int, default=25)
    parser.add_argument("--quality-model-path", type=str, default=None)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = args.chunk_suffix
    t_total = time.time()

    # ── Step 0: データ読み込み ─────────────────────────
    print("\n[Step 0] Loading data...")
    t0 = time.time()
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

    try:
        notes_df = load_notes_cached(RAW_DIR)
        n_tag = notes_cache_tag(RAW_DIR)
        has_notes = True
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        has_notes = False
        notes_df = None
        n_tag = None

    try:
        history_df = load_status_history_cached(RAW_DIR)
        has_history = True
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        has_history = False
        history_df = None

    print(f"  ⏱ Step 0: {_fmt(time.time() - t0)}")

    # ── Step 1: Polarity + フィルタ ──────────────────
    t1 = time.time()
    print("\n[Step 1] Computing polarity...")
    polarity_df = compute_polarity_cached(
        ratings_df, first_n=args.polarity_first_n, ratings_tag=r_tag,
    )
    print(f"\n[Step 1] Filtering notes (>= {args.min_rating_count} ratings)...")
    ratings_filtered = filter_by_rating_count(ratings_df, min_count=args.min_rating_count)
    print(f"  ⏱ Step 1: {_fmt(time.time() - t1)}")

    # ── Step 2: トピック分類 ─────────────────────────
    t2 = time.time()
    print("\n[Step 2] Topic classification...")
    if has_notes:
        notes_unique = notes_df.drop_duplicates("noteId")
        notes_classified = classify_political_topics(notes_unique)
        political_note_ids = notes_classified[notes_classified["is_political"]]["noteId"]
        ratings_political = ratings_filtered[
            ratings_filtered["noteId"].isin(political_note_ids)
        ]
        print(f"    political ratings: {len(ratings_political):,} rows")
    else:
        print("    skipped (no notes file)")
        ratings_political = ratings_filtered
        notes_classified = None
    print(f"  ⏱ Step 2: {_fmt(time.time() - t2)}")

    # ── Step 3: バースト検出 & 分類 ──────────────────
    t3 = time.time()
    print("\n[Step 3] Burst detection...")
    burst_df = detect_bursts(
        ratings_political,
        speed_multiplier=args.burst_speed_multiplier,
        min_count=args.burst_min_count,
    )
    print("\n[Step 3] Burst classification (TypeA/TypeB)...")
    burst_classified = classify_burst_type(burst_df, polarity_df, threshold=args.burst_threshold)
    if not burst_classified.empty:
        burst_classified.drop(columns=["burst_raters"]).to_csv(
            OUT_DIR / f"bursts_v2{suffix}.csv", index=False,
        )
    print(f"  ⏱ Step 3: {_fmt(time.time() - t3)}")

    # ── Step 4 v2: 特徴量 + 回帰 (3 モデル) ─────────
    t4 = time.time()
    print("\n[Step 4 v2] Building features (with ratings_count, bridging_score)...")
    if has_notes:
        quality = compute_quality_cached(
            notes_unique, model_path=args.quality_model_path, notes_tag=n_tag,
        )
    else:
        quality = None

    if not has_history:
        import pandas as pd
        note_ids = ratings_political["noteId"].unique()
        history_df = pd.DataFrame({"noteId": note_ids, "currentStatus": "UNKNOWN"})

    feat_df = compute_features_for_regression_v2(
        ratings_political, burst_classified, history_df, quality,
        polarity_df=polarity_df,
        trend_min_evals=args.trend_min_evals,
    )
    feat_df.to_csv(OUT_DIR / f"features_v2{suffix}.csv", index=False)

    print("\n[Step 4 v2] Logistic regression (M0 / M1 / M2)...")
    if feat_df["type_a"].sum() == 0 and feat_df["type_b"].sum() == 0:
        print("  WARNING: No bursts found. Regression skipped.")
        fits = {}
    elif feat_df["deleted"].nunique() < 2:
        print("  WARNING: deleted has no variance. Regression skipped.")
        fits = {}
    else:
        fits = fit_logistic_regression_v2(feat_df)

    if fits:
        import pandas as pd
        rows = fits_to_rows(fits)
        pd.DataFrame(rows).to_csv(OUT_DIR / f"regression_models_v2{suffix}.csv", index=False)
    print(f"  ⏱ Step 4 v2: {_fmt(time.time() - t4)}")

    # ── Step 5: ターゲット抽出 ───────────────────────
    t5 = time.time()
    print("\n[Step 5] Target extraction...")
    targets = extract_target_notes(feat_df, top_percent=args.target_top_percent)
    targets.to_csv(OUT_DIR / f"target_notes_v2{suffix}.csv", index=False)
    print(f"  ⏱ Step 5: {_fmt(time.time() - t5)}")

    # ── サマリー ──────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"  Pipeline v2 Complete  (total: {_fmt(elapsed)})")
    print("=" * 60)
    print(f"  Total ratings processed: {len(ratings_political):,}")
    print(f"  Bursts detected:         {len(burst_classified):,}")
    print(f"  Feature rows (v2):       {len(feat_df):,}")
    print(f"\n  Output: {OUT_DIR}/  (suffix='{suffix}')")
    print("=" * 60)


if __name__ == "__main__":
    main()
