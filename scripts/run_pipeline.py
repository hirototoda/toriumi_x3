"""
パイプライン統合スクリプト: Step 1〜5 を一気通貫で実行する。

使い方:
  python scripts/run_pipeline.py                     # 全データ
  python scripts/run_pipeline.py --nrows 200000      # 先頭20万行で高速テスト

必要なデータ (data/raw/ に配置):
  - ratings-*.tsv    (評価データ)
  - notes-*.tsv      (ノートデータ)
  - noteStatusHistory-*.tsv (ステータス履歴)
"""

import argparse
import sys
import time
from pathlib import Path

# プロジェクトルートをPATHに追加
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _fmt(seconds: float) -> str:
    """秒を 1m23s 形式にフォーマット"""
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"

from src.io.load_data import load_ratings, load_notes, load_status_history
from src.step1_preprocess.polarity import compute_polarity
from src.step1_preprocess.merge_data import merge_tsv_files
from src.step1_preprocess.filter import filter_by_rating_count
from src.step2_topic.classify import classify_political_topics
from src.step3_burst.detect import detect_bursts
from src.step3_burst.classify_burst import classify_burst_type
from src.step4_regression.features import compute_features_for_regression
from src.step4_regression.logistic import fit_logistic_regression
from src.step5_target.quality_score import compute_quality_score
from src.step5_target.target_extraction import extract_target_notes


RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def main():
    parser = argparse.ArgumentParser(description="Community Notes pipeline")
    parser.add_argument("--nrows", type=int, default=None,
                        help="ratings の読み込み行数制限（notes/history は全量読む）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_total = time.time()

    # ─── Step 0: データ読み込み ───────────────────────
    t0 = time.time()
    print("\n[Step 0] Loading data...")
    ratings_df = load_ratings(RAW_DIR, nrows=args.nrows)

    try:
        notes_df = load_notes(RAW_DIR)  # notes は全量読む（shard が異なるため）
        has_notes = True
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  → notes なしで続行 (Step2 topic filter スキップ, Step5 Q=0)")
        has_notes = False
        notes_df = None

    try:
        history_df = load_status_history(RAW_DIR)  # history は全量読む
        has_history = True
    except FileNotFoundError as e:
        print(f"  WARNING: {e}")
        print("  → statusHistory なしで続行 (deleted はランダム割当)")
        has_history = False
        history_df = None

    print(f"  ⏱ Step 0: {_fmt(time.time() - t0)}")

    # ─── Step 1: Polarity計算 + フィルタ ─────────────
    t1 = time.time()
    print("\n[Step 1] Computing polarity...")
    polarity_df = compute_polarity(ratings_df)

    print("\n[Step 1] Filtering notes (>= 20 ratings)...")
    ratings_filtered = filter_by_rating_count(ratings_df)

    print(f"  ⏱ Step 1: {_fmt(time.time() - t1)}")

    # ─── Step 2: トピック分類 ────────────────────────
    t2 = time.time()
    print("\n[Step 2] Topic classification...")
    if has_notes:
        # ノート単位でユニークにして分類
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

    # ─── Step 3: バースト検出 & 分類 ─────────────────
    t3 = time.time()
    print("\n[Step 3] Burst detection...")
    burst_df = detect_bursts(ratings_political)

    print("\n[Step 3] Burst classification (TypeA/TypeB)...")
    burst_classified = classify_burst_type(burst_df, polarity_df)

    # バースト結果を保存
    if not burst_classified.empty:
        burst_out = burst_classified.drop(columns=["burst_raters"])
        burst_out.to_csv(OUT_DIR / "bursts.csv", index=False)

    print(f"  ⏱ Step 3: {_fmt(time.time() - t3)}")

    # ─── Step 4: 特徴量構築 & ロジスティック回帰 ─────
    t4 = time.time()
    print("\n[Step 4] Building features...")

    # 品質スコア（notes がある場合のみ）
    if has_notes:
        quality = compute_quality_score(notes_unique)
    else:
        quality = None

    # statusHistory がない場合のダミー
    if not has_history:
        import pandas as pd
        note_ids = ratings_political["noteId"].unique()
        history_df = pd.DataFrame({
            "noteId": note_ids,
            "currentStatus": "UNKNOWN",
        })

    feat_df = compute_features_for_regression(
        ratings_political, burst_classified, history_df, quality
    )

    # 回帰を実行（type_aとtype_bに分散がある場合のみ）
    print("\n[Step 4] Logistic regression...")
    if feat_df["type_a"].sum() == 0 and feat_df["type_b"].sum() == 0:
        print("  WARNING: No bursts found. Regression skipped.")
        reg_result = None
    elif feat_df["deleted"].nunique() < 2:
        print("  WARNING: deleted has no variance. Regression skipped.")
        reg_result = None
    else:
        reg_result = fit_logistic_regression(feat_df)

    print(f"  ⏱ Step 4: {_fmt(time.time() - t4)}")

    # ─── Step 5: ターゲット抽出 ──────────────────────
    t5 = time.time()
    print("\n[Step 5] Target extraction...")
    targets = extract_target_notes(feat_df)
    targets.to_csv(OUT_DIR / "target_notes.csv", index=False)

    print(f"  ⏱ Step 5: {_fmt(time.time() - t5)}")

    # ─── 結果サマリー ────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "=" * 60)
    print(f"  Pipeline Complete  (total: {_fmt(elapsed)})")
    print("=" * 60)
    print(f"  Total ratings processed: {len(ratings_political):,}")
    print(f"  Raters with polarity:    {len(polarity_df):,}")
    print(f"  Bursts detected:         {len(burst_classified):,}")
    if not burst_classified.empty:
        print(f"    TypeA (faction):       {(burst_classified['burst_type']=='A').sum()}")
        print(f"    TypeB (natural):       {(burst_classified['burst_type']=='B').sum()}")
    print(f"  Target notes:            {len(targets)}")
    print(f"\n  Output: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
