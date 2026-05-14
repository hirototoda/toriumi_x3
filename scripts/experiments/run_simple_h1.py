"""
Simple 版パイプラインの H1 検証版エントリポイント

H1 仮説:
  元 simple_regression の β_typeA = -1.13 (負) は、
  TypeA バーストが NOT_HELPFUL 一斉ではなく HELPFUL 一斉が多いことに由来する。
  → type_a / type_b をそれぞれ {helpful, nothelp} 方向で切り、4 ダミーで再推定。

既存 src/simple/ と scripts/run_simple.py は一切変更していない。
本スクリプトの出力も別ファイル名なので、元結果は上書きされない:
    data/processed/simple_h1_features.csv
    data/processed/simple_h1_bursts.csv
    data/processed/simple_h1_regression.txt

実行例:
    python scripts/run_simple_h1.py --sample-frac 0.30
    python scripts/run_simple_h1.py --sample-frac 0.05   # 動作確認
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# 共通部分: load / topic / polarity / quality は simple から流用
from src.simple.load       import (
    load_notes, sample_note_ids, load_ratings_for_notes, load_history_for_notes,
)
from src.simple.topic      import filter_political_notes
from src.simple.polarity   import compute_polarity
from src.simple.quality    import quality_score

# H1 差分: burst 検出に direction を足し、回帰を 4 ダミー版に差し替え
from src.simple_h1.burst      import detect_bursts_with_direction, classify_burst_type
from src.simple_h1.regression import build_features_h1, run_logit_h1

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-frac",       type=float, default=0.30)
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--polarity-first-n",  type=int,   default=50)
    p.add_argument("--burst-multiplier",  type=float, default=3.0)
    p.add_argument("--burst-min-count",   type=int,   default=5)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. notes → 政治フィルタ → サンプリング
    notes      = load_notes(RAW_DIR)
    political  = filter_political_notes(notes)
    sample_ids = sample_note_ids(political, frac=args.sample_frac, seed=args.seed)

    # 2. ratings / history をサンプル noteId に絞って読む
    ratings = load_ratings_for_notes(RAW_DIR, sample_ids)
    history = load_history_for_notes(RAW_DIR, sample_ids)
    if ratings.empty:
        print("\nERROR: no ratings for sampled notes. Aborting.", file=sys.stderr)
        sys.exit(1)

    # 3. polarity (simple と同じ)
    polarity = compute_polarity(ratings, first_n=args.polarity_first_n, seed=args.seed)

    # 4. burst 検出 (direction 付き) → A/B 分類 (共通)
    bursts = detect_bursts_with_direction(
        ratings,
        speed_multiplier=args.burst_multiplier,
        min_count=args.burst_min_count,
    )
    bursts = classify_burst_type(bursts, polarity)
    if not bursts.empty:
        bursts.drop(columns=["burst_raters", "burst_levels"]).to_csv(
            OUT_DIR / "simple_h1_bursts.csv", index=False,
        )

    # 5. quality (simple と同じ)
    sampled_notes = political[political["noteId"].isin(sample_ids)]
    quality = quality_score(sampled_notes)

    # 6. 特徴量 (4 ダミー) + 回帰
    feat = build_features_h1(ratings, bursts, history, quality)
    if feat.empty:
        print("\nERROR: no features built. Aborting.", file=sys.stderr)
        sys.exit(1)
    feat.to_csv(OUT_DIR / "simple_h1_features.csv", index=False)
    res = run_logit_h1(feat)

    (OUT_DIR / "simple_h1_regression.txt").write_text(str(res.summary()))

    print(f"\n[done] total {_fmt(time.time() - t0)}, outputs: {OUT_DIR}/simple_h1_*")


if __name__ == "__main__":
    main()
