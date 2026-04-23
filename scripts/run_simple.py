"""
Simple版パイプライン (CS授業用) のエントリポイント

実行例:
    # 全 notes の 30% サンプルで実行
    python scripts/run_simple.py --sample-frac 0.30

    # 動作確認 (5%サンプル, 高速)
    python scripts/run_simple.py --sample-frac 0.05

出力 (data/processed/ 配下):
    simple_features.csv      回帰の入力 (note 単位)
    simple_bursts.csv        検出されたバースト一覧
    simple_regression.txt    回帰の summary

設計はノート README/docs を参照. ロジック概要:
    1. notes 全件読 → noteId を frac サンプリング
    2. ratings / history を「サンプリング noteId のみ」で読み込む
    3. 政治トピック判定 (substring + 単語境界)
    4. polarity = TruncatedSVD(rater × note 行列, k=2)
    5. burst 検出 + TypeA/B 分類
    6. quality = LLM ラベル学習済みの線形和 → sigmoid
    7. ロジスティック回帰 (相関行列 + VIF も print)
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.simple.load       import (
    load_notes, sample_note_ids, load_ratings_for_notes, load_history_for_notes,
)
from src.simple.topic      import filter_political_notes
from src.simple.polarity   import compute_polarity
from src.simple.burst      import detect_bursts, classify_burst_type
from src.simple.quality    import quality_score
from src.simple.regression import build_features, run_logit

RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"


def _fmt(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sample-frac",       type=float, default=0.30, help="noteId サンプリング比 (0..1)")
    p.add_argument("--seed",              type=int,   default=42)
    p.add_argument("--polarity-first-n",  type=int,   default=50,   help="rater 当たり先頭 N 件で polarity 固定")
    p.add_argument("--burst-multiplier",  type=float, default=3.0)
    p.add_argument("--burst-min-count",   type=int,   default=5)
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. notes 全件 → サンプリング
    notes = load_notes(RAW_DIR)
    political = filter_political_notes(notes)
    sample_ids = sample_note_ids(political, frac=args.sample_frac, seed=args.seed)

    # 2. ratings / history をサンプル noteId に絞って読む
    ratings = load_ratings_for_notes(RAW_DIR, sample_ids)
    history = load_history_for_notes(RAW_DIR, sample_ids)
    if ratings.empty:
        print("\nERROR: no ratings for sampled notes. Aborting.")
        return

    # 3. polarity
    polarity = compute_polarity(ratings, first_n=args.polarity_first_n, seed=args.seed)

    # 4. burst 検出 + 分類
    bursts = detect_bursts(
        ratings, speed_multiplier=args.burst_multiplier, min_count=args.burst_min_count,
    )
    bursts = classify_burst_type(bursts, polarity)
    if not bursts.empty:
        bursts.drop(columns=["burst_raters"]).to_csv(OUT_DIR / "simple_bursts.csv", index=False)

    # 5. quality (LLM 学習済重みでスコアリング)
    sampled_notes = political[political["noteId"].isin(sample_ids)]
    quality = quality_score(sampled_notes)

    # 6. 特徴量 + 回帰
    feat = build_features(ratings, bursts, history, quality)
    if feat.empty:
        print("\nERROR: no features built. Aborting.")
        return
    feat.to_csv(OUT_DIR / "simple_features.csv", index=False)
    res = run_logit(feat)

    # summary をテキストでも保存
    (OUT_DIR / "simple_regression.txt").write_text(str(res.summary()))

    print(f"\n[done] total {_fmt(time.time() - t0)}, outputs: {OUT_DIR}/simple_*")


if __name__ == "__main__":
    main()
