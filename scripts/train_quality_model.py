"""
品質モデルのサンプリング・学習スクリプト

使い方:
    # 1) ノートを 200 件サンプリング
    python scripts/train_quality_model.py sample \
        --n 200 --seed 42 --out data/labels/sample_notes.csv

    # 2) Claude Code にラベル付けを依頼:
    #    「data/labels/sample_notes.csv の label 列を埋めて」
    #    → Claude Code が summary を読み、各行に 0 または 1 を書く

    # 3) 学習
    python scripts/train_quality_model.py train \
        --labels data/labels/sample_notes.csv \
        --out models/quality_model.joblib
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.io.load_data import load_notes
from src.step5_target.quality_features import extract_quality_features
from src.step5_target.quality_model import save_model, train_quality_model
from src.step5_target.validation import compute_agreement_rate

RAW_DIR = ROOT / "data" / "raw"


def cmd_sample(args: argparse.Namespace) -> None:
    print(f"[sample] loading notes from {RAW_DIR} ...")
    notes_df = load_notes(RAW_DIR)
    notes_unique = notes_df.drop_duplicates("noteId")
    print(f"[sample] {len(notes_unique):,} unique notes")

    n = min(args.n, len(notes_unique))
    sampled = notes_unique.sample(n=n, random_state=args.seed).copy()
    sampled = sampled[["noteId", "createdAtMillis", "summary"]]
    sampled["label"] = ""  # Claude Code が埋める列

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(out, index=False)
    print(f"[sample] wrote {len(sampled)} rows to {out}")
    print(f"[sample] 次のステップ: Claude Code に label 列の記入を依頼してください")


def cmd_train(args: argparse.Namespace) -> None:
    labels_path = Path(args.labels)
    print(f"[train] loading labels from {labels_path} ...")
    df = pd.read_csv(labels_path, dtype={"noteId": str})

    if "label" not in df.columns:
        raise ValueError(f"{labels_path} に label 列がありません")
    df = df[df["label"].notna()]
    df = df[df["label"].astype(str).str.strip() != ""]
    df["label"] = df["label"].astype(int)
    print(f"[train] {len(df)} labeled notes (0={int((df['label']==0).sum())}, 1={int((df['label']==1).sum())})")

    features = extract_quality_features(df[["noteId", "summary"]])
    labels = df.set_index("noteId")["label"]

    bundle = train_quality_model(features, labels)

    # ホールドアウト評価（reuse validation.compute_agreement_rate）
    rng = np.random.default_rng(0)
    idx = np.arange(len(labels))
    rng.shuffle(idx)
    n_test = max(1, len(idx) // 5)
    test_ids = labels.index[idx[:n_test]]
    train_ids = labels.index[idx[n_test:]]

    sub_feat = features.loc[train_ids]
    sub_lab = labels.loc[train_ids]
    try:
        sub_bundle = train_quality_model(sub_feat, sub_lab)
        from src.step5_target.quality_model import predict_quality
        test_notes = df[df["noteId"].isin(test_ids)][["noteId", "summary"]]
        q_test = predict_quality(test_notes, sub_bundle)
        agreement = compute_agreement_rate(q_test, labels.loc[test_ids])
        print(f"[train] holdout agreement (n={len(test_ids)}): {agreement:.3f}")
    except Exception as e:
        print(f"[train] holdout eval skipped: {e}")

    print(f"[train] trained_at: {bundle['trained_at']}")
    print(f"[train] n_train:    {bundle['n_train']}")
    print(f"[train] cv_auc:     {bundle['cv_auc']:.3f}")
    print(f"[train] coef:       {dict(zip(bundle['feature_names'], bundle['clf'].coef_[0].round(3)))}")
    print(f"[train] intercept:  {bundle['clf'].intercept_[0]:.3f}")

    save_model(bundle, args.out)
    print(f"[train] saved model to {args.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quality model: sample / train")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_sample = sub.add_parser("sample", help="ノートをランダム抽出してラベル用 CSV を出力")
    p_sample.add_argument("--n", type=int, default=200)
    p_sample.add_argument("--seed", type=int, default=42)
    p_sample.add_argument("--out", type=str, default="data/labels/sample_notes.csv")
    p_sample.set_defaults(func=cmd_sample)

    p_train = sub.add_parser("train", help="ラベル済み CSV からモデル学習")
    p_train.add_argument("--labels", type=str, default="data/labels/sample_notes.csv")
    p_train.add_argument("--out", type=str, default="models/quality_model.joblib")
    p_train.set_defaults(func=cmd_train)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
