"""
k=3 で「左陣営 / 中央（穏健） / 右陣営」に分類し、
各手法を比較する。

手法:
  A. KMeans (k=3)
  B. GMM (k=3) — クラスタごとにサイズ・形状が異なってもよい
  C. 閾値ベース — polarity_x の分布からパーセンタイルで区切る
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RANDOM_SEED = 42


def get_2d_coords(nrows=500_000):
    path = next(RAW_DIR.glob("ratings*.tsv"))
    df = pd.read_csv(
        path, sep="\t",
        usecols=["noteId", "raterParticipantId", "helpfulnessLevel"],
        dtype={"noteId": str, "raterParticipantId": str},
        nrows=nrows,
    )
    level_map = {"HELPFUL": 1.0, "SOMEWHAT_HELPFUL": 0.0, "NOT_HELPFUL": -1.0}
    df["score"] = df["helpfulnessLevel"].map(level_map)
    df = df.dropna(subset=["score"])

    counts = df.groupby("raterParticipantId").size()
    active = counts[counts >= 10].index
    df = df[df["raterParticipantId"].isin(active)]

    rng = np.random.RandomState(RANDOM_SEED)
    unique = df["raterParticipantId"].unique()
    n = min(500, len(unique))
    sampled = rng.choice(unique, size=n, replace=False)
    df = df[df["raterParticipantId"].isin(set(sampled))]

    rater_ids = sorted(df["raterParticipantId"].unique())
    note_ids = sorted(df["noteId"].unique())
    ri = {r: i for i, r in enumerate(rater_ids)}
    ni = {n: i for i, n in enumerate(note_ids)}
    mat = csr_matrix(
        (df["score"].values,
         (df["raterParticipantId"].map(ri).values,
          df["noteId"].map(ni).values)),
        shape=(len(rater_ids), len(note_ids)),
    )
    col_mean = np.array(mat.mean(axis=0)).flatten()
    mat_c = csr_matrix(mat.toarray() - col_mean[np.newaxis, :])
    u, s, _ = svds(mat_c, k=2)
    order = np.argsort(-s)
    return u[:, order] * s[order][np.newaxis, :]


def main():
    coords = get_2d_coords()
    px = coords[:, 0]

    # ── 手法A: KMeans k=3 ─────────────────────────────
    km = KMeans(n_clusters=3, random_state=RANDOM_SEED, n_init=10)
    labels_km = km.fit_predict(coords)
    sil_km = silhouette_score(coords, labels_km)

    # ── 手法B: GMM k=3 ────────────────────────────────
    gmm = GaussianMixture(n_components=3, random_state=RANDOM_SEED, covariance_type="full")
    gmm.fit(coords)
    labels_gmm = gmm.predict(coords)
    sil_gmm = silhouette_score(coords, labels_gmm)

    # ── 手法C: polarity_x パーセンタイルで3分割 ───────
    p20 = np.percentile(px, 20)
    p80 = np.percentile(px, 80)
    labels_pct = np.where(px < p20, 0, np.where(px > p80, 2, 1))
    sil_pct = silhouette_score(coords, labels_pct)

    # ── プロット ──────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    colors = ["#3498db", "#95a5a6", "#e74c3c"]  # blue=left, gray=center, red=right
    camp_names = ["Left camp", "Center", "Right camp"]

    for ax, labels, title, sil in [
        (axes[0], labels_km,  "A: KMeans (k=3)", sil_km),
        (axes[1], labels_gmm, "B: GMM (k=3)",    sil_gmm),
        (axes[2], labels_pct, "C: Percentile (20/80)", sil_pct),
    ]:
        # polarity_x の平均でクラスタに左/中/右のラベルを割り当て
        cluster_means = [px[labels == c].mean() for c in sorted(set(labels))]
        order = np.argsort(cluster_means)  # 左→中→右

        for rank, c in enumerate(order):
            mask = labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=colors[rank], alpha=0.5, s=20,
                       label=f"{camp_names[rank]} (n={mask.sum()})")
        ax.set_xlabel("polarity_x")
        ax.set_ylabel("polarity_y")
        ax.set_title(f"{title}\nsilhouette={sil:.3f}")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", lw=0.3)
        ax.axvline(0, color="gray", lw=0.3)

    plt.tight_layout()
    out_path = OUT_DIR / "feasibility_k3_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")

    # ── 結果表示 ─────────────────────────────────────
    print()
    print("=" * 60)
    print("  Method comparison (k=3)")
    print("=" * 60)
    for name, labels, sil in [
        ("A: KMeans",     labels_km,  sil_km),
        ("B: GMM",        labels_gmm, sil_gmm),
        ("C: Percentile", labels_pct, sil_pct),
    ]:
        cluster_means = [px[labels == c].mean() for c in sorted(set(labels))]
        order = np.argsort(cluster_means)
        sizes = [int((labels == c).sum()) for c in order]
        print(f"\n  {name}  (silhouette={sil:.3f})")
        print(f"    Left:   {sizes[0]:>4} raters  (mean polarity_x={cluster_means[order[0]]:+.2f})")
        print(f"    Center: {sizes[1]:>4} raters  (mean polarity_x={cluster_means[order[1]]:+.2f})")
        print(f"    Right:  {sizes[2]:>4} raters  (mean polarity_x={cluster_means[order[2]]:+.2f})")
    print()
    print(f"  Plot: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
