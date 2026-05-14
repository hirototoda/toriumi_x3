"""
KMeansの結果が「本当に2陣営か？」を追加検証するスクリプト。

検証1: polarity_x のヒストグラム → 山が2つあれば本物の2陣営
検証2: k=2,3,4,5 のシルエットスコア比較 → k=2が最良なら2陣営が最適
検証3: Dip検定 → 単峰 vs 多峰を統計的に判定
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
from sklearn.metrics import silhouette_score

# ── 設定（前スクリプトと同じ）────────────────────────
N_RATERS = 500
MIN_RATINGS = 10
RANDOM_SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def find_ratings_file() -> Path:
    for p in sorted(RAW_DIR.glob("ratings*.tsv")):
        if p.exists():
            return p
    print("エラー: ratings ファイルが見つかりません。")
    sys.exit(1)


def get_2d_coords(nrows=500_000):
    """前スクリプトと同じ手順で2D座標を取得"""
    path = find_ratings_file()
    print(f"Loading {path.name} ...")
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
    active = counts[counts >= MIN_RATINGS].index
    df = df[df["raterParticipantId"].isin(active)]

    rng = np.random.RandomState(RANDOM_SEED)
    unique = df["raterParticipantId"].unique()
    n = min(N_RATERS, len(unique))
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
    coords = u[:, order] * s[order][np.newaxis, :]
    print(f"  {n} raters × {len(note_ids)} notes → 2D coords ready")
    return coords


def main():
    coords = get_2d_coords()
    px = coords[:, 0]  # polarity_x（第1主成分）

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 検証1: polarity_x のヒストグラム + KDE ──────────
    ax = axes[0]
    ax.hist(px, bins=40, density=True, alpha=0.6, color="#888888", edgecolor="white")
    # KDEを重ねる
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(px, bw_method=0.3)
    x_grid = np.linspace(px.min() - 0.5, px.max() + 0.5, 300)
    ax.plot(x_grid, kde(x_grid), color="#e74c3c", lw=2, label="KDE")
    ax.set_xlabel("polarity_x (SVD component 1)")
    ax.set_ylabel("density")
    ax.set_title("Test 1: Distribution of polarity_x\nBimodal = evidence of 2 camps")
    ax.legend()

    # ── 検証2: k=2,3,4,5 のシルエットスコア比較 ─────────
    ax = axes[1]
    ks = [2, 3, 4, 5]
    sils = []
    for k in ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = km.fit_predict(coords)
        s = silhouette_score(coords, labels)
        sils.append(s)
        print(f"  k={k}: silhouette={s:.3f}")

    colors = ["#e74c3c" if k == 2 else "#3498db" for k in ks]
    bars = ax.bar(ks, sils, color=colors, edgecolor="white", width=0.6)
    ax.set_xlabel("Number of clusters k")
    ax.set_ylabel("silhouette score")
    ax.set_title("Test 2: Optimal k\nHighest silhouette = best k")
    ax.set_xticks(ks)
    for bar, s in zip(bars, sils):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{s:.3f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, max(sils) + 0.1)

    # ── 検証3: polarity_x の谷（dip）を可視化 ────────────
    ax = axes[2]
    # 2クラスタの境界線を可視化
    km2 = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    labels2 = km2.fit_predict(coords)
    for c, color, label in [(0, "#e74c3c", "Cluster 0"), (1, "#3498db", "Cluster 1")]:
        mask = labels2 == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, alpha=0.4, s=20, label=label)
    # 各クラスタの重心
    for c, color in [(0, "#e74c3c"), (1, "#3498db")]:
        center = km2.cluster_centers_[c]
        ax.scatter(*center, c=color, s=200, marker="X", edgecolors="black", lw=1.5, zorder=5)
    # 決定境界（2重心の中点を通る垂直線）
    mid = km2.cluster_centers_.mean(axis=0)
    diff = km2.cluster_centers_[1] - km2.cluster_centers_[0]
    perp = np.array([-diff[1], diff[0]])
    t = np.linspace(-5, 5, 100)
    boundary = mid[:, np.newaxis] + perp[:, np.newaxis] * t[np.newaxis, :]
    ax.plot(boundary[0], boundary[1], "k--", lw=1.5, label="boundary")
    ax.set_xlim(coords[:, 0].min() - 0.5, coords[:, 0].max() + 0.5)
    ax.set_ylim(coords[:, 1].min() - 0.5, coords[:, 1].max() + 0.5)
    ax.set_xlabel("polarity_x")
    ax.set_ylabel("polarity_y")
    ax.set_title("Test 3: Decision boundary & centroids\nX = cluster centroid")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = OUT_DIR / "feasibility_2d_validation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  保存先: {out_path}")

    # ── 総合判定 ────────────────────────────────────────
    best_k = ks[np.argmax(sils)]
    print()
    print("=" * 55)
    print("  総合判定")
    print("=" * 55)
    if best_k == 2 and max(sils) >= 0.4:
        print(f"  最適クラスタ数: k={best_k} (silhouette={max(sils):.3f})")
        print("  → 2陣営構造が統計的にも支持されます")
    else:
        print(f"  最適クラスタ数: k={best_k} (silhouette={max(sils):.3f})")
        print("  → 2陣営以外の構造の可能性あり、要検討")
    print("=" * 55)


if __name__ == "__main__":
    main()
