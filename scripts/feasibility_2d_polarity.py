"""
フィージビリティチェック:
ratings.tsv から評価者500人をサンプリングし、
評価行列をSVDで2次元に射影して政治的陣営が分離するか可視化する。

使い方:
  python scripts/feasibility_2d_polarity.py                  # 全データ
  python scripts/feasibility_2d_polarity.py --nrows 500000   # 先頭50万行のみ（高速）

出力:
  1. data/processed/feasibility_2d_polarity.png
  2. シルエットスコア（0.5以上 → 明確な2陣営分離）
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # GUI不要
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── 設定 ──────────────────────────────────────────────
N_RATERS = 500          # サンプリングする評価者数
MIN_RATINGS = 10        # 最低評価件数（少なすぎる人を除外）
N_COMPONENTS = 2        # SVD次元
RANDOM_SEED = 42
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def find_ratings_file() -> Path:
    """data/raw/ 内の ratings ファイルを自動検出"""
    candidates = [
        RAW_DIR / "ratings.tsv",
        RAW_DIR / "ratings-00000.tsv",
    ]
    # glob でも探す
    for p in sorted(RAW_DIR.glob("ratings*.tsv")):
        if p not in candidates:
            candidates.append(p)

    for p in candidates:
        if p.exists():
            return p

    print("エラー: ratings ファイルが見つかりません。")
    print()
    print("手順:")
    print("  1. https://communitynotes.x.com/guide/en/under-the-hood/download-data を開く")
    print("  2. 「ratings」の Download リンクをクリック")
    print(f"  3. ダウンロードしたファイルを {RAW_DIR}/ に配置")
    print(f"     例: mv ~/Downloads/ratings-00000.tsv {RAW_DIR}/")
    sys.exit(1)


# ── 1. データ読み込み ──────────────────────────────────
def load_ratings(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """ratings ファイルを必要カラムのみ読み込む"""
    print(f"Loading {path.name} ...")
    df = pd.read_csv(
        path,
        sep="\t",
        usecols=["noteId", "raterParticipantId", "helpfulnessLevel"],
        dtype={"noteId": str, "raterParticipantId": str},
        nrows=nrows,
    )
    print(f"  読み込みレコード数: {len(df):,}")
    return df


# ── 2. 前処理 ──────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """helpfulnessLevel を数値に変換し、評価の少ない人を除外"""
    level_map = {
        "HELPFUL": 1.0,
        "SOMEWHAT_HELPFUL": 0.0,
        "NOT_HELPFUL": -1.0,
    }
    df = df.copy()
    df["score"] = df["helpfulnessLevel"].map(level_map)
    df = df.dropna(subset=["score"])

    # 最低評価件数フィルタ
    counts = df.groupby("raterParticipantId").size()
    active_raters = counts[counts >= MIN_RATINGS].index
    df = df[df["raterParticipantId"].isin(active_raters)]
    print(f"  {MIN_RATINGS}件以上評価した評価者数: {len(active_raters):,}")
    return df


# ── 3. サンプリング & 評価行列構築 ──────────────────────
def build_matrix(df: pd.DataFrame, n_raters: int, seed: int):
    """評価者をサンプリングし、疎行列を構築"""
    rng = np.random.RandomState(seed)

    unique_raters = df["raterParticipantId"].unique()
    if len(unique_raters) < n_raters:
        print(f"  警告: 評価者が{len(unique_raters)}人しかいません（要求: {n_raters}）")
        n_raters = len(unique_raters)

    sampled = rng.choice(unique_raters, size=n_raters, replace=False)
    df_sub = df[df["raterParticipantId"].isin(set(sampled))]

    rater_ids = sorted(df_sub["raterParticipantId"].unique())
    note_ids = sorted(df_sub["noteId"].unique())
    rater_idx = {r: i for i, r in enumerate(rater_ids)}
    note_idx = {n: i for i, n in enumerate(note_ids)}

    rows = df_sub["raterParticipantId"].map(rater_idx).values
    cols = df_sub["noteId"].map(note_idx).values
    vals = df_sub["score"].values

    mat = csr_matrix((vals, (rows, cols)), shape=(len(rater_ids), len(note_ids)))
    print(f"  評価行列: {mat.shape[0]} raters × {mat.shape[1]} notes")
    print(f"  充填率: {mat.nnz / (mat.shape[0] * mat.shape[1]) * 100:.4f}%")
    return mat, rater_ids


# ── 4. SVD → 2D ────────────────────────────────────────
def svd_2d(mat: csr_matrix) -> np.ndarray:
    """疎行列にSVDを適用し、評価者の2D座標を返す"""
    col_mean = np.array(mat.mean(axis=0)).flatten()
    mat_dense = mat.toarray() - col_mean[np.newaxis, :]
    mat_centered = csr_matrix(mat_dense)

    u, s, _ = svds(mat_centered, k=N_COMPONENTS)
    order = np.argsort(-s)
    coords = u[:, order] * s[order][np.newaxis, :]
    print(f"  特異値: {s[order]}")
    return coords


# ── 5. クラスタリング & 可視化 ──────────────────────────
def plot_and_evaluate(coords: np.ndarray):
    """KMeans(k=2)で分けてプロット + シルエットスコア"""
    km = KMeans(n_clusters=2, random_state=RANDOM_SEED, n_init=10)
    labels = km.fit_predict(coords)

    sil = silhouette_score(coords, labels)
    print()
    print("=" * 50)
    print(f"  シルエットスコア: {sil:.3f}")
    if sil >= 0.5:
        print("  → 2陣営に明確に分離しています")
    elif sil >= 0.25:
        print("  → ある程度の分離が見られます（中程度）")
    else:
        print("  → 分離が弱い、または2クラスタ構造ではない可能性")
    for c in [0, 1]:
        print(f"  クラスタ{c}: {(labels == c).sum()}人")
    print("=" * 50)

    # ── プロット ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: KMeansラベルで色分け
    ax = axes[0]
    for c, color, marker in [(0, "#e74c3c", "o"), (1, "#3498db", "s")]:
        mask = labels == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, marker=marker, alpha=0.5, s=20,
                   label=f"Cluster {c} (n={mask.sum()})")
    ax.set_xlabel("SVD component 1 (polarity_x)")
    ax.set_ylabel("SVD component 2 (polarity_y)")
    ax.set_title(f"KMeans(k=2)  silhouette={sil:.3f}")
    ax.legend()
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    # 右: 色なし（先入観なしで構造を確認）
    ax = axes[1]
    ax.scatter(coords[:, 0], coords[:, 1], c="gray", alpha=0.3, s=15)
    ax.set_xlabel("SVD component 1 (polarity_x)")
    ax.set_ylabel("SVD component 2 (polarity_y)")
    ax.set_title("Raw 2D projection (no clustering)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    plt.tight_layout()

    out_path = OUT_DIR / "feasibility_2d_polarity.png"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  プロット保存先: {out_path}")


# ── main ───────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Community Notes 評価者の政治的2D分離フィージビリティチェック"
    )
    parser.add_argument(
        "--nrows", type=int, default=None,
        help="ratings ファイルの先頭N行のみ読み込む（高速テスト用）",
    )
    parser.add_argument(
        "--raters", type=int, default=N_RATERS,
        help=f"サンプリングする評価者数 (default: {N_RATERS})",
    )
    args = parser.parse_args()

    ratings_path = find_ratings_file()
    df = load_ratings(ratings_path, nrows=args.nrows)
    df = preprocess(df)
    mat, rater_ids = build_matrix(df, args.raters, RANDOM_SEED)
    coords = svd_2d(mat)
    plot_and_evaluate(coords)


if __name__ == "__main__":
    main()
