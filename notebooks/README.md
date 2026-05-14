# notebooks/

## 発表用 (これだけ実行)

- **`colab_simple.ipynb`** ← Google Colab で開いて上から順に実行
  - ★ 設定セルで `SAMPLE_FRAC` と `SEED` を指定
  - 出力: `data/processed/simple_regression.txt`, `simple_features.csv`, `simple_bursts.csv`
  - 頑健性チェック: `SEED = 1, 2, 3` で 3 回回す

## experiments/ (発表対象外)

- `colab_simple_h1.ipynb` — H1 派生 (TypeA × direction)
- `colab_full_run_v2.ipynb` — フルデータパイプライン (M0/M1/M2 比較)
