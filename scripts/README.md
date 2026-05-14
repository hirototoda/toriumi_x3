# scripts/

## 発表用 (これだけ使う)

- `run_simple.py` — Colab の `notebooks/colab_simple.ipynb` から subprocess で呼ばれる本体。直接 `python scripts/run_simple.py --sample-frac 0.02 --seed 1` で動作確認可。
- `download_data.sh` — X 公式データ取得補助。

## experiments/ (発表対象外)

`experiments/` 直下は発表に使わない付随研究のスクリプト群:

- `run_simple_h1.py` — H1 派生 (TypeA × direction)
- `run_pipeline.py`, `run_pipeline_v2.py` — フルデータ処理 v1/v2
- `run_by_topic.py`, `run_by_topic_v2.py` — トピック別フル処理 v1/v2
- `merge_chunks.py` — チャンク出力の統合
- `train_quality_model.py` — quality モデル再学習 (普段不要、固定重みで運用)
