# src/

## 発表用 (これだけ理解する)

- **`simple/`** ← `scripts/run_simple.py` から呼ばれるパイプライン本体
  - `load.py`, `topic.py` — サンプリング & 政治トピック抽出
  - `polarity.py` — TruncatedSVD 2次元、最初 50 件で固定
  - `burst.py` — 速度3倍 & 5件閾値、polarity 分散で TypeA/B 判定
  - `quality.py` — LLM 学習済み固定重み (URL数, 文字数, ドメイン信頼性)
  - `regression.py` — 4変数ロジット (trend は bad control として除外)

## 付随研究 (発表対象外)

- `simple_h1/` — `simple/` の H1 派生。`burst.py`/`regression.py` だけ上書きし、他は `from src.simple import ...` で再利用。TypeA/B × {helpful, nothelp} の 4 ダミー回帰で「TypeA の負係数は HELPFUL 一斉が主因では?」を検証。
- `step1_preprocess/` 〜 `step5_target/`, `step4_regression(_v2)/` — フルデータ用パイプライン (チャンク対応)。注意: trend (bad control) と unresolvable burst の強制 B 扱いの問題が残っている。`simple/` の方が手法的に新しい。
- `io/` — フルデータ用の共通ローダー (チャンク対応)。
