# コミュニティノート「陣営反応バースト」仮説検証

## 仮説

X コミュニティノートのステータス変化 (Helpful / Not Helpful / 削除) は、ノートの品質ではなく **同一政治クラスタによる集中的な評価 (陣営反応)** によって引き起こされているのではないか。

---

## 発表で使うのはこの 1 本だけ

```
notebooks/colab_simple.ipynb     ← Colab でこれを実行
└── scripts/run_simple.py        ← セルから subprocess で呼ばれる本体
    └── src/simple/              ← パイプラインの実装 (理解すべきはここ)
```

実行手順:

1. Google Colab で [notebooks/colab_simple.ipynb](notebooks/colab_simple.ipynb) を開く
2. ★ 設定セルで `SAMPLE_FRAC` と `SEED` を指定
3. 上から順にセルを実行 → `data/processed/simple_regression.txt` 等が出力される
4. **頑健性チェック**: `SEED = 1, 2, 3` で 3 回回し、β_typeA の符号が一致 + 2回以上 p<0.05 なら「結果は頑健」と判断

スライド原稿は [docs/slides_simple.md](docs/slides_simple.md)。

---

## フォルダ構造

トップを見れば「発表用は 1 本だけ」と分かる構成にしてある。`experiments/` 直下は発表対象外。

```
.
├── README.md                       ← 本ファイル
├── TODO.md
├── requirements.txt
│
├── data/
│   ├── raw/                        公式TSV (git管理外)
│   ├── interim/                    中間生成物
│   └── processed/                  simple_regression.txt 等の最終出力
│
├── docs/                           スライド原稿・データソースメモ
│
├── notebooks/
│   ├── README.md
│   ├── colab_simple.ipynb          ★ 発表用 (これだけ実行)
│   └── experiments/                付随研究 (H1派生, フル処理版)
│       ├── colab_simple_h1.ipynb
│       └── colab_full_run_v2.ipynb
│
├── scripts/
│   ├── README.md
│   ├── download_data.sh
│   ├── run_simple.py               ★ colab_simple から呼ばれる本体
│   └── experiments/                付随研究の実行スクリプト 7本
│
├── src/
│   ├── README.md
│   ├── io/                         共通ローダー
│   ├── simple/                     ★ 発表用パイプライン (理解すべきはここ)
│   ├── simple_h1/                  H1 派生 (TypeA × direction)
│   └── step1_preprocess/ 〜 step5_target/,
│       step4_regression(_v2)/      フルデータ版
│
├── tests/
└── archive/                        明らかに不要になったものの退避先 (git 履歴は追える)
```

### 各サブディレクトリの位置づけ

- **`notebooks/colab_simple.ipynb`** — 発表用 Colab。設定セル 1 箇所を編集すれば全部回る
- **`scripts/run_simple.py`** — 上のノートブックから呼ばれる本体。ローカルでも `python scripts/run_simple.py --sample-frac 0.02` で動作確認可
- **`src/simple/`** — スライドと 1 対 1 対応の実装本体 (下表)
- **`experiments/` 配下** — H1 派生・フルデータ版など、発表に含めない付随研究
- **`archive/`** — 古い試行・stub の退避先。`git log --follow archive/<path>` で履歴を追える

---

## 発表用パイプラインの中身

[src/simple/](src/simple/) の各ファイルがスライドと 1 対 1 対応:

| スライド | ファイル | 内容 |
|---|---|---|
| 6 (トピック/サンプリング) | [src/simple/load.py](src/simple/load.py), [src/simple/topic.py](src/simple/topic.py) | noteId サンプリング、政治キーワード抽出 |
| 7-8 (Polarity) | [src/simple/polarity.py](src/simple/polarity.py) | TruncatedSVD 2次元 + 最初 50 件で固定 (循環論法回避) |
| 9-10 (Burst, TypeA/B) | [src/simple/burst.py](src/simple/burst.py) | 速度 3 倍 & 5 件、polarity 分散の中央値で A/B 判定 |
| 11 (Quality) | [src/simple/quality.py](src/simple/quality.py) | LLM 学習済み固定重み (URL数, 文字数, ドメイン信頼性) |
| 12 (回帰) | [src/simple/regression.py](src/simple/regression.py) | 4変数ロジット、trend を bad control として除外 |

### 結論の判定基準

回帰係数の符号と有意性で判定:

- **TypeA のみ有意** → 陣営反応が主因 (仮説支持)
- **TypeA・B 両方有意** → 自然拡散が主因の可能性
- **どちらも有意でない** → 別要因を探す

---

## データ入手

X 公式の Community Notes パブリックデータ (独自スクレイピングなし):

1. https://communitynotes.x.com/guide/en/under-the-hood/download-data
2. `notes.tsv`, `ratings.tsv`, `noteStatusHistory.tsv` を `data/raw/` に配置 (Colab 実行時は Drive にアップロード)
3. ダウンロード日を [docs/data_source.md](docs/data_source.md) に記録

データは数百MB〜GB級のため git 管理外 (`.gitignore` 済み)。

---

## セットアップ (ローカル開発用)

```bash
pip install -r requirements.txt

# 動作確認用のミニ実行 (本番データは Colab で)
python scripts/run_simple.py --sample-frac 0.02 --seed 1
```

ローカルマシンは全データを扱えないので、本番実行は必ず Colab で。

---

## 付随研究 (発表対象外)

詳細は各サブディレクトリの README ([notebooks/README.md](notebooks/README.md), [scripts/README.md](scripts/README.md), [src/README.md](src/README.md)) を参照。

- [src/simple_h1/](src/simple_h1/) — H1 派生。バースト方向 (helpful/nothelp) で TypeA を分割した 4 ダミー回帰
- `src/step1_preprocess/` 〜 `step5_target/`, `step4_regression(_v2)/` — フルデータ処理パイプライン。注: trend (bad control) と unresolvable burst の強制 B 扱いの問題が残っており、手法的には `src/simple/` の方が新しい
- [scripts/experiments/](scripts/experiments/) — 上記用の実行スクリプト 7 本 (run_simple_h1, run_pipeline(_v2), run_by_topic(_v2), merge_chunks, train_quality_model)
- [notebooks/experiments/](notebooks/experiments/) — 上記用の Colab ノートブック 2 本
