# コミュニティノート「陣営反応バースト」仮説検証 — 提出版

CS 授業プロジェクトの最終提出パッケージ（ソースコード一式）。

## 仮説

X (Twitter) コミュニティノートのステータス変化 (Helpful / Not Helpful / 削除) は、ノートの品質ではなく **同一政治クラスタによる集中的な評価バースト (陣営反応)** によって引き起こされているのではないか。

## 結果サマリ

本番ラン (frac=0.30, N=14,253 notes, SEED=42) のロジスティック回帰結果：

| 変数 | β | p |
|---|---:|---:|
| `type_a` (同陣営バースト) | −1.129 | <0.001 |
| `type_b` (混在バースト)   | −0.404 | <0.001 |
| `quality`                  | −1.865 | <0.001 |
| `log_ratings_count`        | −0.603 | <0.001 |

`type_a` の効果量が `type_b` の約 2.8 倍、quality を統制しても有意。同 SEED=1 のリランでも符号・桁が一致（頑健性チェック OK）。詳細は [data/processed/runs_2026-05-14.md](data/processed/runs_2026-05-14.md)。

判定: **`type_a` と `type_b` の両方が有意だが、効果量は `type_a` が大きく、陣営反応の主導性は支持される。完全な「陣営反応のみが原因」までは言えず仮説部分支持。**

## 再現手順

### Colab で実行（本番）

1. [notebooks/colab_simple.ipynb](notebooks/colab_simple.ipynb) を Google Colab で開く
2. 冒頭の「★ 設定」セルで `SAMPLE_FRAC` (本番=0.30) と `SEED` (42 / 1 / 2 など) を指定
3. 上から順にセルを実行 → `data/processed/simple_regression.txt` 等が出力される
4. 結果が `data/processed/runs_2026-05-14.md` の表と一致することを確認

実行時間目安：`SAMPLE_FRAC=0.30` で 10〜15 分（Colab 標準ランタイム）。

### ローカルで動作確認のみ

フルデータはローカルマシンに載らないので、サンプル比率を小さくして動作確認のみ可能。

```bash
pip install -r requirements.txt
python scripts/run_simple.py --sample-frac 0.02 --seed 1
pytest tests/  # quality モデルの round-trip テスト 7 件
```

## データ入手

X 公式の Community Notes パブリックデータ（独自スクレイピングなし）:

1. https://communitynotes.x.com/guide/en/under-the-hood/download-data
2. `notes-*.tsv`, `ratings-*.tsv`, `noteStatusHistory-*.tsv` を [data/raw/](data/raw/) に配置（Colab なら Google Drive の共有フォルダに置く）

データ仕様の詳細は [docs/data_source.md](docs/data_source.md)。

## ファイル構成

```
.
├── README.md                           ← 本ファイル
├── requirements.txt                    依存パッケージ
│
├── notebooks/
│   └── colab_simple.ipynb              ★ Colab で実行する本番ランナー
│
├── scripts/
│   ├── run_simple.py                   colab_simple から呼ばれるパイプライン本体
│   └── experiments/
│       └── burst_helpfulness.py        バースト内 helpfulness 集計（後続分析）
│
├── src/
│   ├── simple/                         ★ パイプライン実装（スライドと 1:1 対応）
│   │   ├── load.py                     データ読み込み + noteId サンプリング
│   │   ├── topic.py                    政治トピック抽出
│   │   ├── polarity.py                 TruncatedSVD で rater polarity 推定
│   │   ├── burst.py                    バースト検出 + TypeA/B 分類
│   │   ├── quality.py                  ノート品質スコア（LLM 学習済固定重み）
│   │   └── regression.py               ロジット回帰
│   └── step5_target/                   simple/quality.py が import するユーティリティ
│       ├── domain_trust.py
│       ├── quality_features.py
│       ├── quality_model.py
│       └── quality_score.py
│
├── tests/
│   └── test_quality_model.py           quality モデルの round-trip テスト
│
├── data/
│   ├── raw/                            公式 TSV 配置先（.gitignore 済）
│   └── processed/
│       └── runs_2026-05-14.md          ★ 本番ランの結果記録
│
└── docs/
    ├── slides_simple.md                発表スライドのソース
    ├── slides_simple.pdf               発表スライド PDF
    └── data_source.md                  データソース仕様
```

## パイプラインの中身

[src/simple/](src/simple/) の各ファイルがスライドの段階と 1 対 1 対応：

| スライド | ファイル | 内容 |
|---|---|---|
| トピック / サンプリング | [load.py](src/simple/load.py), [topic.py](src/simple/topic.py) | noteId をシードで `frac` 割サンプリング → 政治キーワードを含むノートのみ残す → 該当 ratings をチャンクストリームで読み込み |
| Polarity | [polarity.py](src/simple/polarity.py) | rater×note 疎行列に TruncatedSVD (k=2)。**各 rater の最初の 50 件のみ** で polarity を固定（循環論法回避：削除後の評価が混ざると目的変数を見ることになる） |
| Burst 検出 + TypeA/B 分類 | [burst.py](src/simple/burst.py) | 5 件連続評価が全体平均速度の 3 倍以上速い区間をバーストと定義。polarity 分散の中央値で TypeA（同陣営集中）と TypeB（混在）に分割。polarity 分散不能なバーストは行ごと除外 |
| Quality | [quality.py](src/simple/quality.py) | LLM ラベル 200 件で事前学習したロジット係数（url_count / char_count / domain_trust）をハードコード。CV AUC=0.894 |
| 回帰 | [regression.py](src/simple/regression.py) | `deleted ~ type_a + type_b + quality + log_ratings_count` (二項 GLM)。相関行列と VIF を必ず出力。**`trend` は bad control として意図的に除外**（helpfulnessLevel から作られ目的変数と同じ生データなので入れると β_typeA を不当に押し下げる） |

### 仮説判定ロジック

`regression.run_logit()` 内：

- **TypeA のみ有意** → 陣営反応が主因（仮説支持）
- **TypeA・B 両方有意** → 自然拡散も寄与（仮説部分支持）
- **どちらも非有意** → 別要因 or サンプル不足

## 注意

- パイプラインは [src/simple/](src/simple/) を上から順に読めば意味が通るように設計。コメントは多くの場合「**なぜ他の選択肢を採らなかったか**」（polarity が最初の 50 件だけなのはなぜか、`trend` をなぜ除外するか、unresolvable burst をなぜ除外するか）を残している。
- quality モデルの係数は [src/simple/quality.py](src/simple/quality.py) にハードコード。学習データ自体は提出対象外（係数のみで動作する）。
