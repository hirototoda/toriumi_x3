---
title: Community Notes における「陣営反応バースト」仮説の検証
subtitle: colab_simple.ipynb パイプラインの説明
---

# スライド構成案（目安 12〜14 枚 / 発表 10〜12 分）

各スライドは `---` で区切り。囲み数式は LaTeX (`$...$`, `$$...$$`) で書く。

---

## スライド 1 — タイトル

**Community Notes は「陣営反応」で潰されているのか？**
— X (旧 Twitter) 公開データによるロジスティック回帰分析 —

- 発表者 / 所属 / 日付
- 一言サブタイトル:「バーストの質」を測って、Not Helpful 化との関係を見る

---

## スライド 2 — 背景と問い

- **Community Notes**: X のクラウドソース型ファクトチェック。ユーザーが note を書き、他ユーザーが Helpful / Not Helpful を評価する。
- 一定の合意が取れた note だけが表示される。取れなければ `CURRENTLY_RATED_NOT_HELPFUL` となり非表示に。
- 懸念:
  > 政治的に不都合な note に対して、反対陣営が短時間で集中的に Not Helpful を押し、合意形成を妨げているのでは?
- **本発表の問い**: この「陣営による集中投票＝陣営反応バースト」が、note の Not Helpful 化を実際に予測するのか。

---

## スライド 3 — 仮説と判定基準

バーストを 2 種類に分けて比較する。

| 種別 | 定義（直感） | 役割 |
|---|---|---|
| **TypeA** | 「同じ陣営」が集中評価したバースト | 陣営反応の指標 |
| **TypeB** | 「いろんな陣営」が集中評価したバースト | 自然拡散の指標（対照） |

**判定**

| 結果 | 解釈 |
|---|---|
| TypeA のみ有意 ($p<0.05$) | **仮説支持** — 陣営反応が主因 |
| TypeA / TypeB 両方有意 | 仮説部分支持 — 自然拡散も効く |
| TypeB のみ有意 | 仮説不支持 |
| どちらも非有意 | 仮説不支持 / サンプル不足 |

---

## スライド 4 — 使用データ

X が公開している 3 種の TSV（日次更新）:

- `notes-*.tsv` — note 本体（`noteId`, 本文 `summary`, classification など）
- `ratings-*.tsv` — 評価ログ（`raterParticipantId`, `noteId`, `helpfulnessLevel`, `createdAtMillis`）
- `noteStatusHistory-*.tsv` — 各 note の最終ステータス（`CURRENTLY_RATED_HELPFUL` 等）

**Simple 版** では `noteId` をランダム 30% サンプリング → 該当 rating/history のみ読み込み（計算時間 10〜15 分目安）。

---

## スライド 5 — パイプライン全体像

`scripts/run_simple.py` の 6 ステップ:

1. notes 全件読込 → **政治トピック**で絞り込み → `noteId` を frac サンプリング
2. サンプリングされた noteId に紐づく ratings / history を読み込み
3. **Polarity 計算** (TruncatedSVD, 次元 2)
4. **バースト検出** + **TypeA/B 分類**
5. **Quality スコア**（LLM ラベル学習済の固定重み）
6. **ロジスティック回帰** (相関行列 / VIF 診断つき)

> ここから各ステップの数式を説明する。

---

## スライド 6 — ① 政治トピック抽出 / ② サンプリング

- `classification` 列と本文 `summary` を **単語境界付きの部分一致**（政治キーワード辞書）でフィルタ。
- サンプリング:
  $$ \text{sample\_ids} = \text{RandomSample}\big(\{\text{noteId}\}_\text{political},\ \text{frac},\ \text{seed}\big) $$
  - `frac = 0.30`, `seed = 42`（再現性のため固定）
  - 頑健性チェック: seed を $1,2,3$ と変え $\beta_{\text{typeA}}$ がブレないか確認。

---

## スライド 7 — ③ Polarity（評価者の立場ベクトル）

**目的**: 各 rater を 2 次元ベクトル $(x, y)$ で表し、「似た note を同じように評価する人」を近づける。

**手順**

1. 評価を数値化:
   $$ s = \begin{cases} +1 & \text{HELPFUL} \\ 0 & \text{SOMEWHAT\_HELPFUL} \\ -1 & \text{NOT\_HELPFUL} \end{cases} $$
2. **各 rater の最初の $N=50$ 件だけ**採用（後述の循環論法回避のため）。
3. 疎行列 $M \in \mathbb{R}^{R \times N_\text{note}}$ を作る:
   $$ M_{ij} = s_{ij} \quad (\text{rater } i\ \text{が note } j\ \text{に付けた score}) $$
4. **Truncated SVD** で rank-2 近似:
   $$ M \approx U_2\, \Sigma_2\, V_2^{\!\top},\qquad U_2 \in \mathbb{R}^{R \times 2} $$
   rater $i$ の polarity $= U_2[i,:] \cdot \Sigma_2$（sklearn の `fit_transform` の返り値）。
5. $<5$ 件しか評価していない rater は除外（雑音防止）。

> 2 次元のうち第 1 成分が「左右」、第 2 成分がそれと直交する変動を捉える。

---

## スライド 8 — なぜ「最初の 50 件だけ」なのか（循環論法回避）

もし全期間の評価を使うと:

- 「note が Not Helpful になった後」の評価も polarity に混ざる
- ⇒ polarity はすでに **目的変数 (deleted) の情報を含んだ量**になる
- ⇒ その polarity で TypeA/B を決めて回帰すると、**目的変数で説明変数を作っている** (target leakage)

**対策**: rater 単位で時系列ソート → 先頭 50 件だけで polarity を固定。
→ polarity は「その人が普段どっち陣営か」を表す **時間的に先行する量** になる。

---

## スライド 9 — ④ バースト検出

**全体の平均評価速度**:
$$ v_\text{avg} = \frac{n_\text{total}}{t_\text{last} - t_\text{first}} \qquad \text{（1 note 内）} $$

**局所速度** (連続する $k = \text{MIN\_COUNT} = 5$ 件のウィンドウ):
$$ v_\text{local}(i) = \frac{k}{\max\!\big(t_{i+k-1} - t_i,\ 1\text{ ms}\big)} $$

**バースト条件**:
$$ v_\text{local}(i) \ \ge\ \alpha \cdot v_\text{avg},\qquad \alpha = \text{BURST\_MULTIPLIER} = 3.0 $$

条件を満たす $i$ が見つかったら、その区間をバーストとする。簡略化のため **1 note につき最初の 1 バーストのみ**採用。

> 直感: 「その note の普段ペースの 3 倍以上速く、5 件連続で評価が来た区間」。

---

## スライド 10 — ⑤ TypeA / TypeB 分類

バースト中の $k$ 人の rater について polarity の**分散の和**を計算:
$$ V = \mathrm{Var}(x_1,\dots,x_k) + \mathrm{Var}(y_1,\dots,y_k) $$

**相対基準で分類**（全バーストの中央値 $\tilde V$ を閾値）:
$$ \text{burst\_type} = \begin{cases} A & \text{if } V \le \tilde V \quad(\text{同じ陣営} = \text{陣営反応}) \\ B & \text{if } V > \tilde V \quad(\text{バラバラ} = \text{自然拡散}) \end{cases} $$

- polarity が分からない rater だけの場合は `NaN` にして回帰で dropna（「強制 B」はバイアスを生むため禁止）。
- 閾値を中央値に取ることで、「この研究全体で相対的にどうか」を見る設計。

---

## スライド 11 — ⑥ Quality スコア（LLM 学習済重み）

**事前学習**: Claude が 200 件の note に $\{0,1\}$ ラベル付 → ロジスティック回帰で係数学習（CV AUC = 0.894）。

**本パイプラインでは固定重み**を使う:

特徴量 $f_k \in \{\text{url\_count}, \text{char\_count}, \text{domain\_trust}\}$ を標準化してから線形和:

$$ z = \beta_0 + \sum_{k} \beta_k \cdot \frac{f_k - \mu_k}{\sigma_k} $$

$$ \text{quality} = \sigma(z) = \frac{1}{1+e^{-z}} \in [0,1] $$

| 変数 | 係数 $\beta_k$ | 寄与度（\|β\| の比） |
|---|---|---|
| `domain_trust` | 1.285 | **44%** |
| `char_count`   | 0.962 | 33% |
| `url_count`    | 0.654 | 23% |
| (intercept)    | 0.519 | — |

> 「信頼できるドメインの URL があるか」が最重要、という LLM ラベルの事前知識と整合。

---

## スライド 12 — ⑥ ロジスティック回帰（メイン）

**目的変数**: $\text{deleted} = \mathbb{1}[\text{status}=\text{CURRENTLY\_RATED\_NOT\_HELPFUL}]$

**回帰式 (log-odds)**:

$$
\log\!\frac{\Pr(\text{deleted}=1)}{\Pr(\text{deleted}=0)}
= \beta_0
+ \beta_1\,\text{type\_a}
+ \beta_2\,\text{type\_b}
+ \beta_3\,\text{quality}
+ \beta_4\,\log(1+\text{ratings\_count})
$$

ここで

- `type_a`, `type_b` $\in \{0,1\}$ (両方 1 になることはない: 1 note 1 バースト)
- `quality` $\in [0,1]$
- $\log(1+\text{ratings\_count})$ で**人気度を control**（バーストは構造的に評価数と正相関するため必須）。

**注意**: ナイーブな control 変数 `trend` (前半スコア−後半スコア) は目的変数と同じ生データから作られる **bad control** なので削除した。

**診断**: 相関行列と VIF を自動 print し、多重共線性がないか確認。

---

## スライド 13 — 実装と再現性

- リポジトリ: `hirototoda/toriumi_x3` / notebook は `notebooks/colab_simple.ipynb`
- 設定セル 1 箇所編集するだけで実験可能:
  - `SAMPLE_FRAC` — サンプリング率
  - `SEED` — 乱数シード（頑健性チェック用）
  - `POLARITY_FIRST_N` — polarity 固定に使う先頭件数
  - `BURST_MULTIPLIER`, `BURST_MIN_COUNT` — バースト閾値
- **頑健性チェック**: 同じ `SAMPLE_FRAC` で `SEED = 1,2,3` を回し、
  - $\beta_{\text{typeA}}$ の**符号が一貫して正**、かつ
  - **2 回以上** $p < 0.05$

  を満たせば「結果は頑健」と判断する。

---

## スライド 14 — 結果と議論（発表直前に差し替え）

- N = （サンプル後の note 数） / deleted = ... / helpful = ... / TypeA = ... / TypeB = ...
- 回帰結果（`data/processed/simple_regression.txt` から転記）:
  - $\beta_{\text{type\_a}} = \ldots\ (p = \ldots)$
  - $\beta_{\text{type\_b}} = \ldots\ (p = \ldots)$
  - $\beta_{\text{quality}} = \ldots\ (p = \ldots)$
  - $\beta_{\text{log\_ratings}} = \ldots\ (p = \ldots)$
- 4 ケースのどれに該当するかで仮説判定。
- **限界**: 政治トピック判定がキーワードベース / polarity 次元 2 / note 単位集計など。

---

## 付録 — 発表の流れ（台本用メモ）

| 時間 | スライド | 話すこと |
|---|---|---|
| 0:00 | 1 | タイトル・自己紹介 |
| 0:30 | 2–3 | 問いと仮説判定表 |
| 2:00 | 4–5 | データとパイプライン全体像 |
| 3:30 | 6 | トピック抽出・サンプリング |
| 4:30 | 7–8 | Polarity の数式と「なぜ 50 件で固定か」 |
| 6:30 | 9–10 | バースト検出と TypeA/B の閾値 |
| 8:00 | 11 | Quality スコアの式と寄与度 |
| 9:00 | 12 | 回帰式（メイン）と bad control の議論 |
| 10:30 | 13–14 | 再現性と結果、質疑 |

---

## 付録 — 想定質問

- 「TruncatedSVD で次元 2 にするのは根拠ある?」
  → 第 1 成分で `explained_variance_ratio` を出力している。もし 1 軸で十分ならそれが分かる。
- 「バースト閾値 3.0 と 5 件はどう決めた?」
  → 頑健性チェック対象のハイパラ。notebook 設定セルで変更可。
- 「quality の LLM ラベル 200 件で足りる?」
  → CV AUC 0.894。寄与度は事前知識と整合。再学習すれば `scripts/experiments/train_quality_model.py` で更新可。
- 「trend を控除したほうが良くない?」
  → 目的変数と同一の生データから作られる bad control で、$\beta_{\text{typeA}}$ を不当に押し下げる。
