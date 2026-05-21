---
title: Community Notes における「陣営反応バースト」仮説の検証
subtitle: colab_simple.ipynb パイプラインの説明
---

# 本編 (6 枚 / 発表 ~6 分)

各スライドは `---` で区切り。囲み数式は LaTeX (`$...$`, `$$...$$`) で書く。
発表者は **手法パート**を担当する想定。結果・考察は他メンバーが補足する前提でサマリのみ載せる。
スライド外の詳細 (Polarity の循環論法回避、Quality スコア、頑健性、想定質問) は本ファイル後半の **Appendix** に温存。

---

## スライド 1 — 問いと仮説

**問い**: Community Notes は「陣営反応」で潰されているのか？
— 政治的に不都合な note に対し、反対陣営が短時間で集中的に Not Helpful を押し、合意形成を妨げているのではないか。

**仮説の切り分け**: バーストを 2 種類に分けて比較する。

| 種別 | 定義（直感） | 役割 |
|---|---|---|
| **TypeA** | 「同じ陣営」が集中評価したバースト | 陣営反応の指標 |
| **TypeB** | 「いろんな陣営」が集中評価したバースト | 自然拡散の指標（対照） |

**判定**: TypeA のみ有意 ($p<0.05$, $\beta_\text{typeA}>0$) なら **仮説支持**。両方有意なら $|\beta|$ の大小で主因を議論。

---

## スライド 2 — データとパイプライン全体像

**データ**: X が公開する Community Notes の 3 種 TSV（日次更新）
- `notes-*.tsv` (本文・分類) / `ratings-*.tsv` (評価ログ) / `noteStatusHistory-*.tsv` (最終ステータス)
- 本番 `SAMPLE_FRAC = 0.30` で noteId 単位サンプリング

**パイプライン** (`scripts/run_simple.py` の 6 ステップ)

1. notes 読込 → **政治トピック**で絞り込み → noteId を `frac` サンプリング
2. 該当 ratings / history のみチャンク読み
3. **Polarity 計算** (TruncatedSVD, 次元 2) — 各 rater を 2D ベクトルに
4. **バースト検出** + **TypeA / TypeB 分類**
5. **Quality スコア** (LLM ラベル学習済の固定重み)
6. **ロジスティック回帰** (相関行列 / VIF 診断つき)

> 本編では 3–4 と 6 を中心に説明。1, 2, 5 は appendix 参照。

---

## スライド 3 — 手法 ① Polarity (評価者の立場ベクトル)

**目的**: 各 rater を 2D ベクトル $(x, y)$ で表し「似た note を同じように評価する人」を近づける。

**手順**

1. 評価を数値化:
   $$ s = \begin{cases} +1 & \text{HELPFUL} \\ 0 & \text{SOMEWHAT\_HELPFUL} \\ -1 & \text{NOT\_HELPFUL} \end{cases} $$
2. **各 rater の最初の $N=50$ 件だけ**採用（後述の循環論法回避のため）。
3. 疎行列 $M \in \mathbb{R}^{R \times N_\text{note}}$ に対し **Truncated SVD** で rank-2 近似:
   $$ M \approx U_2\, \Sigma_2\, V_2^{\!\top},\quad \text{polarity}_i = U_2[i,:]\cdot \Sigma_2 $$

**「最初の 50 件」の理由 (循環論法回避)**: 全期間の評価を使うと「note が削除された後」の評価も polarity に混ざり、polarity が目的変数 `deleted` の情報を含んでしまう (target leakage)。先頭 50 件で固定すれば polarity は **時間的に先行する** 量になる。

---

## スライド 4 — 手法 ② バースト検出と TypeA/B 分類

**バースト条件** (1 ノート内・連続 $k=5$ 件のウィンドウ):
$$ v_\text{local}(i) = \frac{k}{\max(t_{i+k-1}-t_i,\ 1\text{ms})}\ \ge\ 3.0 \cdot v_\text{avg} $$
直感: 「**その note の普段ペースの 3 倍以上速く、5 件連続で評価が来た区間**」。1 note につき先頭 1 バーストのみ採用。

**TypeA / TypeB 分類** — バースト中 $k$ 人の polarity 分散の和を計算:
$$ V = \mathrm{Var}(x_1,\dots,x_k) + \mathrm{Var}(y_1,\dots,y_k) $$
$$ \text{burst\_type} = \begin{cases} A & V \le \tilde V \quad(\text{同じ陣営 = 陣営反応}) \\ B & V > \tilde V \quad(\text{バラバラ = 自然拡散}) \end{cases} $$

$\tilde V$ は全バーストの $V$ の **中央値**。相対基準にすることでサンプル依存を避ける。

---

## スライド 5 — 手法 ③ ロジスティック回帰

**目的変数**: $\text{deleted} = \mathbb{1}[\text{status}=\text{CURRENTLY\_RATED\_NOT\_HELPFUL}]$

**回帰式** (log-odds):

$$
\log\!\frac{\Pr(\text{deleted}=1)}{\Pr(\text{deleted}=0)}
= \beta_0
+ \beta_1\,\text{type\_a}
+ \beta_2\,\text{type\_b}
+ \beta_3\,\text{quality}
+ \beta_4\,\log(1+\text{ratings\_count})
$$

**設計上のポイント**

- `type_a`, `type_b` $\in \{0,1\}$ (1 note 1 バーストなので両方 1 にはならない)
- $\log(1+\text{ratings\_count})$ で**人気度を control** — バーストは構造的に評価数と正相関するため必須
- ナイーブな `trend` (前半−後半スコア) は目的変数と同じ生データ由来の **bad control** なので除外
- 相関行列と VIF を自動 print し多重共線性を確認

---

## スライド 6 — 結果と判定

**実行条件**: `SAMPLE_FRAC = 0.30`, `SEED = 42` / N = 14,253 (政治 notes), deleted 28.4%

**ロジスティック回帰** (`deleted ~ type_a + type_b + quality + log_ratings_count`)

| 変数 | $\beta$ | $p$ |
|---|---:|---:|
| `type_a`            | $-1.129$ | $<0.001$ |
| `type_b`            | $-0.404$ | $<0.001$ |
| `quality`           | $-1.865$ | $<0.001$ |
| `log_ratings_count` | $-0.603$ | $<0.001$ |
| (const)             | $+3.727$ | $<0.001$ |

**判定: 仮説は棄却** (事前基準 = スライド 1 の「$\beta_\text{typeA}>0$ かつ $p<0.05$」に符号で反するため)
- $\beta_\text{typeA} = -1.13$ と強く負 → バーストを持つ note はむしろ **削除されにくい**
- $|\beta_\text{typeA}| > |\beta_\text{typeB}|$ で TypeA の "保護効果" のほうが強い

**解釈**: TypeA は polarity 分散のみで**投票方向 (Helpful / Not Helpful) を区別していない**ため、同陣営集中には「攻撃」と「防衛」の両方が混ざりうる。よって「TypeA バーストが**保護を引き起こした** (支持陣営の防衛 rally)」と因果的に読むよりは、「**サポートされやすい note に TypeA が出やすい**」(合意の取れた note に同陣営の piling-on が後追いで起きる、selection 寄り) と捉える方が現データに対して安全。causal な攻撃 vs 防衛の特定には投票方向の観測が必要で、設計拡張 (pro-rally / con-rally 分割) が今後の課題。頑健性は SEED=1 でも同符号・同桁・$p<0.001$ で再現済 (appendix)。

---

# Appendix (補助スライド・必要に応じて表示)

以下は本編から外した詳細・補助資料。手法の細部、Quality スコア設計、頑健性チェック、想定質問など。

---

## A1 — 政治トピック抽出 / サンプリング

- ノート本文 `summary` を **単語境界 `\b` 付き正規表現**でフィルタ（政治キーワード辞書、約 40 語: `trump`, `election`, `abortion`, `immigration`, `supreme court`, ... ）。
  - 旧実装の単純な部分一致では `vote` → `devote` / `voter` 等が誤マッチしていたため `\b...\b` で囲った。
- サンプリング（noteId 単位、rater 単位ではない）:
  $$ \text{sample\_ids} = \text{RandomSample}\big(\{\text{noteId}\}_\text{political},\ \text{frac},\ \text{seed}\big) $$
  - デフォルト `frac = 0.30`, `seed = 42`（再現性のため固定）
  - サンプリング済 noteId に紐づく ratings のみチャンク読み → メモリ節約
  - 頑健性チェック: seed を $1,2,3$ と変え $\beta_{\text{typeA}}$ がブレないか確認。

---

## A2 — Polarity 補足 (本編スライド 3 の補強)

> 手順・SVD 式は本編スライド 3 を参照。ここでは本編に載せきれなかった設計判断のみ。

- **rater 除外基準**: $<5$ 件しか評価していない rater は除外（polarity が雑音になるため）。`fit_transform` で計算された値を rater $i$ の polarity として採用。
- **成分の解釈**: 2 次元のうち第 1 成分が「左右」、第 2 成分がそれと直交する変動を捉える想定。

**次元 2 の妥当性 (本番実測, frac=0.30, SEED=42, N_rater=401,045)**
`explained_variance_ratio_` は **PC1 = 0.2%, PC2 = 0.1%**。比率が小さいのは rater×note 行列が極端に疎 (ほぼ全エントリ 0) で**疎データ全体の分散**に対する 2 次元近似の比が本質的に小さくなるため。重要なのは「主要な変動方向を 2 軸が拾えているか」で、後段の $\beta_\text{typeA}$ が SE $\approx 0.10$ / $p<0.001$ で SEED 間も同符号・同桁に再現することが間接的 validation (polarity が雑音なら係数は不安定になるはず)。分散 $= \mathrm{Var}(x)+\mathrm{Var}(y)$ なので軸選択にも robust。

---

## A3 — 循環論法回避の論理チェーン (本編スライド 3 の補強)

> 本編スライド 3 では結論のみ。質問されたらこのチェーンで説明する。

全期間の評価を使うと:

- 「note が Not Helpful になった後」の評価も polarity に混ざる
- ⇒ polarity はすでに **目的変数 (`deleted`) の情報を含んだ量**になる
- ⇒ その polarity で TypeA/B を決めて回帰すると、**目的変数で説明変数を作っている** (target leakage)

**対策**: rater 単位で時系列ソート → 先頭 50 件で polarity を固定 → polarity は **時間的に先行する量** になる。

---

## A4 — TypeA/B 分類の設計詳細

**polarity が決まる rater が 2 人未満のバーストの扱い** (= 分類不能):

- バースト行ごと除外 → そのノートは「バーストなし」として回帰に入る (`type_a = type_b = 0`)。
- 旧実装は強制 B にしていたがバイアスを生むため不採用。
- NaN を残して回帰側で dropna する案も試したが NA 比較で TypeError になるため不採用。

**設計のポイント**

- 閾値を中央値に取ることで「この研究全体で相対的にどうか」を見る設計（絶対閾値はサンプル依存になる）。
- 1 ノート 1 バーストなので `type_a` と `type_b` が同時に 1 になることはない。

---

## A5 — Quality スコア（LLM 学習済重み）

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

## A6 — 実装と再現性

- リポジトリ: `hirototoda/toriumi_x3` / notebook は `notebooks/colab_simple.ipynb`
- 設定セル 1 箇所編集するだけで実験可能:
  - `SAMPLE_FRAC` — サンプリング率
  - `SEED` — 乱数シード（頑健性チェック用）
  - `POLARITY_FIRST_N` — polarity 固定に使う先頭件数
  - `BURST_MULTIPLIER`, `BURST_MIN_COUNT` — バースト閾値
- **頑健性チェック**: **本番 `SAMPLE_FRAC = 0.30`** で `SEED = 42` を本実行とし、`SEED = 1` を**保険**として 1 本追加実行済。
  - $\beta_{\text{typeA}}$: SEED=42 で $-1.129$、SEED=1 で $-1.020$ → **同符号・同桁**で再現。差 (0.11) は標準誤差 ($\approx 0.10$) と同程度。
  - 他の 3 係数 (`type_b`, `quality`, `log_ratings_count`) も全て同符号・同桁・$p<0.001$ で安定。
  - $p < 0.001$ が両 SEED で得られているため、SEED=2,3 は省略（時間制約と限界効用の観点）。
  - スモークテスト (frac=0.01) は検出力が低く SEED 振りで判定がパタつくため、頑健性チェックの対象外（パイプライン動作確認に専念）。

---

## A7 — 結果フル表 + 頑健性

**サンプル**
- N = 14,253 (政治トピック notes, frac=0.30)
- deleted = 4,049 (28.4%) / helpful = 10,204 (71.6%)
- TypeA バースト持ち note = 7,990 / TypeB バースト持ち note = 5,705 / バーストなし = 558
- 検出バースト総数 = 106,673 (note 単位は最初の 1 バーストのみ採用)

**ロジスティック回帰** (`deleted ~ type_a + type_b + quality + log_ratings_count`)

| 変数 | $\beta$ | $p$ | 95% CI |
|---|---:|---:|:---:|
| `type_a`            | $-1.129$ | $<0.001$ | $[-1.333,\ -0.926]$ |
| `type_b`            | $-0.404$ | $<0.001$ | $[-0.610,\ -0.198]$ |
| `quality`           | $-1.865$ | $<0.001$ | $[-2.005,\ -1.724]$ |
| `log_ratings_count` | $-0.603$ | $<0.001$ | $[-0.646,\ -0.559]$ |
| (const)             | $+3.727$ | $<0.001$ | $[\,3.478,\ 3.975\,]$ |

- Pseudo R² (Cox–Snell) = 0.147, $\log L = -7376.4$, Deviance = 14,753

**多重共線性診断 (本番実測)**

| 変数 | VIF | 判定 |
|---|---:|---|
| `type_a`            | 7.43 | 注意 |
| `type_b`            | 7.49 | 注意 |
| `quality`           | 1.01 | OK |
| `log_ratings_count` | 1.12 | OK |

- $\mathrm{corr}(\text{type\_a},\ \log\text{ratings\_count}) = 0.032$ (ほぼゼロ — `log_ratings_count` は独立な control として機能)
- $\mathrm{corr}(\text{type\_a},\ \text{type\_b}) = -0.923$ (1 note 1 バーストかつ 96% のノートが A/B に分類されるためほぼ相補) → これが type_a/type_b の VIF>5 の構造的理由
- **この相補性は悪性共線ではなくダミーコーディングの帰結**: 「バーストなし / TypeA / TypeB」の 3 値カテゴリを参照カテゴリ「バーストなし」+ 2 ダミーで表現しているだけ。各 $\beta$ は「バーストなし note と比べた効果」として解釈すれば共線性の影響は受けない。TypeA と TypeB を直接比べたいときは差 $\beta_\text{type\_a} - \beta_\text{type\_b} \approx -0.73$ を見ればよく、これは共線性の影響を受けにくい。
- catastrophic 判定の VIF>10 には未達。SE $\approx 0.10$ で全係数 $p<0.001$ → 悪性共線なら SE が膨張するはずなので、結論は安定。

**サンプル選抜の影響 (限界として要言及)**
- `deleted` は `CURRENTLY_RATED_HELPFUL` / `CURRENTLY_RATED_NOT_HELPFUL` の 2 値のみで定義 ([src/simple/regression.py:28](src/simple/regression.py#L28) の `VALID_STATUSES`)。`NEEDS_MORE_RATINGS` の note は除外。
- 検出バースト **106,673 件のうち最終特徴量に残るのは 13,695 件** のみ。バースト持ち note の大半は合意未達。

**符号の妥当性チェック**
- `quality` ↓ ($-1.87$): 質の高い note ほど消えにくい（事前予想と一致）
- `log_ratings_count` ↓ ($-0.60$): 評価数の多い note ほど消えにくい（人気度 control が効いている）
- バースト 2 変数も含めて 4 変数すべて負 → 「注目された / 質が高い note は消えにくい」という一貫した像。

**頑健性** (SEED=1 を保険として追加実行済): 全 4 係数が同符号・同桁・$p<0.001$ で再現。

| 変数 | SEED=42 β | SEED=1 β | $p$ |
|---|---:|---:|:---:|
| `type_a`            | $-1.129$ | $-1.020$ | $<0.001$ (両 run) |
| `type_b`            | $-0.404$ | $-0.384$ | $<0.001$ (両 run) |
| `quality`           | $-1.865$ | $-1.752$ | $<0.001$ (両 run) |
| `log_ratings_count` | $-0.603$ | $-0.615$ | $<0.001$ (両 run) |

> SEED=1 では N=14,170 / deleted=3,904 (27.6%) / TypeA=7,795 / TypeB=5,833 / バースト総数 106,836。サンプル構成も SEED=42 (N=14,253) と整合。

**限界**
- 政治トピック判定がキーワードベース / polarity 次元 2 / note 単位集計
- TypeA は投票方向を区別しないため、本設計では「陣営反応 = 攻撃」と「陣営防衛 = 防御」を分離できない（→ 今後の課題）
- frac=0.30 単一規模での検証 (SEED=42 と SEED=1 の 2 本で頑健性は確認済)

---

## A8 — 想定質問 (インデックス)

> 各回答は対応する A セクション / 本編スライドを指す形で簡潔に。詳細はリンク先で。

### 結果の解釈

- **$\beta_\text{typeA}$ が負 (仮説と逆) は何を意味する?** → 本編スライド 6 に解釈サマリ。実装上の根拠は [src/simple/burst.py:62-98](src/simple/burst.py#L62-L98) (polarity 分散のみで投票方向を見ない)。
- **TypeA を pro-rally / con-rally に分けたら?** → 未実装。バースト中の helpfulnessLevel 平均符号で 2 分する拡張が候補。今後の課題。
- **両方有意（負）の場合コード側の判定文は?** → [src/simple/regression.py:127-130](src/simple/regression.py#L127-L130) の verdict 文字列は **p のみ**で「両方有意 → 部分支持」を返す簡易実装。一方、**本研究の事前基準はスライド 1 の「$\beta_\text{typeA}>0$ かつ $p<0.05$」** であり、符号が逆 ($\beta_\text{typeA}=-1.13$) の時点で棄却が事前基準と整合する (上書きではなく一貫)。コード側の文字列は符号チェックを含めない初期実装の名残で、判定の根拠ではない。

### 設計の妥当性

- 「TruncatedSVD で次元 2 にするのは根拠ある?」
  → A2 参照。PC1=0.2% / PC2=0.1% (本番実測) で比率は小さいが、これは行列が極端に疎なため。後段の $\beta_\text{typeA}$ が SEED 間で同符号・同桁に再現することが間接的 validation。

- 「バースト閾値 3.0 と 5 件はどう決めた?」
  → notebook 設定セルで変更可能（[scripts/run_simple.py:57-58](scripts/run_simple.py#L57-L58)）。本研究ではこの値の頑健性まではスコープ外。閾値を変えると TypeA/B 件数のスケールは動くが、median 分割なので相対的な分類は大きくは変わらない設計。

- 「TypeA/B を `polarity_variance` の median で 2 分するのは恣意的では? 強制的に 50/50 になるし、絶対閾値ではない」
  → **指摘は正しい**。median 分割は (1) 結果集合の 50% を強制的に TypeA に振る、(2) 閾値そのものはサンプル構成依存、(3) 連続量 `polarity_variance` を二値化するので情報損失あり、という 3 点で恣意的。本研究で採用したのは以下の理由:
    - **相対基準は意図的設計**: 絶対閾値 (例: $V < 0.1$ を TypeA) を置くとサンプル構成 (frac, 政治トピック比率) で TypeA の定義自体が変わり、「同じ分析を別データセットでやった」比較が崩れる。median 分割なら「この分析内の同陣営度上位 50%」と一貫した解釈ができる。
    - **median の位置は SEED 間で安定**: SEED=42 と SEED=1 で $\beta_\text{typeA}$ が同符号・同桁 ($-1.13$ vs $-1.02$) → median 自体が SEED で大きく動いていない実証。
    - **二値化は解釈優先**: 連続変数だと $\beta$ の単位が「variance 1 単位上がるごとの log-odds 変化」になり聴衆に伝わりにくい。「分散小さい群 vs 大きい群」の対比の方が直感的。
  → **限界として明示**: 連続版 (`polarity_variance` をそのまま回帰に入れる) や quartile 分割 (上下 25%) での sensitivity check は未実施 — 今後の課題。

- 「quality の LLM ラベル 200 件で足りる?」
  → 事前 CV AUC = 0.894 (A5)、本番でも $\beta=-1.86$ を維持 (A7) で外挿は妥当。再学習は `scripts/experiments/train_quality_model.py`。

- 「`trend` を control に入れたほうが良くない?」
  → 本編スライド 5 参照。`helpfulnessLevel` 系列由来で目的変数と同じ生データ → bad control。

- 「`type_a` と `log_ratings_count` の正相関で多重共線性は?」
  → A7 の VIF 表参照。当初構造的正相関を懸念したが、実測 corr=0.032 でほぼゼロ。type_a/type_b の VIF>5 は両者の相補性 (corr=-0.923) によるもので catastrophic ではない。

### サンプル設計と外的妥当性

- 「`deleted` は `CURRENTLY_RATED_NOT_HELPFUL` だけ? `NEEDS_MORE_RATINGS` はどうした?」
  → A7「サンプル選抜の影響」参照。`HELPFUL`/`NOT_HELPFUL` の 2 値のみで `NEEDS_MORE_RATINGS` は除外。検出バースト 106,673 → 残 13,695 でバースト持ち note の大半は合意未達 — 限界として要言及。

- 「frac=0.30 単一規模での検証で十分?」
  → A6/A7 の SEED=1 比較で全 4 係数が同符号・同桁・$p<0.001$ を確認済。$p<0.001$ & SE 小なので frac を上げても点推定はほとんど動かない見込み。

- 「政治トピック判定がキーワード40語ベース、これでカバー漏れない?」
  → A1 の通り precision 側 (`\b` で誤マッチ排除) は対応済。**recall 側** (政治的だがキーワードに無い note) は取りこぼす。今回の 14k notes は "政治の中でも辞書ヒットした部分" の subsample という限界がある。

- 「1 note 1 バーストの簡略化で情報を捨てていない?」
  → 捨てている (`break` で先頭バーストのみ採用、[src/simple/burst.py:55](src/simple/burst.py#L55))。複数バーストを区別する設計 (e.g. 最も激しいバーストを採用、バースト回数を control 変数化) は今後の拡張候補。

- 「multiple comparison 補正は?」
  → 本研究はスコープ外。素直に $p<0.05$ で判定。本番では全 4 変数とも $p<0.001$ なので Bonferroni 補正 ($\alpha = 0.05/4 = 0.0125$) でも結論は同じ。
