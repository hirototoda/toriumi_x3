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
| TypeA / TypeB 両方有意 | 仮説部分支持 — 自然拡散も効く（注） |
| TypeB のみ有意 | 仮説不支持 |
| どちらも非有意 | 仮説不支持 / サンプル不足 |

> 注: 両方有意のときは $\beta_\text{typeA}$ と $\beta_\text{typeB}$ の**大小**で陣営反応の主因度を議論する。$\beta_\text{typeA} > \beta_\text{typeB}$ なら陣営反応寄り、$\beta_\text{typeB} > \beta_\text{typeA}$ なら自然拡散寄りと読む。

---

## スライド 4 — 使用データ

X が公開している 3 種の TSV（日次更新）:

- `notes-*.tsv` — note 本体（`noteId`, 本文 `summary`, classification など）
- `ratings-*.tsv` — 評価ログ（`raterParticipantId`, `noteId`, `helpfulnessLevel`, `createdAtMillis`）
- `noteStatusHistory-*.tsv` — 各 note の最終ステータス（`CURRENTLY_RATED_HELPFUL` 等）

**Simple 版** では `noteId` をランダムサンプリング → 該当 rating/history のみ読み込み。
- **本番**: `SAMPLE_FRAC = 0.30`（計算時間 10〜15 分目安）
- **スモークテスト**: `SAMPLE_FRAC = 0.01`（パイプライン動作確認用、N≈数百）

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

- ノート本文 `summary` を **単語境界 `\b` 付き正規表現**でフィルタ（政治キーワード辞書、約 40 語: `trump`, `election`, `abortion`, `immigration`, `supreme court`, ... ）。
  - 旧実装の単純な部分一致では `vote` → `devote` / `voter` 等が誤マッチしていたため `\b...\b` で囲った。
- サンプリング（noteId 単位、rater 単位ではない）:
  $$ \text{sample\_ids} = \text{RandomSample}\big(\{\text{noteId}\}_\text{political},\ \text{frac},\ \text{seed}\big) $$
  - デフォルト `frac = 0.30`, `seed = 42`（再現性のため固定）
  - サンプリング済 noteId に紐づく ratings のみチャンク読み → メモリ節約
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

**相対基準で分類**（残ったバーストの polarity 分散の**中央値** $\tilde V$ を閾値）:
$$ \text{burst\_type} = \begin{cases} A & \text{if } V \le \tilde V \quad(\text{同じ陣営} = \text{陣営反応}) \\ B & \text{if } V > \tilde V \quad(\text{バラバラ} = \text{自然拡散}) \end{cases} $$

**polarity が決まる rater が 2 人未満のバーストの扱い** (= 分類不能):

- バースト行ごと除外 → そのノートは「バーストなし」として回帰に入る (`type_a = type_b = 0`)。
- 旧実装は強制 B にしていたがバイアスを生むため不採用。
- NaN を残して回帰側で dropna する案も試したが NA 比較で TypeError になるため不採用。

**設計のポイント**

- 閾値を中央値に取ることで「この研究全体で相対的にどうか」を見る設計（絶対閾値はサンプル依存になる）。
- 1 ノート 1 バーストなので `type_a` と `type_b` が同時に 1 になることはない。

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

> 補足: スモークテスト (frac=0.01, N=461) でも `log_ratings_count` は $\beta=-0.649,\ p<0.001$ と強く有意で、人気度 control を入れる設計の正しさが実データでも確認できている。

---

## スライド 13 — 実装と再現性

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

## スライド 14 — 結果と議論

**実行条件**: `SAMPLE_FRAC = 0.30`, `SEED = 42`, `POLARITY_FIRST_N = 50`, `BURST_MULTIPLIER = 3.0`, `BURST_MIN_COUNT = 5`

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
- VIF / 相関行列はサマリ未掲載（notebook 出力で別途確認）

**判定**: **仮説は棄却** — それも*逆方向に*有意。
当初仮説は「TypeA バースト (陣営反応) が Not Helpful 化を**引き起こす** ($\beta_\text{typeA}>0$)」だったが、本番では $\beta_\text{typeA}=-1.13\ (p<0.001)$ と**強く負**。バーストを持つ note はむしろ **削除されにくい**。TypeB も負だが effect は小さい ($-0.40$)。$|\beta_\text{typeA}| > |\beta_\text{typeB}|$ で、TypeA の "保護効果" のほうが強い。

**符号の妥当性チェック**
- `quality` ↓ ($-1.87$): 質の高い note ほど消えにくい（事前予想と一致）
- `log_ratings_count` ↓ ($-0.60$): 評価数の多い note ほど消えにくい（人気度 control が効いている）
- バースト 2 変数も含めて 4 変数すべて負 → 「注目された / 質が高い note は消えにくい」という一貫した像。

**逆方向の解釈**: TypeA の定義はバースト参加者の **polarity 分散**が小さいことを捉えるだけで、**投票方向 (Helpful / Not Helpful) は区別していない**。同陣営による集中投票は本来「反対陣営が攻撃する」シナリオを想定していたが、実データでは **支持陣営が note を守るための rally** を多く拾っている可能性が高い。"陣営反応" 仮説の検証には、TypeA をさらに **pro-rally / con-rally** に分ける拡張設計が必要。

**頑健性** (SEED=1 を保険として追加実行済): 全 4 係数が同符号・同桁・$p<0.001$ で再現。$\beta_\text{typeA}$ は SEED=42 で $-1.129$、SEED=1 で $-1.020$ (差 0.11 ≈ SE) → **結論は SEED に頑健**。

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

### 結果の解釈について

- **「$\beta_\text{typeA}$ が負になった = 仮説とは逆だが、これは何を意味する?」**
  → TypeA の定義はバースト参加者の polarity *分散*のみを見ており、**投票の方向 (Helpful / Not Helpful) を区別していない**（[src/simple/burst.py:62-98](src/simple/burst.py#L62-L98)）。同陣営の集中投票には「反対派が note を攻撃」と「支持派が note を防衛」の両方が混ざる。本番で $\beta_\text{typeA}=-1.13$ と強い負だったことは、実データでは**防衛 rally のほうが多数派**だった可能性が高いことを示唆する。"陣営反応" 仮説の正面検証には設計拡張が必要 (下記)。

- **「TypeA を pro-rally / con-rally に分けたらどうなるの?」**
  → 本研究では未実装。バースト中の評価の helpfulnessLevel の平均符号で 2 分する拡張が考えられる。今後の課題。

- **「両方有意（負）の場合、コード側の判定文は何と出る?」**
  → 現状 [src/simple/regression.py:127-130](src/simple/regression.py#L127-L130) の verdict 文字列は **符号を見ず p のみ**で判定するため、「TypeA/B 両方有意 → 自然拡散も寄与 (仮説部分支持)」と出る。発表ではスライド 14 で**符号を踏まえた判定**に上書きしている。

### 設計の妥当性

- 「TruncatedSVD で次元 2 にするのは根拠ある?」
  → [src/simple/polarity.py](src/simple/polarity.py) で `explained_variance_ratio_` を出力。本番ログ実測値: **第 1 成分=[TODO_PC1]%, 第 2 成分=[TODO_PC2]%** (`data/processed/simple_run_log.txt` の `[polarity] ... explained_variance_ratio=[...]` 行から記入)。第 1 成分が支配的なら 1 軸でも近似可能だが、現設計は分散 = $\mathrm{Var}(x)+\mathrm{Var}(y)$ で 2 次元 polarity 平面の散らばりを測るため軸選択に robust。

- 「バースト閾値 3.0 と 5 件はどう決めた?」
  → notebook 設定セルで変更可能（[scripts/run_simple.py:57-58](scripts/run_simple.py#L57-L58)）。本研究ではこの値の頑健性まではスコープ外。閾値を変えると TypeA/B 件数のスケールは動くが、median 分割なので相対的な分類は大きくは変わらない設計。

- 「quality の LLM ラベル 200 件で足りる?」
  → 事前学習 CV AUC = 0.894。本番 14k notes でも `quality` は最大の effect ($\beta=-1.86$) を維持し、寄与度（domain_trust 44%）は事前知識と整合。再学習は `scripts/experiments/train_quality_model.py`。

- 「`trend` を control に入れたほうが良くない?」
  → 目的変数 `deleted` と**同じ生データ** (`helpfulnessLevel` 系列) から作られる **bad control**。入れると目的変数の一部を control する形になり $\beta_\text{typeA}$ を不当に押し下げる。

- 「`type_a` と `log_ratings_count` の正相関で多重共線性は?」
  → バースト判定が `min_count=5` 件以上を要求するため構造的な正相関は存在する（[src/simple/regression.py:14-20](src/simple/regression.py#L14-L20) 参照）。だからこそ `log_ratings_count` を control に入れている。本番ログ実測値 (`data/processed/simple_run_log.txt` の `[diag] VIF` 行):
    - `type_a` VIF = **[TODO]**, `type_b` VIF = **[TODO]**, `quality` VIF = **[TODO]**, `log_ratings_count` VIF = **[TODO]** — 全て 5 未満なら多重共線性は問題なし
    - 相関 `corr(type_a, log_ratings_count)` = **[TODO]** (`[diag] correlation matrix` から)

### サンプル設計と外的妥当性

- 「`deleted` は `CURRENTLY_RATED_NOT_HELPFUL` だけ? `NEEDS_MORE_RATINGS` はどうした?」
  → 回帰は `CURRENTLY_RATED_HELPFUL` / `CURRENTLY_RATED_NOT_HELPFUL` の 2 値のみで実施（[src/simple/regression.py:28](src/simple/regression.py#L28) の `VALID_STATUSES`）。「**最終的な合意に至った note**」だけが分析対象。
  → 副作用として、`NEEDS_MORE_RATINGS` の note は除外される。本番では検出バースト 106,673 件のうち最終特徴量に残るのは 13,695 件のみで、**バースト持ち note の大半は合意未達**。この選抜の影響は限界として要言及。

- 「frac=0.30 単一規模での検証で十分?」
  → 本番 $p<0.001$ で SE も小さいため、frac=0.50, 1.0 に上げても点推定はほとんど動かない見込み。SEED=1 の保険実行で同符号・同桁を確認することで頑健性は担保。

- 「政治トピック判定がキーワード40語ベース、これでカバー漏れない?」
  → 単語境界 `\b` 付き正規表現で `vote` → `devote` 等の誤マッチは排除済み（[src/simple/topic.py](src/simple/topic.py)）。ただし *recall* 側 (政治的だがキーワードに無い note) は取りこぼす。今回の被験 14k notes は "政治の中でも辞書ヒットした部分" の subsample である点は限界。

- 「1 note 1 バーストの簡略化で情報を捨てていない?」
  → 捨てている (`break` で先頭バーストのみ採用、[src/simple/burst.py:55](src/simple/burst.py#L55))。複数バーストを区別する設計 (e.g. 最も激しいバーストを採用、バースト回数を control 変数化) は今後の拡張候補。

- 「multiple comparison 補正は?」
  → 本研究はスコープ外。素直に $p<0.05$ で判定。本番では全 4 変数とも $p<0.001$ なので Bonferroni 補正 ($\alpha = 0.05/4 = 0.0125$) でも結論は同じ。
