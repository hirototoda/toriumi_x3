# トピック選択戦略: A+B 補強案

**ステータス**: 未実装（拡張用に保存）

---

## 現在の実装: 案A（トピック指定型）

複数の政治トピックに絞ってパイプラインを並列実行し、トピック別の比較表を出す。

### 実装済みトピック

| トピック | キーワード |
|---|---|
| vaccine/covid | vaccine, covid, pandemic, mask, booster, pfizer, moderna |
| israel/palestine | israel, palestine, gaza, hamas, idf, netanyahu, ceasefire |
| trump | trump, maga, indictment, mar-a-lago, j6, january 6 |
| immigration | immigration, border, migrant, asylum, deportation, illegal alien |
| gun control | gun control, second amendment, 2nd amendment, shooting, nra, firearm |

---

## 拡張案: B → A 補強（データドリブン + トピック確認）

### 概要

トピックを事前に決めず、データから「陣営が戦っている場所」を自動検出し、
そのノートがどのトピックに属するかを後から確認する。

### 実装方針

```python
# Step B-1: 各ノートの「評価者polarity分散」を計算
for note_id in political_notes:
    raters = ratings[ratings.noteId == note_id]
    px = raters.merge(polarity)["polarity_x"]
    py = raters.merge(polarity)["polarity_y"]
    note_polarity_var[note_id] = px.var() + py.var()

# Step B-2: 分散上位25%を「戦場ノート」として抽出
battleground = notes where note_polarity_var >= 75th percentile

# Step B-3: 戦場ノートの summary を見てトピックを確認
#   → 「データが自動的に見つけた戦場は○○トピックだった」

# Step B-4: 発見されたトピックで案Aの回帰を実行（確認）
```

### 発表での使い方

```
1. まず B で「戦場ノート」を自動検出（恣意性ゼロ）
2. 「データが見つけた戦場は○○トピックだった」と報告
3. 確認として A で○○トピックに絞って回帰実行
4. → 「データドリブンで発見し、トピック絞り込みで確認」
```

### メリット

- 「なぜそのトピックを選んだの？」→「選んだのではなく、データが見つけた」
- 案A の具体性 + 案B の方法論的強さ、両取り
- 査読やゼミでの質疑に耐える
