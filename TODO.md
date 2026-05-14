# TODO — 発表までの作業

スライド `docs/slides_simple.md` の Q&A 数値埋め込みから発表当日まで。
Colab で frac=0.30, SEED=42 の cell 9 を実行中 → 完了後に着手。

## Phase 1 — 数値埋め込み (cell 9 完了直後, ~5 分)

- [ ] cell 9 完了確認 → 以下 4 ブロックをチャットに貼り付け
  - [ ] `[polarity] ... explained_variance_ratio=[..., ...]`
  - [ ] `[diag] VIF (Variance Inflation Factor):` 4 行
  - [ ] `[diag] correlation matrix:` 表
  - [ ] (おまけ) burst_helpfulness 再実行後の `[TypeA]/[TypeB] mean_score`
- [ ] `docs/slides_simple.md` の placeholder 7 箇所を実数値に置換
  - [ ] Q5 (L317): `TODO_PC1`, `TODO_PC2`
  - [ ] Q9 (L330-331): VIF×4 と相関×1
- [ ] `data/processed/runs_2026-05-14.md` の Run B 末尾テンプレ 3 ブロックに実値流し込み
- [ ] 1 コミットで commit & push

## Phase 2 — 内容レビュー (~15〜30 分)

- [ ] Q5/Q9 を音読チェック (数値が文意と整合: VIF 全部 < 5 か、PC1 が支配的か等)
- [ ] スライド 1〜14 本文の最終読み合わせ (Q&A 数値追加で矛盾が出ていないか)
- [ ] 想定質問パート全体の流れ確認 (抜け漏れがないか)

## Phase 3 — PDF レンダリング & 見た目チェック (~10〜20 分)

- [ ] `slides_simple.md` → `slides_simple.pdf` 再生成 (普段の手順: Marp / Pandoc 等)
- [ ] PDF 目視: 数式の崩れ / コードブロックはみ出し / ページ割り / フッター
- [ ] 崩れがあれば markdown 修正 → 再レンダリング

## Phase 4 — 発表準備 (~30〜60 分)

- [ ] ドライラン 1 回 (時計を見て 14 枚 × 10〜12 分の想定通りか)
- [ ] talking points メモ (特に Q5/Q9 の「なぜ 2 次元」「なぜ多重共線性 OK」を即答できるように)
- [ ] 想定質問 Q1〜Qn の口頭回答練習 (埋めた数値を口で言える状態に)

## Phase 5 — 発表当日

- [ ] PDF をローカル & クラウド両方に持参 (バックアップ)
- [ ] Colab notebook も開いておく (再現性質問に備える)
- [ ] 発表 (10〜12 分) + Q&A

## クリティカルパス

Phase 1 → 3 → 4.1 (ドライラン) が最短。Phase 2 / 4.2 / 4.3 は時間が許す限り深掘り。
