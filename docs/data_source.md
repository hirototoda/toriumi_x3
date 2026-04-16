# データソース

## 公式ダウンロードページ

https://communitynotes.x.com/guide/en/under-the-hood/download-data

X社が公式に公開しているCommunity Notesのパブリックデータを使用する。
**独自スクレイピングはしない方針**。データは定期的に更新される。

## 使用ファイルとカラム

### notes.tsv

| カラム名 | 型 | 説明 |
|----------|------|------|
| noteId | int64 | ノートの一意ID |
| createdAtMillis | int64 | ノート作成時刻（UNIXミリ秒） |
| summary (or text) | str | ノート本文 |

> TODO: 実データのヘッダーで正確なカラム名（text or summary）を確認する

### ratings.tsv

| カラム名 | 型 | 説明 |
|----------|------|------|
| noteId | int64 | 評価対象ノートのID |
| raterParticipantId | str | 評価者のID |
| createdAtMillis | int64 | 評価時刻（UNIXミリ秒） |
| helpfulnessLevel | str | 評価レベル（HELPFUL / SOMEWHAT_HELPFUL / NOT_HELPFUL） |

### noteStatusHistory.tsv

| カラム名 | 型 | 説明 |
|----------|------|------|
| noteId | int64 | ノートのID |
| timestampMillis | int64 | ステータス変化時刻（UNIXミリ秒） |
| noteStatus | str | ステータス（CURRENTLY_RATED_HELPFUL / CURRENTLY_RATED_NOT_HELPFUL 等） |

> TODO: 実データのヘッダーで正確なカラム名を確認する（timestampMillisOfFirstNonNMRStatus / currentStatus 等）

## 3ファイルの結合キー

`noteId` で結合する。

## ダウンロード記録

| 日付 | ダウンロード者 | 備考 |
|------|----------------|------|
| (未実施) | - | - |
