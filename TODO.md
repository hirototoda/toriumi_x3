# TODO

## 前提: 公式データのダウンロード

- [ ] https://communitynotes.x.com/guide/en/under-the-hood/download-data から
      `notes.tsv`, `ratings.tsv`, `noteStatusHistory.tsv` を `data/raw/` にダウンロード
- [ ] ダウンロード日付を `docs/data_source.md` に記録

## 今日の授業のゴール1: 手動作業の前まで完了させる

- [ ] Step0 I/O: 公式TSVのヘッダー確認 & usecols指定でのローダー実装
- [ ] Step1 前処理: 3ファイルをnoteIdで結合するコード
- [ ] Step1 polarity計算: bridgingアルゴリズム(公式OSS)の再実装
- [ ] Step1 polarity固定: 各評価者の最初の50件でpolarityを確定
- [ ] Step1 フィルタ: 総評価数20件以上のノートに絞り込み
- [ ] Step2 トピック分類: 政治トピック抽出(APIコスト$0の手法を確定・実装)
- [ ] Step3 バースト検出: 平均評価速度の3倍以上 かつ 5件以上の区間抽出
- [ ] Step3 バースト分類: polarity分散でTypeA/TypeBラベル付け

→ ここまで完了で「手動作業(200件ラベル付け)を始められる状態」

## 今日の授業のゴール2: その後のコードを基本完成させる

- [ ] Step4 統制変数: Trend(過去スコア推移) と Quality(Q) の算出
- [ ] Step4 ロジスティック回帰: β0〜β4推定 + 有意性検定(statsmodels)
- [ ] Step5 品質スコアQ: URL数 + ドメイン信頼性 + 文字数(正規化)
- [ ] Step5 ターゲット抽出: {Q上位25%} ∩ {TypeAバースト} ∩ {非Helpful}
- [ ] Step5 検証パイプライン: 200件サンプリング+ラベル投入待ち状態まで

## 授業後に残す手動作業

- [ ] 200件のノートを人手/LLMで「高品質/低品質」にラベル付け
- [ ] Qスコアとの一致率を確認

## 注意点

- データは公式公開TSVのみを使用(独自スクレイピングはしない)
- Step2のトピック分類手法が未定 → 早い段階で方針を決め切る
- Step1の「最初の50件でpolarity固定」は循環論法回避のため必須
- 品質スコアQはhelpfulness投票と独立であることが仮説検証の生命線
