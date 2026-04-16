# コミュニティノート「信者バトル」仮説検証

## 仮説

Xコミュニティノートのステータス変化（Helpful / Not Helpful / 削除など）は、ノートの品質ではなく、**同一政治クラスタによる集中的な評価（陣営反応）**によって引き起こされている。

## データ入手手順

本プロジェクトでは、X公式が公開しているCommunity Notesのパブリックデータを使用する（独自スクレイピングはしない）。

1. 公式ページを開く: https://communitynotes.x.com/guide/en/under-the-hood/download-data
2. 以下の3ファイルをダウンロードする:
   - `notes.tsv`
   - `ratings.tsv`
   - `noteStatusHistory.tsv`
3. ダウンロードしたファイルを `data/raw/` に配置する
4. ダウンロード日を `docs/data_source.md` に記録する

> **注意**: データファイルは数百MB〜GB級のため、Gitには含めない（`.gitignore`で除外済み）。

## 分析パイプライン（5ステップ）

| Step | 内容 | 担当 |
|------|------|------|
| Step 1 | 前処理・polarity計算（3TSV結合、bridging再実装、最初の50件で固定） | Sさん (p3-4) |
| Step 2 | トピック分類（政治トピック抽出、APIコスト$0目標） | Gさん (p5-6) |
| Step 3 | バースト検出・分類（3倍速+5件以上、TypeA/TypeB判定） | Gさん (p5-6) |
| Step 4 | ロジスティック回帰（β0〜β4推定、統制変数で品質を排除） | 自分 (p7-9) |
| Step 5 | 品質スコアQ・ターゲット抽出・検証 | 自分 (p7-9) |

### 結論の判断基準

- **TypeAのみ有意** → 陣営反応が主因（仮説支持）
- **TypeA・B両方有意** → 自然拡散が主因の可能性
- **どちらも無意** → 別要因を探す

## フォルダ構造

```
.
├── README.md               # 本ファイル
├── TODO.md                 # TODOリスト
├── .gitignore
├── requirements.txt
├── data/
│   ├── raw/                # 公式TSVを配置（git管理外）
│   ├── interim/            # 中間生成物
│   └── processed/          # 最終分析用データ
├── scripts/
│   └── download_data.sh    # データダウンロード補助
├── src/
│   ├── io/                 # TSVローダー
│   ├── step1_preprocess/   # 結合・polarity・フィルタ
│   ├── step2_topic/        # トピック分類
│   ├── step3_burst/        # バースト検出・分類
│   ├── step4_regression/   # ロジスティック回帰
│   └── step5_target/       # 品質スコア・ターゲット抽出・検証
├── notebooks/              # Jupyter探索・結果確認
├── tests/
└── docs/                   # スライドメモ・データソースメモ
```

## セットアップ

```bash
pip install -r requirements.txt
```

## 担当分担

- **Sさん**: p3-4（データ前処理・Step1 polarity計算）
- **Gさん**: p5-6（Step2 トピック分類・Step3 バースト検出）
- **自分**: p7-9（Step4 回帰・Step5 品質スコア・結論）
