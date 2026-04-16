#!/bin/bash
# Community Notes 公式データダウンロード補助スクリプト
#
# 公式ページ: https://communitynotes.x.com/guide/en/under-the-hood/download-data
# ブラウザで上記ページを開き、最新の notes.tsv, ratings.tsv, noteStatusHistory.tsv を
# data/raw/ 配下にダウンロードしてください。
#
# TODO: 公式ダウンロードURLが固定化された場合、curl/wgetで自動化する

echo "公式ページを開いてください:"
echo "https://communitynotes.x.com/guide/en/under-the-hood/download-data"
echo ""
echo "以下の3ファイルを data/raw/ にダウンロードしてください:"
echo "  - notes.tsv"
echo "  - ratings.tsv"
echo "  - noteStatusHistory.tsv"
