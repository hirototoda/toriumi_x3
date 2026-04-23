"""
Simple版パイプライン (CS授業用)

設計方針:
  - 1 関数 1 役割、各ファイル 50〜100 行
  - サンプリングで全データ問題を回避 (案 C)
  - quality は LLM ラベル学習済みの重みをハードコード (透明)
  - 回帰は trend を除いた 4 変数 (type_a, type_b, quality, log_ratings_count)

呼び出し: scripts/run_simple.py 参照
"""
