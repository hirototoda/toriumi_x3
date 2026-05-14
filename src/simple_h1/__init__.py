"""
Simple 版パイプラインの H1 派生版

H1 仮説:
  TypeA バーストの負係数 (β=-1.13) は、バーストが NOT_HELPFUL 一斉ではなく
  HELPFUL 一斉が多いことに由来する可能性がある。
  → バーストの「方向」(helpful 多数 / nothelp 多数) を切って、
    type_a × 方向, type_b × 方向 の 4 ダミーで回帰しなおす。

設計方針:
  - src/simple/ は書き換えない (コードも結果もそのまま保存)
  - 差分のある burst.py / regression.py だけ本パッケージに置く
  - load / topic / polarity / quality はそのまま src.simple から import

呼び出し: scripts/experiments/run_simple_h1.py
"""
