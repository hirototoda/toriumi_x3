"""
チャンク出力ファイル (*_f*_r*.csv) を 1 本の *_all.csv に統合する。

使い方:
  python scripts/merge_chunks.py

対象 (data/processed/ 配下):
  - bursts_f*_r*.csv            → bursts_all.csv
  - target_notes_f*_r*.csv      → target_notes_all.csv
  - topic_comparison_f*_r*.csv  → topic_comparison_all.csv

挙動:
  - concat してから noteId (または topic) でデドゥープ (最初を採用)
  - 既存の *_all.csv があれば上書き
  - 1 チャンクしか存在しなくても実行可能
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"


def _extract_chunk_tag(name: str) -> str:
    """'bursts_f0_r1.csv' → 'f0_r1' を取り出す"""
    stem = Path(name).stem
    # 末尾の _f?_r? パターンを取る
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.startswith("f") and p[1:].isdigit():
            return "_".join(parts[i:])
    return stem


def merge_one(pattern: str, out_name: str, dedup_on: str | None) -> None:
    """pattern にマッチする CSV を読み込んで out_name に結合保存する。

    dedup_on:
      - カラム名を指定: concat 後にそのキーでデドゥープ (最初を採用)
      - None:          デドゥープせず、全行残す (各行に chunk 列を付与)
    """
    paths = sorted(PROCESSED.glob(pattern))
    if not paths:
        print(f"  [skip] {pattern}: ファイルなし")
        return

    print(f"\n● {pattern} → {out_name}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        df["chunk"] = _extract_chunk_tag(p.name)
        print(f"    {p.name:40s}  {len(df):>8,} rows")
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    total = len(combined)

    if dedup_on and dedup_on in combined.columns:
        combined = combined.drop_duplicates(subset=[dedup_on], keep="first")
        deduped = total - len(combined)
        if deduped:
            print(f"    ⚠ {deduped:,} 件の重複 ({dedup_on}) を除去 (最初のレコードを採用)")

    out_path = PROCESSED / out_name
    combined.to_csv(out_path, index=False)
    print(f"    → {out_path}  ({len(combined):,} rows)")


def main() -> int:
    if not PROCESSED.exists():
        print(f"ERROR: {PROCESSED} が存在しません", file=sys.stderr)
        return 1

    print(f"Merging chunk outputs from {PROCESSED}")

    merge_one("bursts_f*_r*.csv",           "bursts_all.csv",           dedup_on="noteId")
    merge_one("target_notes_f*_r*.csv",     "target_notes_all.csv",     dedup_on="noteId")
    merge_one("topic_comparison_f*_r*.csv", "topic_comparison_all.csv", dedup_on=None)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
