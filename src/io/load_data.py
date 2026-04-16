"""
公式Community Notesデータ(TSV)のローダー

データ入手元: https://communitynotes.x.com/guide/en/under-the-hood/download-data
ファイルサイズが大きいため、usecolsで必要なカラムのみ読み込む。
複数ファイル（ratings-00000〜00007等）がある場合は全て結合する。
max_files で読み込むファイル数を制限できる。
"""

import pandas as pd
from pathlib import Path


RATINGS_COLS = [
    "noteId", "raterParticipantId", "createdAtMillis", "helpfulnessLevel",
]
NOTES_COLS = [
    "noteId", "createdAtMillis", "summary",
]
HISTORY_COLS = [
    "noteId", "currentStatus",
]


def _find_files(directory: Path, prefix: str, max_files: int | None = None) -> list[Path]:
    """directory 内で prefix にマッチする .tsv を探す（max_files で上限指定可）"""
    candidates = sorted(directory.glob(f"{prefix}*.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"{directory} に {prefix}*.tsv が見つかりません。"
            f"\nhttps://communitynotes.x.com/guide/en/under-the-hood/download-data"
            f"\nからダウンロードして配置してください。"
        )
    if max_files is not None:
        candidates = candidates[:max_files]
    return candidates


def _load_multi(paths: list[Path], usecols, dtype, nrows: int | None = None) -> pd.DataFrame:
    """複数ファイルを読み込んで結合する。nrows は合計行数の上限。"""
    dfs = []
    remaining = nrows
    for path in paths:
        print(f"  Loading {path.name} ...")
        df = pd.read_csv(
            path, sep="\t", usecols=usecols, dtype=dtype,
            nrows=remaining,
        )
        dfs.append(df)
        print(f"    {len(df):,} rows")
        if remaining is not None:
            remaining -= len(df)
            if remaining <= 0:
                break
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(dfs)} file(s)")
    return combined


def load_ratings(raw_dir: Path, nrows: int | None = None, max_files: int | None = None) -> pd.DataFrame:
    """ratings*.tsv を読み込んで結合する。max_files でファイル数を制限。"""
    paths = _find_files(raw_dir, "ratings", max_files=max_files)
    return _load_multi(
        paths, usecols=RATINGS_COLS,
        dtype={"noteId": str, "raterParticipantId": str},
        nrows=nrows,
    )


def load_notes(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """notes*.tsv を全て読み込んで結合する"""
    paths = _find_files(raw_dir, "notes")
    return _load_multi(
        paths, usecols=NOTES_COLS,
        dtype={"noteId": str},
        nrows=nrows,
    )


def load_status_history(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """noteStatusHistory*.tsv を全て読み込んで結合する"""
    paths = _find_files(raw_dir, "noteStatusHistory")
    return _load_multi(
        paths, usecols=HISTORY_COLS,
        dtype={"noteId": str},
        nrows=nrows,
    )
