"""
公式Community Notesデータ(TSV)のローダー

データ入手元: https://communitynotes.x.com/guide/en/under-the-hood/download-data
ファイルサイズが大きいため、usecolsで必要なカラムのみ読み込む。
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


def _find_file(directory: Path, prefix: str) -> Path:
    """directory 内で prefix にマッチする .tsv を探す"""
    candidates = sorted(directory.glob(f"{prefix}*.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"{directory} に {prefix}*.tsv が見つかりません。"
            f"\nhttps://communitynotes.x.com/guide/en/under-the-hood/download-data"
            f"\nからダウンロードして配置してください。"
        )
    return candidates[0]


def load_ratings(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """ratings.tsv を読み込む(必要カラムのみ)"""
    path = _find_file(raw_dir, "ratings")
    print(f"  Loading {path.name} ...")
    df = pd.read_csv(
        path, sep="\t", usecols=RATINGS_COLS,
        dtype={"noteId": str, "raterParticipantId": str},
        nrows=nrows,
    )
    print(f"    {len(df):,} rows")
    return df


def load_notes(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """notes.tsv を読み込む(必要カラムのみ)"""
    path = _find_file(raw_dir, "notes")
    print(f"  Loading {path.name} ...")
    df = pd.read_csv(
        path, sep="\t", usecols=NOTES_COLS,
        dtype={"noteId": str},
        nrows=nrows,
    )
    print(f"    {len(df):,} rows")
    return df


def load_status_history(raw_dir: Path, nrows: int | None = None) -> pd.DataFrame:
    """noteStatusHistory.tsv を読み込む(必要カラムのみ)"""
    path = _find_file(raw_dir, "noteStatusHistory")
    print(f"  Loading {path.name} ...")
    df = pd.read_csv(
        path, sep="\t", usecols=HISTORY_COLS,
        dtype={"noteId": str},
        nrows=nrows,
    )
    print(f"    {len(df):,} rows")
    return df
