"""
公式Community Notesデータ(TSV)のローダー

データ入手元: https://communitynotes.x.com/guide/en/under-the-hood/download-data
ファイルサイズが大きいため、usecolsで必要なカラムのみ読み込む。
複数ファイル（ratings-00000〜00007等）がある場合は全て結合する。

チャンク実行サポート (load_ratings のみ):
  - max_files:   読み込むファイル数の上限
  - file_offset: 先頭から file_offset 個のファイルを飛ばす
  - skip_rows:   対象ファイルを連結した仮想ストリームの先頭 skip_rows 行を飛ばす
                 (ファイル境界を自動でまたいで消化する)
  - nrows:       skip_rows 以降の合計行数の上限
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

# ストリーミング読み込みのチャンク行数 (load_ratings のチャンク実行向け)
_READ_CHUNKSIZE = 200_000


def _find_files(
    directory: Path,
    prefix: str,
    max_files: int | None = None,
    file_offset: int = 0,
) -> list[Path]:
    """directory 内で prefix にマッチする .tsv を探す。

    file_offset で先頭から N 個を飛ばし、max_files で残りの上限を設定する。
    """
    candidates = sorted(directory.glob(f"{prefix}*.tsv"))
    if not candidates:
        raise FileNotFoundError(
            f"{directory} に {prefix}*.tsv が見つかりません。"
            f"\nhttps://communitynotes.x.com/guide/en/under-the-hood/download-data"
            f"\nからダウンロードして配置してください。"
        )
    if file_offset:
        if file_offset >= len(candidates):
            raise FileNotFoundError(
                f"file_offset={file_offset} が {prefix}*.tsv の個数 {len(candidates)} 以上です。"
            )
        candidates = candidates[file_offset:]
    if max_files is not None:
        candidates = candidates[:max_files]
    return candidates


def _load_multi(
    paths: list[Path],
    usecols,
    dtype,
    nrows: int | None = None,
    skip_rows: int = 0,
) -> pd.DataFrame:
    """複数ファイルを読み込んで結合する。

    skip_rows>0 または nrows 指定あり: チャンクストリーム読みで範囲を切り出す。
    どちらも未指定: 従来通り各ファイルを一括読み。
    """
    # 従来パス (全量読み)
    if skip_rows == 0 and nrows is None:
        dfs = []
        for path in paths:
            print(f"  Loading {path.name} ...")
            df = pd.read_csv(path, sep="\t", usecols=usecols, dtype=dtype)
            dfs.append(df)
            print(f"    {len(df):,} rows")
        combined = pd.concat(dfs, ignore_index=True)
        print(f"  Total: {len(combined):,} rows from {len(dfs)} file(s)")
        return combined

    # チャンクパス (skip_rows / nrows 指定時)
    dfs = []
    remaining_skip = skip_rows
    remaining_take = nrows  # None は無制限

    for path in paths:
        if remaining_take is not None and remaining_take <= 0:
            break

        print(f"  Loading {path.name} ...")
        file_taken = 0
        file_skipped = 0
        reader = pd.read_csv(
            path, sep="\t", usecols=usecols, dtype=dtype,
            chunksize=_READ_CHUNKSIZE,
        )
        for chunk in reader:
            n = len(chunk)

            # このチャンクを丸ごとスキップ
            if remaining_skip >= n:
                remaining_skip -= n
                file_skipped += n
                continue

            # 部分スキップ
            if remaining_skip > 0:
                chunk = chunk.iloc[remaining_skip:]
                file_skipped += remaining_skip
                remaining_skip = 0

            # take 上限超過なら切り詰めて終了
            if remaining_take is not None and len(chunk) >= remaining_take:
                chunk = chunk.iloc[:remaining_take]
                dfs.append(chunk)
                file_taken += len(chunk)
                remaining_take = 0
                break

            dfs.append(chunk)
            file_taken += len(chunk)
            if remaining_take is not None:
                remaining_take -= len(chunk)

        msg = f"    {file_taken:,} rows"
        if file_skipped:
            msg += f" (skipped {file_skipped:,})"
        print(msg)

    if not dfs:
        return pd.DataFrame(columns=list(usecols))

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} rows from {len(paths)} file(s)")
    return combined


def load_ratings(
    raw_dir: Path,
    nrows: int | None = None,
    max_files: int | None = None,
    file_offset: int = 0,
    skip_rows: int = 0,
) -> pd.DataFrame:
    """ratings*.tsv を読み込んで結合する。

    チャンク実行時は file_offset と skip_rows で処理範囲を指定する。
    """
    paths = _find_files(raw_dir, "ratings", max_files=max_files, file_offset=file_offset)
    return _load_multi(
        paths, usecols=RATINGS_COLS,
        dtype={"noteId": str, "raterParticipantId": str},
        nrows=nrows, skip_rows=skip_rows,
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
