"""
データロード + noteId サンプリング

設計:
  1. notes ファイル全部を読む (notes は ratings に比べて小さい)
  2. noteId を random_state 固定で frac サンプリング
  3. ratings をチャンク読みしながら、サンプリング済 noteId だけ拾う
  4. history も同じ noteId で絞る

これにより
  - メモリは「サンプリング後の ratings」だけで済む
  - polarity / burst の計算は「全データの真サンプル」になる
  - 同じ seed で再実行すれば結果が再現する
"""

from pathlib import Path

import pandas as pd

RATINGS_COLS = ["noteId", "raterParticipantId", "createdAtMillis", "helpfulnessLevel"]
NOTES_COLS   = ["noteId", "createdAtMillis", "summary"]
HISTORY_COLS = ["noteId", "currentStatus"]

_CHUNK = 500_000  # ratings ストリーム読みのチャンクサイズ


def _find(raw_dir: Path, prefix: str) -> list[Path]:
    paths = sorted(raw_dir.glob(f"{prefix}*.tsv"))
    if not paths:
        raise FileNotFoundError(f"{raw_dir} に {prefix}*.tsv が見つかりません")
    return paths


def load_notes(raw_dir: Path) -> pd.DataFrame:
    paths = _find(raw_dir, "notes")
    dfs = [pd.read_csv(p, sep="\t", usecols=NOTES_COLS, dtype={"noteId": str}) for p in paths]
    notes = pd.concat(dfs, ignore_index=True).drop_duplicates("noteId")
    print(f"[load] notes: {len(notes):,} unique")
    return notes


def sample_note_ids(notes: pd.DataFrame, frac: float, seed: int) -> set[str]:
    """noteId 単位で frac サンプリングする (rater サンプリングではない)"""
    sampled = notes["noteId"].sample(frac=frac, random_state=seed)
    print(f"[load] sampled noteIds: {len(sampled):,} ({frac*100:.0f}%)")
    return set(sampled)


def load_ratings_for_notes(raw_dir: Path, target_note_ids: set[str]) -> pd.DataFrame:
    """ratings をチャンク読みしながら target_note_ids にマッチする行だけ集める"""
    paths = _find(raw_dir, "ratings")
    kept = []
    for p in paths:
        print(f"[load]   scanning {p.name} ...")
        for chunk in pd.read_csv(
            p, sep="\t", usecols=RATINGS_COLS,
            dtype={"noteId": str, "raterParticipantId": str},
            chunksize=_CHUNK,
        ):
            kept.append(chunk[chunk["noteId"].isin(target_note_ids)])
    ratings = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=RATINGS_COLS)
    print(f"[load] ratings (sampled): {len(ratings):,} rows")
    return ratings


def load_history_for_notes(raw_dir: Path, target_note_ids: set[str]) -> pd.DataFrame:
    paths = _find(raw_dir, "noteStatusHistory")
    dfs = [pd.read_csv(p, sep="\t", usecols=HISTORY_COLS, dtype={"noteId": str}) for p in paths]
    history = pd.concat(dfs, ignore_index=True).drop_duplicates("noteId")
    history = history[history["noteId"].isin(target_note_ids)]
    print(f"[load] history (sampled): {len(history):,} rows")
    return history
