"""
中間成果物のキャッシュ。

run_pipeline.py と run_by_topic.py を続けて走らせた際に、
同じデータ読み込み・polarity・quality を二重に計算しないための薄いラッパ。

保存先:
  data/cache/
    ratings_<hash>.pkl.gz
    notes_<hash>.pkl.gz
    history_<hash>.pkl.gz
    polarity_<hash>.pkl.gz
    quality_<hash>.pkl.gz

キャッシュキー:
  入力 TSV の (ファイル名, サイズ) と、計算パラメータのハッシュ。
  チャンクごとに自動で別キーになるため、stale になることはない想定。

無効化:
  不要になったら data/cache/ を丸ごと削除すれば良い。
  環境変数 TORIUMI_NO_CACHE=1 でランタイム無効化も可能。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache"

_DISABLED = os.environ.get("TORIUMI_NO_CACHE", "").strip() not in ("", "0", "false", "False")


def _hash(obj) -> str:
    s = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _file_sig(paths) -> list:
    return [(p.name, p.stat().st_size) for p in paths]


def _read(path: Path) -> Optional[pd.DataFrame]:
    if _DISABLED or not path.exists():
        return None
    try:
        return pd.read_pickle(path, compression="gzip")
    except Exception as e:
        print(f"  [cache] read failed ({e}); recomputing {path.name}")
        return None


def _write(df: pd.DataFrame, path: Path) -> None:
    if _DISABLED:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_pickle(tmp, compression="gzip")
    os.replace(tmp, path)


def ratings_cache_tag(
    raw_dir: Path,
    *,
    nrows: Optional[int],
    max_files: Optional[int],
    file_offset: int,
    skip_rows: int,
) -> str:
    from src.io.load_data import _find_files
    paths = _find_files(raw_dir, "ratings", max_files=max_files, file_offset=file_offset)
    return _hash({"sig": _file_sig(paths), "nrows": nrows, "skip": skip_rows})


def notes_cache_tag(raw_dir: Path) -> str:
    from src.io.load_data import _find_files
    paths = _find_files(raw_dir, "notes")
    return _hash({"sig": _file_sig(paths)})


def history_cache_tag(raw_dir: Path) -> str:
    from src.io.load_data import _find_files
    paths = _find_files(raw_dir, "noteStatusHistory")
    return _hash({"sig": _file_sig(paths)})


def load_ratings_cached(
    raw_dir: Path,
    *,
    nrows: Optional[int] = None,
    max_files: Optional[int] = None,
    file_offset: int = 0,
    skip_rows: int = 0,
) -> pd.DataFrame:
    from src.io.load_data import load_ratings
    tag = ratings_cache_tag(
        raw_dir, nrows=nrows, max_files=max_files,
        file_offset=file_offset, skip_rows=skip_rows,
    )
    path = CACHE_DIR / f"ratings_{tag}.pkl.gz"
    df = _read(path)
    if df is not None:
        print(f"  [cache HIT] ratings ({len(df):,} rows) ← {path.name}")
        return df
    print(f"  [cache miss] ratings; loading from TSV...")
    df = load_ratings(
        raw_dir, nrows=nrows, max_files=max_files,
        file_offset=file_offset, skip_rows=skip_rows,
    )
    _write(df, path)
    return df


def load_notes_cached(raw_dir: Path) -> pd.DataFrame:
    from src.io.load_data import load_notes
    tag = notes_cache_tag(raw_dir)
    path = CACHE_DIR / f"notes_{tag}.pkl.gz"
    df = _read(path)
    if df is not None:
        print(f"  [cache HIT] notes ({len(df):,} rows)")
        return df
    print(f"  [cache miss] notes; loading from TSV...")
    df = load_notes(raw_dir)
    _write(df, path)
    return df


def load_status_history_cached(raw_dir: Path) -> pd.DataFrame:
    from src.io.load_data import load_status_history
    tag = history_cache_tag(raw_dir)
    path = CACHE_DIR / f"history_{tag}.pkl.gz"
    df = _read(path)
    if df is not None:
        print(f"  [cache HIT] history ({len(df):,} rows)")
        return df
    print(f"  [cache miss] history; loading from TSV...")
    df = load_status_history(raw_dir)
    _write(df, path)
    return df


def compute_polarity_cached(
    ratings_df: pd.DataFrame,
    *,
    first_n: int,
    ratings_tag: str,
) -> pd.DataFrame:
    """
    ratings_tag は ratings キャッシュのハッシュ文字列 (ratings_cache_tag で取得)。
    ratings_df 自体をハッシュすると巨大で遅いため、呼び出し側から渡す方式にしている。
    """
    from src.step1_preprocess.polarity import compute_polarity
    tag = _hash({"ratings": ratings_tag, "first_n": first_n})
    path = CACHE_DIR / f"polarity_{tag}.pkl.gz"
    df = _read(path)
    if df is not None:
        print(f"  [cache HIT] polarity ({len(df):,} raters)")
        return df
    print(f"  [cache miss] polarity; computing SVD...")
    df = compute_polarity(ratings_df, first_n=first_n)
    _write(df, path)
    return df


def compute_quality_cached(
    notes_unique: pd.DataFrame,
    *,
    model_path: Optional[Union[str, Path]],
    notes_tag: str,
) -> pd.Series:
    from src.step5_target.quality_score import compute_quality_score, DEFAULT_MODEL_PATH
    path_obj = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if path_obj.exists():
        st = path_obj.stat()
        model_sig = (str(path_obj), st.st_size, st.st_mtime_ns)
    else:
        model_sig = "heuristic"

    tag = _hash({"notes": notes_tag, "model": model_sig})
    cache_path = CACHE_DIR / f"quality_{tag}.pkl.gz"
    df = _read(cache_path)
    if df is not None:
        print(f"  [cache HIT] quality ({len(df):,} notes)")
        s = df.set_index("noteId")["quality"]
        s.name = "quality"
        return s

    print(f"  [cache miss] quality; computing...")
    q = compute_quality_score(notes_unique, model_path=model_path)
    out = pd.DataFrame({"noteId": q.index, "quality": q.values})
    _write(out, cache_path)
    return q
