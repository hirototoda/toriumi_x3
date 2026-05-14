"""
バースト検出 + TypeA/B 分類 + 方向 (helpful / nothelp) 付与

src/simple/burst.py との差分:
  * detect_bursts_with_direction:
      burst_raters に加えて burst_levels (helpfulnessLevel の配列) と
      burst_direction ∈ {"helpful", "nothelp", "mixed"} を計算する。
      ロジック自体は simple.burst.detect_bursts と同一 (速度閾値・1 note 1 burst)。
  * classify_burst_type は src.simple.burst から reuse (TypeA/B 判定は共通)。

方向の定義 (厳密多数決):
  burst 内 min_count 件の helpfulnessLevel を集計し、
    n_helpful > n_nothelp        → "helpful"
    n_nothelp > n_helpful        → "nothelp"
    それ以外 (同数 / SOMEWHAT のみ) → "mixed"
  mixed は polarity 不明 burst と同様、後段で「バーストなし」扱い
  (= 回帰では 4 ダミーすべて 0) にするため、DataFrame から除外する。
"""

import numpy as np
import pandas as pd

# 分類ロジックは共通 (polarity 分散の中央値で A/B) → 既存 simple.burst を再利用
from src.simple.burst import classify_burst_type  # noqa: F401  (re-export for run script)


def detect_bursts_with_direction(
    ratings: pd.DataFrame,
    speed_multiplier: float = 3.0,
    min_count: int = 5,
) -> pd.DataFrame:
    """simple.burst.detect_bursts と同じ速度判定 + 方向ラベル付与."""
    rows = []
    for note_id, g in ratings.groupby("noteId"):
        g = g.sort_values("createdAtMillis")
        times  = g["createdAtMillis"].to_numpy()
        raters = g["raterParticipantId"].to_numpy()
        levels = g["helpfulnessLevel"].to_numpy()

        if len(times) < min_count:
            continue
        total_span = times[-1] - times[0]
        if total_span <= 0:
            continue
        avg_speed = len(times) / total_span

        for i in range(len(times) - min_count + 1):
            span = max(times[i + min_count - 1] - times[i], 1)
            local_speed = min_count / span
            if local_speed >= avg_speed * speed_multiplier:
                burst_levels = levels[i:i + min_count].tolist()
                n_help    = sum(1 for x in burst_levels if x == "HELPFUL")
                n_nothelp = sum(1 for x in burst_levels if x == "NOT_HELPFUL")
                if   n_help > n_nothelp:    direction = "helpful"
                elif n_nothelp > n_help:    direction = "nothelp"
                else:                       direction = "mixed"
                rows.append({
                    "noteId":          note_id,
                    "burst_start":     int(times[i]),
                    "burst_end":       int(times[i + min_count - 1]),
                    "burst_count":     min_count,
                    "burst_raters":    raters[i:i + min_count].tolist(),
                    "burst_levels":    burst_levels,
                    "burst_n_helpful": n_help,
                    "burst_n_nothelp": n_nothelp,
                    "burst_direction": direction,
                })
                break  # 1 ノート 1 バースト (simple 版と同じ簡略化)

    out = pd.DataFrame(rows)
    if out.empty:
        print("[burst-h1] no bursts detected")
        return out

    n_h = int((out["burst_direction"] == "helpful").sum())
    n_n = int((out["burst_direction"] == "nothelp").sum())
    n_m = int((out["burst_direction"] == "mixed").sum())
    print(f"[burst-h1] detected: {len(out):,} notes "
          f"(helpful={n_h}, nothelp={n_n}, mixed={n_m})")

    # mixed は分析から除外 (= その note は「バーストなし」扱い)
    before = len(out)
    out = out[out["burst_direction"] != "mixed"].reset_index(drop=True)
    if before != len(out):
        print(f"[burst-h1] dropped mixed bursts: {before - len(out)} "
              f"(残り {len(out):,})")
    return out
