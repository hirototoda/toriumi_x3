"""
Step 3: バースト検出

各ノートの評価タイムラインを分析し、バースト（急激な評価集中）を検出する。

バーストの定義（相対定義）:
- 各ノートの平均評価速度の3倍以上の速度
- かつ 5件以上の評価が集中している区間
"""

import numpy as np
import pandas as pd


def detect_bursts(
    df: pd.DataFrame,
    speed_multiplier: float = 3.0,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    各ノートの評価タイムラインからバースト区間を検出する。

    Returns
    -------
    pd.DataFrame
        columns: noteId, burst_start, burst_end, burst_count, burst_speed
        + バースト内の各評価者 raterParticipantId のリスト(burst_raters)
    """
    results = []

    for note_id, group in df.groupby("noteId"):
        group = group.sort_values("createdAtMillis")
        times = group["createdAtMillis"].values
        raters = group["raterParticipantId"].values

        if len(times) < min_count:
            continue

        # 全体の平均速度（件/ミリ秒）
        total_span = times[-1] - times[0]
        if total_span <= 0:
            continue
        avg_speed = len(times) / total_span

        # スライディングウィンドウでバースト検出
        window = min_count
        for i in range(len(times) - window + 1):
            span = times[i + window - 1] - times[i]
            if span <= 0:
                # 同一ミリ秒 → 無限速度 → バースト
                span = 1
            local_speed = window / span

            if local_speed >= avg_speed * speed_multiplier:
                # バーストの末端を拡張
                end = i + window - 1
                while end + 1 < len(times):
                    next_span = times[end + 1] - times[i]
                    if next_span <= 0:
                        next_span = 1
                    if (end + 2 - i) / next_span >= avg_speed * speed_multiplier:
                        end += 1
                    else:
                        break

                results.append({
                    "noteId": note_id,
                    "burst_start": int(times[i]),
                    "burst_end": int(times[end]),
                    "burst_count": end - i + 1,
                    "burst_speed": (end - i + 1) / max(times[end] - times[i], 1),
                    "burst_raters": list(raters[i:end + 1]),
                })
                break  # 1ノート1バーストに簡略化

    burst_df = pd.DataFrame(results)
    print(f"    bursts detected: {len(burst_df):,} notes")
    return burst_df
