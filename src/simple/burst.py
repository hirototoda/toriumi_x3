"""
バースト検出 + TypeA/B 分類

定義:
  バースト = 連続する min_count 件の評価が「全体平均速度の speed_multiplier 倍」を超える区間
  1 ノート 1 バースト (簡略化)

TypeA / TypeB:
  バースト中の評価者の polarity 分散が小さい → A (陣営反応)
  大きい                                  → B (自然拡散)
  分散の閾値 = 全バースト分散の中央値 (相対基準)

polarity が分からない rater しか居ないバースト = 分類不能。
この場合は bursts DataFrame から行ごと除外する (= そのノートは「バーストなし」
扱いになり、回帰の type_a / type_b が両方 0 になる)。
旧実装は強制 B にしていたがバイアスを生むため不採用。
NaN burst_type を残す案も試したが回帰側で NA 比較の TypeError を引き起こすので不採用。
「除外」が一番副作用が無く、簡潔。
"""

import numpy as np
import pandas as pd


def detect_bursts(
    ratings: pd.DataFrame,
    speed_multiplier: float = 3.0,
    min_count: int = 5,
) -> pd.DataFrame:
    """各ノートで最初に見つかったバーストを 1 行返す"""
    rows = []
    for note_id, g in ratings.groupby("noteId"):
        g = g.sort_values("createdAtMillis")
        times  = g["createdAtMillis"].to_numpy()
        raters = g["raterParticipantId"].to_numpy()

        if len(times) < min_count:
            continue
        total_span = times[-1] - times[0]
        if total_span <= 0:
            continue
        avg_speed = len(times) / total_span

        for i in range(len(times) - min_count + 1):
            span = max(times[i + min_count - 1] - times[i], 1)  # 同時刻なら 1ms とみなす
            local_speed = min_count / span
            if local_speed >= avg_speed * speed_multiplier:
                rows.append({
                    "noteId": note_id,
                    "burst_start":  int(times[i]),
                    "burst_end":    int(times[i + min_count - 1]),
                    "burst_count":  min_count,
                    "burst_raters": raters[i:i + min_count].tolist(),
                })
                break  # 1 ノート 1 バーストに簡略化

    out = pd.DataFrame(rows)
    print(f"[burst] detected: {len(out):,} notes")
    return out


def classify_burst_type(bursts: pd.DataFrame, polarity: pd.DataFrame) -> pd.DataFrame:
    """polarity 分散の中央値で TypeA / TypeB を分ける.
    polarity 不明 (分散 NaN) のバーストは行ごと除外して返す.
    """
    if bursts.empty:
        out = bursts.copy()
        out["polarity_variance"] = pd.Series(dtype=float)
        out["burst_type"] = pd.Series(dtype=str)
        return out

    px_map = polarity.set_index("raterParticipantId")["polarity_x"]
    py_map = polarity.set_index("raterParticipantId")["polarity_y"]

    variances = []
    for raters in bursts["burst_raters"]:
        px = [px_map.get(r) for r in raters]
        py = [py_map.get(r) for r in raters]
        px = [v for v in px if v is not None and not np.isnan(v)]
        py = [v for v in py if v is not None and not np.isnan(v)]
        variances.append(np.var(px) + np.var(py) if len(px) >= 2 else np.nan)

    out = bursts.copy()
    out["polarity_variance"] = variances

    n_dropped = int(out["polarity_variance"].isna().sum())
    out = out.dropna(subset=["polarity_variance"]).reset_index(drop=True)
    if out.empty:
        out["burst_type"] = pd.Series(dtype=str)
        print(f"[burst] no polarity-resolvable bursts (dropped {n_dropped})")
        return out

    threshold = out["polarity_variance"].median()
    out["burst_type"] = np.where(out["polarity_variance"] <= threshold, "A", "B")
    n_a = int((out["burst_type"] == "A").sum())
    n_b = int((out["burst_type"] == "B").sum())
    print(f"[burst] TypeA={n_a}, TypeB={n_b}, dropped(unresolvable)={n_dropped} (threshold={threshold:.4f})")
    return out
