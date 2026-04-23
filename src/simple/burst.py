"""
バースト検出 + TypeA/B 分類

定義:
  バースト = 連続する min_count 件の評価が「全体平均速度の speed_multiplier 倍」を超える区間
  1 ノート 1 バースト (簡略化)

TypeA / TypeB:
  バースト中の評価者の polarity 分散が小さい → A (陣営反応)
  大きい                                  → B (自然拡散)
  分散の閾値 = 全バースト分散の中央値 (相対基準)

polarity が分からない rater しか居ないバースト → 分類不能なので NaN にして
回帰側で dropna する (旧コードは強制 B にしていた; これはバイアスを生むので削除)
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
    """polarity 分散の中央値で TypeA / TypeB を分ける. polarity 不明は NaN."""
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

    valid = out["polarity_variance"].dropna()
    if valid.empty:
        out["burst_type"] = np.nan
        print("[burst] no polarity-resolvable bursts")
        return out

    threshold = valid.median()
    # NaN と文字列 ("A"/"B") を同居させるため dtype=object で組み立てる
    btype = pd.Series([pd.NA] * len(out), index=out.index, dtype=object)
    mask_valid = out["polarity_variance"].notna()
    btype.loc[mask_valid &  (out["polarity_variance"] <= threshold)] = "A"
    btype.loc[mask_valid &  (out["polarity_variance"] >  threshold)] = "B"
    out["burst_type"] = btype
    n_a = (out["burst_type"] == "A").sum()
    n_b = (out["burst_type"] == "B").sum()
    n_n = out["burst_type"].isna().sum()
    print(f"[burst] TypeA={n_a}, TypeB={n_b}, NaN={n_n} (threshold={threshold:.4f})")
    return out
