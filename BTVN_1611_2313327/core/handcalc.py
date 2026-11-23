"""Ví dụ giải tay: 18 sinh viên, 2 thuộc tính, số tròn để tự tính được trên giấy.

Mọi bước được cài đặt THỦ CÔNG (vòng lặp, công thức rõ ràng) để hiển thị từng con số,
sau đó đối chiếu với sklearn để chứng minh tính tay = máy tính.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATS = ["arrival_minutes", "wake_up_hour"]
VALID = {"arrival_minutes": (-90, 120), "wake_up_hour": (3, 12)}

# (mã, giờ đến so với giờ học (phút), giờ thức dậy, nhóm thật)
_ROWS = [
    ("E1", -18, 5.5, "early"), ("E2", -15, 5.5, "early"), ("E3", -12, 6.0, "early"),
    ("E4", -14, np.nan, "early"), ("E5", -10, 5.5, "early"),
    ("O1", -3, 6.0, "on_time"), ("O2", -2, 6.5, "on_time"), ("O3", -1, 6.0, "on_time"),
    ("O4", 0, 6.5, "on_time"), ("O5", 0, 6.0, "on_time"), ("O6", 1, 6.5, "on_time"),
    ("O7", 2, 6.0, "on_time"), ("O8", np.nan, 6.5, "on_time"), ("O9", -4, 25.0, "on_time"),
    ("O10", 999, 6.0, "on_time"),
    ("L1", 10, 7.0, "late"), ("L2", 15, 7.5, "late"), ("L3", 20, 7.0, "late"),
]


def tiny_dataset() -> pd.DataFrame:
    return pd.DataFrame(_ROWS, columns=["student_id", *FEATS, "_true_group"])


def apply_rules(df: pd.DataFrame):
    R = df.copy()
    log = []
    for c, (lo, hi) in VALID.items():
        for i in R.index[(R[c] < lo) | (R[c] > hi)]:
            log.append({"SV": R.at[i, "student_id"], "Thuộc tính": c, "Giá trị": R.at[i, c],
                        "Khoảng hợp lệ": f"[{lo}, {hi}]", "Xử lý": "→ NaN (coi như thiếu)"})
            R.at[i, c] = np.nan
    return R, pd.DataFrame(log)


def quantile_steps(values: pd.Series, q: float) -> dict:
    """Phân vị kiểu nội suy tuyến tính (mặc định của pandas/numpy): vị trí h = q·(n−1), đánh số từ 0."""
    s = np.sort(values.dropna().to_numpy())
    h = q * (len(s) - 1)
    lo = int(np.floor(h))
    frac = h - lo
    hi = min(lo + 1, len(s) - 1)
    val = s[lo] + frac * (s[hi] - s[lo])
    return {"sorted": s, "h": h, "lo": lo, "hi": hi, "frac": frac, "x_lo": s[lo], "x_hi": s[hi], "value": val}


def median_steps(values: pd.Series) -> dict:
    s = np.sort(values.dropna().to_numpy())
    n = len(s)
    if n % 2:
        return {"sorted": s, "n": n, "pos": [n // 2], "value": s[n // 2]}
    return {"sorted": s, "n": n, "pos": [n // 2 - 1, n // 2], "value": (s[n // 2 - 1] + s[n // 2]) / 2}


def zscore(df: pd.DataFrame):
    """z = (x − μ) / σ, σ là độ lệch chuẩn tổng thể (chia n) — giống StandardScaler."""
    mu = df[FEATS].mean()
    sigma = df[FEATS].std(ddof=0)
    return (df[FEATS] - mu) / sigma, mu, sigma


def kmeans_manual(Z: np.ndarray, init_idx, max_iter: int = 20):
    """Lloyd: (1) tính khoảng cách tới từng tâm, (2) gán vào tâm gần nhất, (3) tâm mới = trung bình cụm.
    Dừng khi phép gán không đổi."""
    C = Z[list(init_idx)].astype(float).copy()
    prev = None
    iters = []
    for it in range(1, max_iter + 1):
        D = np.sqrt(((Z[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))
        lab = D.argmin(axis=1)
        newC = np.array([Z[lab == k].mean(axis=0) if (lab == k).any() else C[k] for k in range(len(C))])
        sse = float(sum(((Z[lab == k] - C[k]) ** 2).sum() for k in range(len(C))))
        changed = None if prev is None else int((lab != prev).sum())
        iters.append({"iter": it, "centroids": C.copy(), "dist": D, "labels": lab.copy(), "new_centroids": newC,
                      "sse": sse, "changed": changed})
        if prev is not None and changed == 0:
            break
        prev, C = lab, newC
    return iters


def dbscan_manual(Z: np.ndarray, eps: float, min_pts: int):
    """N(i) = {j : d(i, j) ≤ eps} (tính cả chính i). Core nếu |N(i)| ≥ minPts.
    Duyệt điểm theo thứ tự; gặp core chưa có cụm thì mở cụm mới và loang qua các core láng giềng."""
    n = len(Z)
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))
    nbrs = [np.where(D[i] <= eps)[0] for i in range(n)]
    core = np.array([len(nb) >= min_pts for nb in nbrs])
    labels = np.full(n, -1)
    trace = []
    cid = -1
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        cid += 1
        labels[i] = cid
        queue, order = [i], []
        while queue:
            p = queue.pop(0)
            order.append(p)
            if not core[p]:
                continue
            for q in nbrs[p]:
                if labels[q] == -1:
                    labels[q] = cid
                    queue.append(q)
        trace.append((cid, order))
    kind = np.where(core, "core", np.where(labels >= 0, "border", "noise"))
    return D, nbrs, core, labels, kind, trace
