"""Sinh bộ dữ liệu giả lập: hành vi đến lớp của sinh viên.

Có 3 nhóm ẩn (ground truth) - đi sớm / đúng giờ / đi trễ - với tỉ lệ MẤT CÂN BẰNG.
Nhãn ẩn `_true_group` KHÔNG được đưa vào mô hình; chỉ dùng để kiểm chứng
xem mô hình gom cụm có "học" được đúng cấu trúc thật hay không.

Sau đó chèn "bẩn" có kiểm soát: thiếu (missing), nhiễu (noisy), ngoại lai
(outliers), sai đơn vị, sai chính tả nhãn phân loại, và bản ghi trùng.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

GROUPS = ["early", "on_time", "late"]
GROUP_VI = {"early": "Đi sớm", "on_time": "Đúng giờ", "late": "Đi trễ"}
TRANSPORTS = ["walk", "bike", "motorbike", "bus"]
MIN_PER_KM = {"walk": 12.0, "bike": 5.0, "motorbike": 2.5, "bus": 4.0}

NUMERIC_COLS = [
    "arrival_minutes", "arrival_std", "wake_up_hour", "distance_km",
    "commute_min", "sleep_hours", "gpa", "age",
]
COLUMNS = ["student_id", *NUMERIC_COLS, "transport"]

FEATURE_INFO = pd.DataFrame([
    ("student_id", "Mã sinh viên", "Định danh (nominal)", "Không dùng để gom cụm - chỉ để phát hiện trùng"),
    ("arrival_minutes", "Số phút đến lớp so với giờ bắt đầu, trung bình học kỳ (âm = đến sớm)", "Số - ratio", "Rất liên quan"),
    ("arrival_std", "Độ dao động giờ đến (phút)", "Số - ratio", "Liên quan (người hay trễ thì thất thường)"),
    ("wake_up_hour", "Giờ thức dậy (giờ, vd 6.25 = 6h15)", "Số - interval", "Liên quan"),
    ("distance_km", "Khoảng cách nhà - trường (km)", "Số - ratio", "Liên quan yếu"),
    ("commute_min", "Thời gian di chuyển (phút)", "Số - ratio", "DƯ THỪA - suy ra từ distance & phương tiện"),
    ("sleep_hours", "Số giờ ngủ trung bình", "Số - ratio", "Liên quan yếu"),
    ("gpa", "Điểm trung bình (thang 4)", "Số - ratio", "Gần như không liên quan"),
    ("age", "Tuổi", "Số - ratio", "Không liên quan"),
    ("transport", "Phương tiện: walk / bike / motorbike / bus", "Phân loại (nominal)", "Liên quan yếu"),
], columns=["Thuộc tính", "Ý nghĩa", "Kiểu (Chap 2)", "Vai trò thật (do ta thiết kế)"])

# Hồ sơ hành vi của từng nhóm ẩn: các phân phối chồng lấn nhau một phần
PROFILES = {
    "early": dict(arrival=(-15.0, 5.0), arrival_std=(3.0, 1.0), wake=(5.6, 0.35),
                  dist_median=2.5, sleep=(7.2, 0.8), transport=[0.35, 0.25, 0.30, 0.10], gpa_shift=0.1),
    "on_time": dict(arrival=(0.0, 2.5), arrival_std=(4.0, 1.2), wake=(6.15, 0.35),
                    dist_median=4.5, sleep=(6.8, 0.9), transport=[0.20, 0.20, 0.45, 0.15], gpa_shift=0.0),
    "late": dict(arrival=(14.0, 6.0), arrival_std=(8.0, 2.5), wake=(6.9, 0.45),
                 dist_median=7.5, sleep=(6.0, 1.0), transport=[0.05, 0.10, 0.35, 0.50], gpa_shift=-0.1),
}

TRANSPORT_VARIANTS = {
    "walk": ["Walk", " walk ", "đi bộ", "WALK"],
    "bike": ["Bike", "xe đạp", "bicycle "],
    "motorbike": ["Motorbike", "xe máy", "moto", "Motor bike"],
    "bus": ["Bus", "xe buýt", " BUS"],
}

OUTLIER_VALUES = {
    "arrival_minutes": [180.0, 240.0, -300.0, 999.0],   # quên quẹt thẻ, nhập sai
    "sleep_hours": [0.0, 22.0, -3.0],
    "wake_up_hour": [0.0, 25.0, -1.0],
    "age": [3.0, 150.0, 99.0],
    "distance_km": [-5.0],
    "gpa": [8.5, 9.2, -1.0],                            # nhập nhầm thang 10
    "commute_min": [600.0, 999.0],
}


def generate_clean(n: int = 2000, ratios=(0.18, 0.72, 0.10), seed: int = 42) -> pd.DataFrame:
    """Dữ liệu 'sạch' (ground truth) trước khi chèn lỗi."""
    rng = np.random.default_rng(seed)
    p = np.asarray(ratios, dtype=float)
    p = p / p.sum()
    groups = rng.choice(GROUPS, size=n, p=p)

    df = pd.DataFrame({"student_id": [f"SV{100000 + i}" for i in range(n)], "_true_group": groups})
    for c in NUMERIC_COLS:
        df[c] = np.nan
    df["transport"] = ""

    for g, prof in PROFILES.items():
        m = groups == g
        k = int(m.sum())
        if k == 0:
            continue
        df.loc[m, "arrival_minutes"] = rng.normal(*prof["arrival"], k)
        df.loc[m, "arrival_std"] = np.abs(rng.normal(*prof["arrival_std"], k)) + 0.5
        df.loc[m, "wake_up_hour"] = rng.normal(*prof["wake"], k)
        df.loc[m, "distance_km"] = rng.lognormal(np.log(prof["dist_median"]), 0.5, k)
        df.loc[m, "sleep_hours"] = rng.normal(*prof["sleep"], k)
        df.loc[m, "transport"] = rng.choice(TRANSPORTS, size=k, p=prof["transport"])
        df.loc[m, "gpa"] = rng.normal(3.0 + prof["gpa_shift"], 0.45, k)

    walk = df["transport"] == "walk"
    df.loc[walk, "distance_km"] *= 0.35
    df["commute_min"] = df["distance_km"] * df["transport"].map(MIN_PER_KM) + 3 + rng.normal(0, 3, n)
    df["commute_min"] = df["commute_min"].clip(lower=1)
    df["gpa"] = df["gpa"].clip(0, 4)
    df["sleep_hours"] = df["sleep_hours"].clip(3, 11)
    df["age"] = rng.integers(18, 26, n).astype(float)

    df = df.round({"arrival_minutes": 1, "arrival_std": 2, "wake_up_hour": 2, "distance_km": 2,
                   "commute_min": 1, "sleep_hours": 2, "gpa": 2})
    return df[[*COLUMNS, "_true_group"]]


def inject_dirt(clean: pd.DataFrame, missing: float = 0.08, outlier: float = 0.02, noise: float = 0.10,
                dup: float = 0.03, typo: float = 0.15, unit: float = 0.03, seed: int = 42):
    """Chèn lỗi có kiểm soát. Trả về (dữ liệu bẩn, nhật ký số lượng lỗi đã chèn)."""
    rng = np.random.default_rng(seed + 1)
    df = clean.copy()
    n = len(df)
    log: dict[str, int] = {}

    # 1) Nhiễu ngẫu nhiên (random error) trên các thuộc tính đo đạc
    noisy_total = 0
    for c in ["arrival_minutes", "wake_up_hour", "distance_km", "sleep_hours", "commute_min"]:
        idx = rng.random(n) < noise
        df.loc[idx, c] = df.loc[idx, c] + rng.normal(0, clean[c].std() * 0.5, idx.sum())
        noisy_total += int(idx.sum())
    log["Ô bị nhiễu (noise)"] = noisy_total

    # 2) Không nhất quán khi nhập nhãn phân loại
    tcol = df.columns.get_loc("transport")
    idx = np.where(rng.random(n) < typo)[0]
    for i in idx:
        df.iat[i, tcol] = rng.choice(TRANSPORT_VARIANTS[df.iat[i, tcol]])
    log["Nhãn 'transport' viết không nhất quán"] = len(idx)

    # 3) Sai đơn vị: khoảng cách ghi bằng mét thay vì km
    idx = rng.random(n) < unit
    df.loc[idx, "distance_km"] = (df.loc[idx, "distance_km"] * 1000).round(0)
    log["distance ghi bằng mét (sai đơn vị)"] = int(idx.sum())

    # 4) Ngoại lai / giá trị vô lý
    out_total = 0
    for c, values in OUTLIER_VALUES.items():
        idx = np.where(rng.random(n) < outlier)[0]
        df.iloc[idx, df.columns.get_loc(c)] = rng.choice(values, size=len(idx))
        out_total += len(idx)
    log["Ô ngoại lai / vô lý (outliers)"] = out_total

    # 5) Thiếu dữ liệu: MCAR + một phần MNAR (người trễ hay bỏ trống giờ dậy)
    miss_total = 0
    rates = {"arrival_minutes": 0.6, "arrival_std": 1.0, "wake_up_hour": 1.0, "distance_km": 0.8,
             "commute_min": 1.0, "sleep_hours": 1.2, "gpa": 1.0, "transport": 0.5}
    for c, r in rates.items():
        idx = rng.random(n) < missing * r
        df.loc[idx, c] = None if c == "transport" else np.nan
        miss_total += int(idx.sum())
    late = (df["_true_group"] == "late").to_numpy() & (rng.random(n) < missing)
    df.loc[late, "wake_up_hour"] = np.nan
    miss_total += int(late.sum())
    log["Ô bị thiếu (missing, gồm MNAR)"] = miss_total

    # 6) Bản ghi trùng (nhập 2 lần)
    k = int(round(dup * n))
    dups = df.sample(k, random_state=seed) if k > 0 else df.iloc[:0]
    df = pd.concat([df, dups], ignore_index=True)
    log["Bản ghi trùng"] = k

    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df, log


def generate(n=2000, ratios=(0.18, 0.72, 0.10), seed=42, **dirt):
    clean = generate_clean(n, ratios, seed)
    raw, log = inject_dirt(clean, seed=seed, **dirt)
    return raw, clean, log
