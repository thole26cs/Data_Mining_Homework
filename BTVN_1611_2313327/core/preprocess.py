"""Tiền xử lý: làm sạch, chuẩn hoá, xử lý ngoại lai, điền khuyết, mã hoá.

`StudentCleaner` là một transformer của sklearn: các tham số (median, ngưỡng IQR,
mode...) được HỌC trên tập train và chỉ ÁP DỤNG lên tập validation/test,
tránh rò rỉ dữ liệu (data leakage). Nó được lưu cùng mô hình trong learnt.pkl
để dữ liệu mới đi qua đúng các bước giống hệt lúc huấn luyện.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer

from .data_gen import NUMERIC_COLS, TRANSPORTS

TRANSPORT_CANON = {
    "walk": "walk", "đi bộ": "walk",
    "bike": "bike", "xe đạp": "bike", "bicycle": "bike",
    "motorbike": "motorbike", "xe máy": "motorbike", "moto": "motorbike", "motor bike": "motorbike",
    "bus": "bus", "xe buýt": "bus",
}

# Luật miền (domain knowledge): giá trị ngoài khoảng này là vô lý -> coi như thiếu
VALID_RANGE = {
    "arrival_minutes": (-90, 120), "arrival_std": (0, 60), "wake_up_hour": (3, 12),
    "distance_km": (0, 60), "commute_min": (0, 240), "sleep_hours": (2, 14),
    "gpa": (0, 4), "age": (16, 45),
}

# KHÔNG đưa arrival_minutes vào IQR: đuôi phải của nó chính là nhóm thiểu số "đi trễ"
IQR_COLS_DEFAULT = ("arrival_std", "distance_km", "commute_min", "sleep_hours", "gpa")

UNKNOWN = "chưa rõ"
STRATA_ORDER = ["sớm (≤ -5')", "đúng giờ (-5'..5')", "trễ (> 5')", UNKNOWN]


def deduplicate(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = df.drop_duplicates(subset="student_id", keep="first").reset_index(drop=True)
    return out, len(df) - len(out)


def normalize_transport(s: pd.Series) -> pd.Series:
    return s.astype(object).str.strip().str.lower().map(TRANSPORT_CANON)


def fix_units(X: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    X = X.copy()
    m = X["distance_km"] > VALID_RANGE["distance_km"][1]
    X.loc[m, "distance_km"] = X.loc[m, "distance_km"] / 1000
    return X, int(m.sum())


def apply_rules(X: pd.DataFrame, return_report: bool = False):
    """Các bước làm sạch không cần học tham số: chuẩn hoá nhãn, sửa đơn vị, luật miền."""
    report = {}
    X = X.copy()
    if return_report:
        report["transport_variants_before"] = X["transport"].nunique(dropna=True)
    X["transport"] = normalize_transport(X["transport"])
    d = X["distance_km"].to_numpy(dtype=float, copy=True)
    m = d > VALID_RANGE["distance_km"][1]
    d[m] /= 1000
    X["distance_km"] = d
    invalid = {}
    for c, (lo, hi) in VALID_RANGE.items():
        v = X[c].to_numpy(dtype=float, copy=True)
        bad = (v < lo) | (v > hi)
        invalid[c] = int(bad.sum())
        v[bad] = np.nan
        X[c] = v
    if not return_report:
        return X
    report["transport_variants_after"] = X["transport"].nunique(dropna=True)
    report["unit_fixed"] = int(m.sum())
    report["invalid_to_nan"] = invalid
    return X, report


def arrival_strata(arrival: pd.Series) -> np.ndarray:
    """Tầng (strata) dựa trên thuộc tính quan sát được - KHÔNG dùng nhãn ẩn."""
    s = pd.cut(arrival, bins=[-np.inf, -5, 5, np.inf], labels=STRATA_ORDER[:3]).astype(object)
    return s.where(arrival.notna(), UNKNOWN).to_numpy()


def iqr_bounds(s: pd.Series, k: float) -> tuple[float, float]:
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - k * (q3 - q1), q3 + k * (q3 - q1)


class StudentCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, iqr_k: float = 3.0, iqr_cols: tuple = IQR_COLS_DEFAULT, impute: str = "median"):
        self.iqr_k = iqr_k
        self.iqr_cols = iqr_cols
        self.impute = impute

    def _clip(self, R: pd.DataFrame) -> pd.DataFrame:
        for c, (lo, hi) in self.bounds_.items():
            R[c] = R[c].clip(lo, hi)
        return R

    def fit(self, X, y=None):
        return self.fit_rules(apply_rules(X))

    def transform(self, X):
        return self.transform_rules(apply_rules(X))

    def fit_rules(self, R: pd.DataFrame):
        """Học tham số từ dữ liệu ĐÃ qua apply_rules (tránh tính lặp)."""
        R = R.copy()
        self.bounds_ = {c: iqr_bounds(R[c].dropna(), self.iqr_k) for c in self.iqr_cols}
        R = self._clip(R)
        self.medians_ = R[NUMERIC_COLS].median()
        self.means_ = R[NUMERIC_COLS].mean()
        mode = R["transport"].mode()
        self.mode_ = mode.iat[0] if len(mode) else TRANSPORTS[0]
        if self.impute == "knn":
            self.center_ = self.medians_
            self.scale_ = R[NUMERIC_COLS].std().replace(0, 1)
            self.knn_ = KNNImputer(n_neighbors=5).fit(((R[NUMERIC_COLS] - self.center_) / self.scale_).to_numpy())
        return self

    def transform_rules(self, R: pd.DataFrame):
        R = self._clip(R.copy())
        if self.impute == "knn":
            Z = ((R[NUMERIC_COLS] - self.center_) / self.scale_).to_numpy()
            R[NUMERIC_COLS] = self.knn_.transform(Z) * self.scale_.to_numpy() + self.center_.to_numpy()
        elif self.impute == "mean":
            R[NUMERIC_COLS] = R[NUMERIC_COLS].fillna(self.means_)
        else:
            R[NUMERIC_COLS] = R[NUMERIC_COLS].fillna(self.medians_)
        R["transport"] = R["transport"].fillna(self.mode_)
        for t in TRANSPORTS:
            R[f"tr_{t}"] = (R["transport"] == t).astype(float)
        return R


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in [c for c in df.columns if not c.startswith("_")]:
        s = df[c]
        row = {"Thuộc tính": c, "Thiếu": int(s.isna().sum()), "% thiếu": round(100 * s.isna().mean(), 1),
               "Số giá trị khác nhau": int(s.nunique(dropna=True))}
        if c in VALID_RANGE:
            lo, hi = VALID_RANGE[c]
            row["Vô lý (ngoài miền)"] = int(((s < lo) | (s > hi)).sum())
            lo2, hi2 = iqr_bounds(s.dropna(), 1.5)
            row["Ngoại lai IQR×1.5"] = int(((s < lo2) | (s > hi2)).sum())
            row["min"], row["max"] = round(s.min(), 2), round(s.max(), 2)
        rows.append(row)
    return pd.DataFrame(rows)
