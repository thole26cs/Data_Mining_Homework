"""DBSCAN (bổ sung): pipeline đầy đủ + Grid Search CV + phát hiện ngoại lai + phân tích mật độ.

Xây dựng song song với SampledKMeans để so sánh công bằng: cùng bộ làm sạch, cùng cách
chọn mẫu, cùng bộ thuộc tính, cùng Stratified K-fold.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score, recall_score, silhouette_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler

from .data_gen import NUMERIC_COLS
from .model import FEATURE_SETS, SCALERS, draw_sample, kdistance_eps, y_codes
from .preprocess import VALID_RANGE, StudentCleaner, apply_rules, arrival_strata

DB_METRICS = ["F1_macro", "ARI", "NMI", "Late_recall", "Noise_pct", "N_clusters", "Silhouette"]
DB_GRID_DEFAULT = {"features": ["core", "behavior"], "scaler": ["standard", "robust"],
                   "min_samples": [5, 10, 20, 40], "eps_scale": [0.5, 0.75, 1.0, 1.5],
                   "sampling": ["stratified"], "frac": [1.0]}


def rule_group(mean_arrival: float) -> int:
    """Gán nghĩa cụm KHÔNG dùng nhãn: luật miền ±5 phút trên giờ đến trung bình của cụm."""
    if mean_arrival <= -5:
        return 0
    return 1 if mean_arrival <= 5 else 2


class SampledDBSCAN(BaseEstimator, ClusterMixin):
    """Làm sạch -> chọn mẫu -> chọn thuộc tính -> chuẩn hoá -> DBSCAN.

    eps = eps_scale × eps tại "khuỷu" đồ thị k-distance (tính trên mẫu train) để eps_scale so sánh
    được giữa các bộ thuộc tính/scaler; có thể truyền eps cố định.
    DBSCAN không có predict: điểm mới được gán theo core point gần nhất nếu khoảng cách ≤ eps,
    ngược lại là nhiễu (-1) — đúng định nghĩa "density-reachable".
    """

    def __init__(self, sampling="stratified", frac=1.0, features="core", scaler="standard",
                 eps_scale=1.0, min_samples=10, eps=None, impute="median", iqr_k=3.0, random_state=0):
        self.sampling = sampling
        self.frac = frac
        self.features = features
        self.scaler = scaler
        self.eps_scale = eps_scale
        self.min_samples = min_samples
        self.eps = eps
        self.impute = impute
        self.iqr_k = iqr_k
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        R = apply_rules(X)
        self.cleaner_ = StudentCleaner(iqr_k=self.iqr_k, impute=self.impute).fit_rules(R)
        Xc = self.cleaner_.transform_rules(R)
        self.sample_idx_ = draw_sample(arrival_strata(R["arrival_minutes"]), self.sampling, self.frac, rng)
        self.features_ = FEATURE_SETS[self.features]
        S = Xc.iloc[self.sample_idx_][self.features_]
        self.scaler_ = SCALERS[self.scaler]().fit(S)
        Z = self.scaler_.transform(S)
        self.knee_eps_, _ = kdistance_eps(Z, self.min_samples)
        self.eps_ = float(self.eps) if self.eps else self.knee_eps_ * self.eps_scale
        t0 = time.perf_counter()
        db = DBSCAN(eps=self.eps_, min_samples=self.min_samples).fit(Z)
        self.dbscan_time_ = time.perf_counter() - t0
        self.labels_ = db.labels_
        core = db.core_sample_indices_
        self.core_points_, self.core_labels_ = Z[core], db.labels_[core]
        self.nn_ = NearestNeighbors(n_neighbors=1).fit(self.core_points_) if len(core) else None
        clusters = sorted(int(c) for c in np.unique(self.labels_) if c != -1)
        self.n_clusters_ = len(clusters)
        arr = Xc.iloc[self.sample_idx_]["arrival_minutes"].to_numpy()
        self.cluster_to_group_ = {c: rule_group(arr[self.labels_ == c].mean()) for c in clusters}
        self.cluster_to_group_[-1] = -1
        return self

    def transform_features(self, X):
        return self.scaler_.transform(self.cleaner_.transform(X)[self.features_])

    def predict(self, X):
        Z = self.transform_features(X)
        if self.nn_ is None:
            return np.full(len(Z), -1)
        d, i = self.nn_.kneighbors(Z)
        lab = self.core_labels_[i[:, 0]].copy()
        lab[d[:, 0] > self.eps_] = -1
        return lab

    def predict_group(self, X):
        return np.array([self.cluster_to_group_.get(int(c), -1) for c in self.predict(X)])


def db_scorer(est, X, y):
    """Điểm nhiễu (-1) bị tính là SAI khi tính F1/recall (không thuộc nhóm nào)."""
    Z = est.transform_features(X)
    lab = est.predict(X)
    grp = np.array([est.cluster_to_group_.get(int(c), -1) for c in lab])
    yc = y_codes(y)
    m = lab != -1
    sil = (silhouette_score(Z[m], lab[m], sample_size=min(1000, int(m.sum())), random_state=0)
           if len(np.unique(lab[m])) > 1 else -1.0)
    return {"F1_macro": f1_score(yc, grp, average="macro", labels=[0, 1, 2], zero_division=0),
            "ARI": adjusted_rand_score(yc, lab), "NMI": normalized_mutual_info_score(yc, lab),
            "Late_recall": recall_score(yc, grp, labels=[2], average="macro", zero_division=0),
            "Noise_pct": 100 * float((~m).mean()), "N_clusters": float(len(np.unique(lab[m]))),
            "Silhouette": sil}


def run_dbscan_grid(X, y, strata, grid: dict, refit="F1_macro", n_splits=5, seed=42):
    cv = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata))
    gs = GridSearchCV(SampledDBSCAN(random_state=seed), grid, scoring=db_scorer, refit=refit, cv=cv,
                      n_jobs=1, error_score="raise")
    t0 = time.perf_counter()
    gs.fit(X, y)
    gs.metric_ = refit
    return gs, time.perf_counter() - t0


def db_results_frame(gs) -> pd.DataFrame:
    r = pd.DataFrame(gs.cv_results_)
    out = r[[c for c in r.columns if c.startswith("param_")]].rename(columns=lambda c: c.replace("param_", ""))
    for m in DB_METRICS:
        out[m] = r[f"mean_test_{m}"]
        out[f"{m}_std"] = r[f"std_test_{m}"]
    out["fit_time_s"] = r["mean_fit_time"]
    out["chosen"] = False
    out.loc[gs.best_index_, "chosen"] = True
    return out.sort_values(gs.metric_, ascending=False).reset_index(drop=True)


def density_by_group(Z, y, k=10) -> pd.DataFrame:
    """Khoảng cách tới láng giềng thứ k của mỗi điểm, theo nhóm thật: lớn = vùng thưa."""
    d, _ = NearestNeighbors(n_neighbors=k + 1).fit(Z).kneighbors(Z)
    return pd.DataFrame({"group": np.asarray(y), "kdist": d[:, -1]})


def invalid_rows(X) -> np.ndarray:
    """Dòng có ít nhất 1 giá trị vô lý / sai đơn vị trên dữ liệu thô (ground truth cho bài toán phát hiện ngoại lai)."""
    bad = np.zeros(len(X), dtype=bool)
    for c, (lo, hi) in VALID_RANGE.items():
        v = X[c].to_numpy(dtype=float)
        bad |= (v < lo) | (v > hi)
    return bad


def outlier_detection(X, y, quantile=0.85):
    """DBSCAN (điểm nhiễu) vs IQR×1.5 trong vai trò PHÁT HIỆN NGOẠI LAI trên dữ liệu thô chưa làm sạch.

    eps = phân vị `quantile` của k-distance (≈ chấp nhận tối đa 1-quantile dữ liệu là ngoại lai).
    Không dùng điểm khuỷu: các giá trị cực đoan (999 phút…) kéo dài đuôi k-distance làm khuỷu lệch rất xa.
    """
    truth = invalid_rows(X)
    late = np.asarray(y) == "late"
    raw = X[NUMERIC_COLS].astype(float)
    Z = RobustScaler().fit_transform(raw.fillna(raw.median()))
    ms = 2 * Z.shape[1]
    knee, kdist = kdistance_eps(Z, ms)
    eps = float(np.quantile(kdist, quantile))
    t0 = time.perf_counter()
    noise = DBSCAN(eps=eps, min_samples=ms).fit_predict(Z) == -1
    t_db = time.perf_counter() - t0
    iqr = np.zeros(len(X), dtype=bool)
    for c in NUMERIC_COLS:
        s = raw[c]
        q1, q3 = s.quantile([0.25, 0.75])
        iqr |= ((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).to_numpy()
    rows = []
    for name, flag in [(f"DBSCAN – điểm nhiễu (eps={eps:.2f}, minPts={ms})", noise), ("IQR×1.5 trên từng cột", iqr)]:
        tp = int((flag & truth).sum())
        rows.append({"Phương pháp": name, "Số dòng bị đánh dấu": int(flag.sum()),
                     "Precision": tp / max(1, int(flag.sum())), "Recall": tp / max(1, int(truth.sum())),
                     "SV trễ hợp lệ bị đánh dấu nhầm": int((flag & ~truth & late).sum())})
    return pd.DataFrame(rows), int(truth.sum()), t_db, knee, eps
