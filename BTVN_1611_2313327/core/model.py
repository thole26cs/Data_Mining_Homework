"""Mô hình: chọn mẫu + trích chọn thuộc tính + K-Means (k=3), Grid Search CV,
so sánh CV vs Bootstrap, so sánh thuật toán, đánh giá độ ổn định.
"""
from __future__ import annotations

import datetime as _dt
import pickle
import time
from functools import partial

import numpy as np
import pandas as pd
import sklearn
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp, spearmanr
from sklearn.base import BaseEstimator, ClusterMixin, clone
from sklearn.cluster import DBSCAN, HDBSCAN, AgglomerativeClustering, KMeans, MiniBatchKMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
                             f1_score, normalized_mutual_info_score, precision_score, recall_score,
                             silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from .data_gen import GROUP_VI, GROUPS, generate_clean
from .preprocess import UNKNOWN, StudentCleaner, apply_rules, arrival_strata

GROUP_CODE = {g: i for i, g in enumerate(GROUPS)}

FEATURE_SETS = {
    "core": ["arrival_minutes", "arrival_std", "wake_up_hour"],
    "behavior": ["arrival_minutes", "arrival_std", "wake_up_hour", "distance_km", "sleep_hours"],
    "all": ["arrival_minutes", "arrival_std", "wake_up_hour", "distance_km", "sleep_hours",
            "commute_min", "gpa", "age", "tr_walk", "tr_bike", "tr_motorbike", "tr_bus"],
}
FEATURE_SET_VI = {
    "core": "core - 3 thuộc tính giờ giấc",
    "behavior": "behavior - core + khoảng cách, giấc ngủ",
    "all": "all - mọi thuộc tính (kể cả dư thừa/không liên quan)",
}
SCALERS = {"standard": StandardScaler, "robust": RobustScaler, "minmax": MinMaxScaler}
SAMPLING_VI = {
    "random": "Ngẫu nhiên không hoàn lại (SRSWOR)",
    "stratified": "Phân tầng theo giờ đến",
    "balanced": "Cân bằng tầng (over/under-sampling)",
    "bootstrap": "Bootstrap (SRSWR - có hoàn lại)",
}


def draw_sample(strata: np.ndarray, method: str, frac: float, rng: np.random.Generator) -> np.ndarray:
    n = len(strata)
    m = max(10, int(round(frac * n)))
    if method == "random":
        return np.arange(n) if m >= n else rng.choice(n, m, replace=False)
    if method == "bootstrap":
        return rng.choice(n, m, replace=True)
    groups = {s: np.where(strata == s)[0] for s in np.unique(strata)}
    out = []
    if method == "stratified":
        for ix in groups.values():
            k = min(len(ix), max(1, int(round(frac * len(ix)))))
            out.append(rng.choice(ix, k, replace=False))
    elif method == "balanced":
        known = {s: ix for s, ix in groups.items() if s != UNKNOWN} or groups
        quota = max(1, m // len(known))
        for ix in known.values():
            out.append(rng.choice(ix, quota, replace=len(ix) < quota))
    else:
        raise ValueError(f"Unknown sampling method: {method}")
    return np.concatenate(out)


class SampledKMeans(BaseEstimator, ClusterMixin):
    """Pipeline đầy đủ: làm sạch -> chọn mẫu -> chọn thuộc tính -> chuẩn hoá -> K-Means.

    Không dùng nhãn ẩn y trong fit (y chỉ để GridSearchCV truyền cho hàm chấm điểm).
    """

    def __init__(self, sampling="stratified", frac=1.0, features="behavior", scaler="standard",
                 n_clusters=3, impute="median", iqr_k=3.0, random_state=0):
        self.sampling = sampling
        self.frac = frac
        self.features = features
        self.scaler = scaler
        self.n_clusters = n_clusters
        self.impute = impute
        self.iqr_k = iqr_k
        self.random_state = random_state

    def fit(self, X, y=None):
        rng = np.random.default_rng(self.random_state)
        R = apply_rules(X)
        self.cleaner_ = StudentCleaner(iqr_k=self.iqr_k, impute=self.impute).fit_rules(R)
        Xc = self.cleaner_.transform_rules(R)
        strata = arrival_strata(R["arrival_minutes"])
        self.sample_idx_ = draw_sample(strata, self.sampling, self.frac, rng)
        self.features_ = FEATURE_SETS[self.features]
        S = Xc.iloc[self.sample_idx_][self.features_]
        self.scaler_ = SCALERS[self.scaler]().fit(S)
        self.kmeans_ = KMeans(self.n_clusters, n_init=5, random_state=self.random_state)
        lab = self.kmeans_.fit_predict(self.scaler_.transform(S))
        # Gán nghĩa cho cụm KHÔNG dùng nhãn: sắp theo giờ đến trung bình (sớm -> trễ)
        arr = Xc.iloc[self.sample_idx_]["arrival_minutes"].to_numpy()
        means = np.array([arr[lab == c].mean() if (lab == c).any() else np.inf for c in range(self.n_clusters)])
        rank = np.argsort(np.argsort(means))
        self.cluster_to_group_ = {c: int(min(r, len(GROUPS) - 1)) for c, r in enumerate(rank)}
        return self

    def transform_features(self, X):
        return self.scaler_.transform(self.cleaner_.transform(X)[self.features_])

    def predict(self, X):
        return self.kmeans_.predict(self.transform_features(X))

    def predict_group(self, X):
        return np.vectorize(self.cluster_to_group_.get)(self.predict(X))

    def centroids_original(self) -> pd.DataFrame:
        C = pd.DataFrame(self.scaler_.inverse_transform(self.kmeans_.cluster_centers_), columns=self.features_)
        C.insert(0, "Nhóm", [GROUP_VI[GROUPS[self.cluster_to_group_[c]]] for c in range(self.n_clusters)])
        C.insert(0, "Cụm", range(self.n_clusters))
        return C.sort_values("arrival_minutes").reset_index(drop=True)


# ---------------------------------------------------------------- scorers (estimator, X, y)
def y_codes(y) -> np.ndarray:
    return pd.Series(np.asarray(y)).map(GROUP_CODE).to_numpy()


def score_f1(est, X, y):
    return f1_score(y_codes(y), est.predict_group(X), average="macro", labels=[0, 1, 2], zero_division=0)


def score_ari(est, X, y):
    return adjusted_rand_score(y, est.predict(X))


def score_nmi(est, X, y):
    return normalized_mutual_info_score(y, est.predict(X))


def score_late_recall(est, X, y):
    return recall_score(y_codes(y), est.predict_group(X), labels=[2], average="macro", zero_division=0)


def score_silhouette(est, X, y):
    Z, lab = est.transform_features(X), est.predict(X)
    if len(np.unique(lab)) < 2:
        return -1.0
    return silhouette_score(Z, lab, sample_size=min(1000, len(Z)), random_state=0)


SCORERS = {"F1_macro": score_f1, "ARI": score_ari, "NMI": score_nmi,
           "Late_recall": score_late_recall, "Silhouette": score_silhouette}


def multi_scorer(est, X, y):
    """Một lần làm sạch + dự đoán, trả về mọi chỉ số (nhanh hơn 5 scorer riêng)."""
    Z = est.transform_features(X)
    lab = est.kmeans_.predict(Z)
    grp = np.vectorize(est.cluster_to_group_.get)(lab)
    yc = y_codes(y)
    sil = silhouette_score(Z, lab, sample_size=min(1000, len(Z)), random_state=0) if len(np.unique(lab)) > 1 else -1.0
    return {"F1_macro": f1_score(yc, grp, average="macro", labels=[0, 1, 2], zero_division=0),
            "ARI": adjusted_rand_score(yc, lab), "NMI": normalized_mutual_info_score(yc, lab),
            "Late_recall": recall_score(yc, grp, labels=[2], average="macro", zero_division=0),
            "Silhouette": sil}


SAMPLING_SIMPLICITY = {"stratified": 0, "random": 1, "balanced": 2, "bootstrap": 3}


def simplicity_key(p: dict):
    return (len(FEATURE_SETS[p["features"]]), SAMPLING_SIMPLICITY.get(p["sampling"], 9),
            p["frac"], p["scaler"] != "standard", p.get("impute", "median") != "median")


def one_se_refit(cv_results, metric="F1_macro"):
    """Quy tắc 1-SE: trong các cấu hình có điểm >= (tốt nhất - 1 sai số chuẩn), chọn cấu hình
    ĐƠN GIẢN nhất (ít thuộc tính, mẫu phân tầng, mẫu nhỏ). Tránh 'thắng' nhờ may rủi của CV."""
    mean = np.asarray(cv_results[f"mean_test_{metric}"])
    std = np.asarray(cv_results[f"std_test_{metric}"])
    k = sum(1 for key in cv_results if key.startswith("split") and key.endswith(f"_test_{metric}"))
    b = int(np.nanargmax(mean))
    thr = mean[b] - std[b] / np.sqrt(max(k, 1))
    ok = [i for i in range(len(mean)) if mean[i] >= thr]
    return int(min(ok, key=lambda i: simplicity_key(cv_results["params"][i])))


def run_grid_search(X, y, strata, grid: dict, refit="F1_macro", n_splits=5, seed=42, one_se=True):
    cv = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, strata))
    refit_arg = partial(one_se_refit, metric=refit) if one_se else refit
    gs = GridSearchCV(SampledKMeans(random_state=seed), grid, scoring=multi_scorer, refit=refit_arg, cv=cv,
                      n_jobs=1, error_score="raise")
    t0 = time.perf_counter()
    gs.fit(X, y)
    gs.metric_ = refit
    gs.argmax_index_ = int(np.nanargmax(gs.cv_results_[f"mean_test_{refit}"]))
    return gs, time.perf_counter() - t0


def best_cv_score(gs) -> float:
    return float(gs.cv_results_[f"mean_test_{gs.metric_}"][gs.best_index_])


def cv_results_frame(gs) -> pd.DataFrame:
    r = pd.DataFrame(gs.cv_results_)
    keep = [c for c in r.columns if c.startswith("param_")]
    out = r[keep].rename(columns=lambda c: c.replace("param_", ""))
    for m in SCORERS:
        out[f"{m}"] = r[f"mean_test_{m}"]
        out[f"{m}_std"] = r[f"std_test_{m}"]
    out["fit_time_s"] = r["mean_fit_time"]
    out["chosen"] = False
    out.loc[gs.best_index_, "chosen"] = True
    return out.sort_values(gs.metric_, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- representativeness
def representativeness(X, y, frac, reps=100, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    R = apply_rules(X)
    strata = arrival_strata(R["arrival_minutes"])
    arr = R["arrival_minutes"].to_numpy()
    pop = arr[~np.isnan(arr)]
    is_late = np.asarray(y) == "late"
    rows = []
    for method in SAMPLING_VI:
        for _ in range(reps):
            idx = draw_sample(strata, method, frac, rng)
            a = arr[idx]
            rows.append({"method": method,
                         "late_share_true": is_late[idx].mean(),
                         "late_share_strata": (strata[idx] == "trễ (> 5')").mean(),
                         "ks": ks_2samp(a[~np.isnan(a)], pop).statistic,
                         "unique_ratio": len(np.unique(idx)) / len(idx)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- CV vs Bootstrap
def cv_vs_bootstrap(param_list, X_tr, y_tr, strata_tr, X_te, y_te, metric="F1_macro",
                    n_splits=5, B=30, seed=42):
    score = SCORERS[metric]
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    cv = list(StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X_tr, strata_tr))
    rows = []
    for i, params in enumerate(param_list):
        est = SampledKMeans(**params, random_state=seed)
        t0 = time.perf_counter()
        cvs = [score(clone(est).fit(X_tr.iloc[tr]), X_tr.iloc[va], y_tr.iloc[va]) for tr, va in cv]
        t_cv = time.perf_counter() - t0

        t0 = time.perf_counter()
        bss = []
        for _ in range(B):
            idx = rng.integers(0, n, n)
            oob = np.setdiff1d(np.arange(n), idx)
            bss.append(score(clone(est).fit(X_tr.iloc[idx]), X_tr.iloc[oob], y_tr.iloc[oob]))
        t_bs = time.perf_counter() - t0

        full = clone(est).fit(X_tr)
        s_train = score(full, X_tr, y_tr)
        s_test = score(full, X_te, y_te)
        rows.append({"config": i + 1, **params,
                     "CV_mean": np.mean(cvs), "CV_std": np.std(cvs),
                     "Boot_OOB_mean": np.mean(bss), "Boot_OOB_std": np.std(bss),
                     "Boot_632": 0.368 * s_train + 0.632 * np.mean(bss),
                     "Test": s_test, "CV_time_s": t_cv, "Boot_time_s": t_bs})
    df = pd.DataFrame(rows)
    summary = {}
    for name, col, tcol in [("K-fold CV", "CV_mean", "CV_time_s"),
                            ("Bootstrap OOB", "Boot_OOB_mean", "Boot_time_s"),
                            ("Bootstrap .632", "Boot_632", "Boot_time_s")]:
        chosen = df[col].idxmax()
        summary[name] = {
            "Sai lệch TB so với test (MAE)": float((df[col] - df["Test"]).abs().mean()),
            "Độ lệch chuẩn ước lượng TB": float(df["CV_std" if col == "CV_mean" else "Boot_OOB_std"].mean()),
            "Spearman(xếp hạng, test)": float(spearmanr(df[col], df["Test"]).statistic) if len(df) > 2 else np.nan,
            "Test của cấu hình được chọn": float(df.loc[chosen, "Test"]),
            "Thời gian tổng (s)": float(df[tcol].sum()),
        }
    return df, pd.DataFrame(summary).T


# ---------------------------------------------------------------- algorithm comparison
def hungarian_map(y_true_codes, labels):
    clusters = [c for c in np.unique(labels) if c != -1]
    if not clusters:
        return np.full_like(labels, -1)
    M = np.array([[np.sum((y_true_codes == g) & (labels == c)) for c in clusters] for g in range(3)])
    r, c = linear_sum_assignment(-M)
    mapping = {clusters[j]: i for i, j in zip(r, c)}
    return np.array([mapping.get(l, -1) for l in labels])


def kdistance_eps(Z, min_samples):
    d, _ = NearestNeighbors(n_neighbors=min_samples).fit(Z).kneighbors(Z)
    k = np.sort(d[:, -1])
    x = np.linspace(0, 1, len(k))
    y = (k - k[0]) / (k[-1] - k[0] + 1e-12)
    return float(k[np.argmax(x - y)]), k


def _metrics(name, labels, Z, yc, y, t):
    mapped = hungarian_map(yc, labels)
    mask = labels != -1
    sil = silhouette_score(Z[mask], labels[mask], sample_size=min(2000, mask.sum()), random_state=0) \
        if len(np.unique(labels[mask])) > 1 else np.nan
    return {"Mô hình": name, "Thời gian (s)": t, "Số cụm": len(np.unique(labels[mask])),
            "% nhiễu": 100 * (~mask).mean(), "ARI": adjusted_rand_score(y, labels),
            "NMI": normalized_mutual_info_score(y, labels),
            "F1_macro": f1_score(yc, mapped, average="macro", labels=[0, 1, 2], zero_division=0),
            "Late_recall": recall_score(yc, mapped, labels=[2], average="macro", zero_division=0),
            "Silhouette": sil}


def compare_models(X_all, y_all, best_est, seed=42):
    """Mọi thuật toán dùng CÙNG tiền xử lý + bộ thuộc tính của cấu hình tốt nhất."""
    cleaner = StudentCleaner(iqr_k=best_est.iqr_k, impute=best_est.impute).fit(X_all)
    feats = FEATURE_SETS[best_est.features]
    Z = SCALERS[best_est.scaler]().fit_transform(cleaner.transform(X_all)[feats])
    y = np.asarray(y_all)
    yc = y_codes(y)
    rows, labels_all = [], {}
    d = Z.shape[1]
    ms = max(5, 2 * d)
    eps_auto, kdist = kdistance_eps(Z, ms)

    def run(name, fn):
        t0 = time.perf_counter()
        lab = np.asarray(fn())
        t = time.perf_counter() - t0
        rows.append(_metrics(name, lab, Z, yc, y, t))
        labels_all[name] = lab

    run("K-Means (cấu hình Grid Search)", lambda: clone(best_est).fit(X_all).predict(X_all))
    run("K-Means (toàn bộ dữ liệu)", lambda: KMeans(3, n_init=10, random_state=seed).fit_predict(Z))
    run("MiniBatch K-Means", lambda: MiniBatchKMeans(3, n_init=10, random_state=seed).fit_predict(Z))
    run("Gaussian Mixture (GMM)", lambda: GaussianMixture(3, n_init=3, random_state=seed).fit(Z).predict(Z))
    run("Agglomerative (Ward)", lambda: AgglomerativeClustering(3).fit_predict(Z))
    run(f"DBSCAN (eps tự động={eps_auto:.2f}, minPts={ms})", lambda: DBSCAN(eps=eps_auto, min_samples=ms).fit_predict(Z))

    # DBSCAN "được nhìn đáp án": dò eps/minPts theo ARI -> cận trên, lạc quan
    best = (-1, None, None)
    t0 = time.perf_counter()
    for eps in np.linspace(0.1, 1.5, 15):
        for m in (5, 10, 20, 40):
            lab = DBSCAN(eps=eps, min_samples=m).fit_predict(Z)
            a = adjusted_rand_score(y, lab)
            if a > best[0]:
                best = (a, eps, m)
    t_tune = time.perf_counter() - t0
    lab = DBSCAN(eps=best[1], min_samples=best[2]).fit_predict(Z)
    rows.append({**_metrics(f"DBSCAN dò theo nhãn (eps={best[1]:.2f}, minPts={best[2]})", lab, Z, yc, y, t_tune)})
    labels_all[rows[-1]["Mô hình"]] = lab
    run("HDBSCAN", lambda: HDBSCAN(min_cluster_size=max(15, len(Z) // 50)).fit_predict(Z))
    return pd.DataFrame(rows), labels_all, Z, kdist, eps_auto


def scaling_benchmark(sizes, ratios, features="behavior", seed=42):
    rows = []
    feats = [f for f in FEATURE_SETS[features] if not f.startswith("tr_")]
    for n in sizes:
        df = generate_clean(n, ratios, seed)
        Z = StandardScaler().fit_transform(df[feats])
        y = df["_true_group"].to_numpy()
        eps, _ = kdistance_eps(Z[: min(n, 3000)], 10)
        algos = {
            "K-Means": lambda: KMeans(3, n_init=3, random_state=seed).fit_predict(Z),
            "MiniBatch K-Means": lambda: MiniBatchKMeans(3, n_init=3, random_state=seed).fit_predict(Z),
            "GMM": lambda: GaussianMixture(3, random_state=seed).fit(Z).predict(Z),
            "DBSCAN": lambda: DBSCAN(eps=eps, min_samples=10).fit_predict(Z),
        }
        if n <= 6000:
            algos["Agglomerative"] = lambda: AgglomerativeClustering(3).fit_predict(Z)
        if n <= 12000:
            algos["HDBSCAN"] = lambda: HDBSCAN(min_cluster_size=max(15, n // 50)).fit_predict(Z)
        for name, fn in algos.items():
            t0 = time.perf_counter()
            lab = fn()
            rows.append({"n": n, "Thuật toán": name, "Thời gian (s)": time.perf_counter() - t0,
                         "ARI": adjusted_rand_score(y, lab)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- reliability
def stability(params, X_tr, X_te, B=20, seed=42):
    ref = SampledKMeans(**params, random_state=seed).fit(X_tr)
    ref_pred = ref.predict(X_te)
    rng = np.random.default_rng(seed)
    n = len(X_tr)
    boot, seeds = [], []
    for b in range(B):
        idx = rng.integers(0, n, n)
        m = SampledKMeans(**params, random_state=seed + b + 1).fit(X_tr.iloc[idx])
        boot.append(adjusted_rand_score(ref_pred, m.predict(X_te)))
        m2 = SampledKMeans(**params, random_state=seed + 1000 + b).fit(X_tr)
        seeds.append(adjusted_rand_score(ref_pred, m2.predict(X_te)))
    return np.array(boot), np.array(seeds)


def k_scan(Z, ks=range(2, 9), seed=42):
    rows = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        rows.append({"k": k, "Inertia (SSE)": km.inertia_,
                     "Silhouette": silhouette_score(Z, km.labels_, sample_size=min(2000, len(Z)), random_state=0),
                     "Davies-Bouldin": davies_bouldin_score(Z, km.labels_),
                     "Calinski-Harabasz": calinski_harabasz_score(Z, km.labels_)})
    return pd.DataFrame(rows)


def surrogate_rules(est, X, max_depth=3):
    Xc = est.cleaner_.transform(X)[est.features_]
    target = est.predict_group(X)
    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=0).fit(Xc, target)
    present = sorted(np.unique(target))
    names = [GROUP_VI[GROUPS[i]] for i in present]
    rules = export_text(tree, feature_names=list(Xc.columns), class_names=names, decimals=2)
    imp = pd.Series(tree.feature_importances_, index=Xc.columns).sort_values(ascending=False)
    return rules, float(tree.score(Xc, target)), imp


def evaluate_preprocessing(raw_with_dups, X_dedup, y_dedup, seed=42):
    """So sánh: không tiền xử lý vs. xoá ngoại lai IQR thô bạo vs. pipeline đầy đủ."""
    feats = FEATURE_SETS["behavior"]
    yc = y_codes(y_dedup)
    rows = []

    # A. Không tiền xử lý: bỏ dòng thiếu, giữ nguyên ngoại lai/sai đơn vị/trùng
    A = raw_with_dups.dropna(subset=feats)
    Z = StandardScaler().fit_transform(A[feats])
    lab = KMeans(3, n_init=10, random_state=seed).fit_predict(Z)
    yA = y_codes(A["_true_group"])
    rows.append({"Phương án": "A. Không tiền xử lý (chỉ dropna)", "% dòng giữ lại": 100 * len(A) / len(raw_with_dups),
                 "ARI": adjusted_rand_score(yA, lab),
                 "F1_macro": f1_score(yA, hungarian_map(yA, lab), average="macro", labels=[0, 1, 2], zero_division=0),
                 "Late_recall": recall_score(yA, hungarian_map(yA, lab), labels=[2], average="macro", zero_division=0),
                 "Late_precision": precision_score(yA, hungarian_map(yA, lab), labels=[2], average="macro", zero_division=0),
                 "Kích thước cụm": str(sorted(np.bincount(lab).tolist()))})

    # B. Làm sạch nhưng XOÁ ngoại lai IQR×1.5 trên arrival_minutes (sai lầm phổ biến)
    R = apply_rules(X_dedup)
    lo, hi = R["arrival_minutes"].quantile([0.25, 0.75])
    iqr = hi - lo
    keep = ~((R["arrival_minutes"] < lo - 1.5 * iqr) | (R["arrival_minutes"] > hi + 1.5 * iqr))
    estB = SampledKMeans("random", 1.0, "behavior", "standard", random_state=seed).fit(X_dedup[keep.to_numpy()])
    pB = estB.predict_group(X_dedup)
    rows.append({"Phương án": "B. Làm sạch + XOÁ ngoại lai IQR×1.5 ở arrival", "% dòng giữ lại": 100 * keep.mean(),
                 "ARI": adjusted_rand_score(yc, pB), "F1_macro": f1_score(yc, pB, average="macro", zero_division=0),
                 "Late_recall": recall_score(yc, pB, labels=[2], average="macro", zero_division=0),
                 "Late_precision": precision_score(yc, pB, labels=[2], average="macro", zero_division=0),
                 "Kích thước cụm": str(sorted(np.bincount(estB.predict(X_dedup)).tolist()))})

    # C. Pipeline đầy đủ (luật miền + IQR×3 trừ arrival + median + chuẩn hoá)
    estC = SampledKMeans("random", 1.0, "behavior", "standard", random_state=seed).fit(X_dedup)
    pC = estC.predict_group(X_dedup)
    rows.append({"Phương án": "C. Pipeline tiền xử lý đầy đủ", "% dòng giữ lại": 100.0,
                 "ARI": adjusted_rand_score(yc, pC), "F1_macro": f1_score(yc, pC, average="macro", zero_division=0),
                 "Late_recall": recall_score(yc, pC, labels=[2], average="macro", zero_division=0),
                 "Late_precision": precision_score(yc, pC, labels=[2], average="macro", zero_division=0),
                 "Kích thước cụm": str(sorted(np.bincount(estC.predict(X_dedup)).tolist()))})
    removed_late = int(((~keep).to_numpy() & (np.asarray(y_dedup) == "late")).sum())
    return pd.DataFrame(rows), removed_late, int((~keep).sum())


def save_learnt(path, payload: dict) -> bytes:
    payload = {**payload, "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
               "versions": {"sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__}}
    blob = pickle.dumps(payload)
    with open(path, "wb") as f:
        f.write(blob)
    return blob
