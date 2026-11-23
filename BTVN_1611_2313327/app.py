"""Streamlit: Gom cụm sinh viên đi học SỚM / ĐÚNG GIỜ / TRỄ bằng K-Means (k=3).

Mạch trình bày mỗi bước: 💡 Ý tưởng -> 🧭 Dẫn dắt -> ⚖️ Phản biện -> 📊 Minh chứng.
Chạy:  streamlit run app.py
"""
import os

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, davies_bouldin_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import f_classif

from core import data_gen as dg
from core import dbscan as dbm
from core import model as md
from core import preprocess as pp

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Gom cụm giờ đi học", page_icon="🎓", layout="wide")

LEARNT_PATH = Path(__file__).parent / "learnt.pkl"
COLORS = {"Đi sớm": "#2563eb", "Đúng giờ": "#16a34a", "Đi trễ": "#dc2626", "Nhiễu": "#9ca3af"}
VI = dg.GROUP_VI
GROUP_ORDER = ["Đi sớm", "Đúng giờ", "Đi trễ"]
GRID_DEFAULT = {"sampling": ["random", "stratified", "balanced", "bootstrap"], "frac": [0.3, 0.6, 1.0],
                "features": ["core", "behavior", "all"], "scaler": ["standard", "robust"], "impute": ["median"]}

STAGES = {"idea": ("💡", "Ý tưởng"), "lead": ("🧭", "Dẫn dắt"), "critique": ("⚖️", "Phản biện"),
          "proof": ("📊", "Minh chứng")}


def stage(kind: str, text: str | None = None):
    icon, title = STAGES[kind]
    st.markdown(f"### {icon} {title}")
    if text:
        st.markdown(text)


def debate(claim: str, rebuttal: str):
    c1, c2 = st.columns(2)
    c1.info(f"**🤖 Chatbot / lập luận thường gặp**\n\n{claim}")
    c2.warning(f"**🧑‍🎓 Phản biện**\n\n{rebuttal}")


# ============================================================ sidebar & data
PAGES = ["0. Tổng quan", "1. Dữ liệu giả lập & mất cân bằng", "2. Tiền xử lý",
         "3. Chọn mẫu, thuộc tính & Grid Search CV", "4. K-Means k=3 & learnt.pkl",
         "5. DBSCAN (bổ sung)", "6. So sánh mô hình", "7. Đánh giá & độ tin cậy"]

with st.sidebar:
    st.title("🎓 Gom cụm giờ đi học")
    page = st.radio("Trang", PAGES, label_visibility="collapsed")
    with st.expander("⚙️ Cấu hình dữ liệu giả lập"):
        n = st.slider("Số sinh viên", 500, 5000, 2000, 250)
        seed = int(st.number_input("Seed", 0, 9999, 42))
        p_early = st.slider("% nhóm đi sớm", 5, 40, 18)
        p_late = st.slider("% nhóm đi trễ (thiểu số)", 3, 30, 10)
        st.caption(f"% nhóm đúng giờ = {100 - p_early - p_late}")
        miss = st.slider("% thiếu (missing)", 0, 30, 8)
        outl = st.slider("% ngoại lai mỗi cột", 0, 10, 2)
        noise = st.slider("% dòng bị nhiễu mỗi cột", 0, 40, 10)
        dup = st.slider("% bản ghi trùng", 0, 10, 3)
    st.caption("Nhãn thật `_true_group` bị ẩn khỏi mô hình — chỉ dùng để kiểm chứng.")

CFG = (("n", n), ("seed", seed), ("ratios", (p_early / 100, 1 - (p_early + p_late) / 100, p_late / 100)),
       ("missing", miss / 100), ("outlier", outl / 100), ("noise", noise / 100), ("dup", dup / 100))


@st.cache_data(show_spinner="Đang sinh dữ liệu…")
def load_data(cfg):
    c = dict(cfg)
    raw, clean, log = dg.generate(c["n"], c["ratios"], c["seed"], missing=c["missing"], outlier=c["outlier"],
                                  noise=c["noise"], dup=c["dup"])
    d, ndup = pp.deduplicate(raw)
    X, y = d.drop(columns="_true_group"), d["_true_group"]
    strata = pp.arrival_strata(pp.apply_rules(X)["arrival_minutes"])
    Xtr, Xte, ytr, yte, s_tr, s_te = train_test_split(X, y, strata, test_size=0.2, stratify=strata,
                                                      random_state=c["seed"])
    return dict(raw=raw, clean=clean, log=log, ndup=ndup, X=X, y=y, strata=strata,
                Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, s_tr=s_tr, s_te=s_te)


D = load_data(CFG)

for k, v in [("grid", GRID_DEFAULT), ("refit", "F1_macro"), ("n_splits", 5), ("one_se", False)]:
    st.session_state.setdefault(k, v)


def gkey():
    g = st.session_state.grid
    return (CFG, tuple((k, tuple(v)) for k, v in g.items()), st.session_state.refit,
            st.session_state.n_splits, st.session_state.one_se)


@st.cache_resource(show_spinner="Đang chạy Grid Search CV (≈30 giây cho lưới mặc định)…")
def get_grid(gk):
    cfg, grid, refit, n_splits, one_se = gk
    D_ = load_data(cfg)
    return md.run_grid_search(D_["Xtr"], D_["ytr"], D_["s_tr"], {k: list(v) for k, v in grid},
                              refit=refit, n_splits=n_splits, seed=dict(cfg)["seed"], one_se=one_se)


def group_names(codes):
    return [VI[dg.GROUPS[c]] for c in codes]


def fmt(x, d=3):
    return f"{x:.{d}f}"


# ============================================================ pages
def page_overview():
    st.title("Gom cụm sinh viên: đi học sớm, đúng giờ hay trễ?")
    st.markdown(
        "Bài tập Data Mining (Chap 2: dữ liệu, chất lượng dữ liệu, tiền xử lý, chọn mẫu, giảm chiều). "
        "Ứng dụng đi qua toàn bộ quy trình và **mỗi bước đều có 4 phần**: "
        "💡 *Ý tưởng* → 🧭 *Dẫn dắt* → ⚖️ *Phản biện* → 📊 *Minh chứng (giải thích được kết quả)*.")
    st.markdown("#### Yêu cầu đề bài → nơi được giải quyết")
    st.table(pd.DataFrame([
        ("Tạo dữ liệu giả lập mất cân bằng, chưa tiền xử lý (missing, noisy, outliers)", "Trang 1"),
        ("Làm rõ dữ liệu có mất cân bằng không; tiền xử lý gì và vì sao", "Trang 1, 2"),
        ("Chọn mẫu phản ánh đặc điểm bài toán bằng Grid Search Cross-Validation", "Trang 3"),
        ("Trích chọn thuộc tính cho mô hình gom cụm", "Trang 3"),
        ("Phản biện: vì sao CV tốt hơn Bootstrap (và khi nào không)", "Trang 3"),
        ("K-Means k = 3 (sớm / đúng giờ / trễ); lưu kết quả learnt.pkl", "Trang 4"),
        ("DBSCAN đầy đủ: chọn eps/minPts, Grid Search CV, phát hiện ngoại lai, learnt_dbscan.pkl", "Trang 5"),
        ("So sánh mô hình khác; phản biện K-Means vs DBSCAN", "Trang 5, 6"),
        ("Đánh giá: mô hình học được gì, có đáng tin cậy không", "Trang 7"),
    ], columns=["Yêu cầu", "Trang"]))
    st.markdown("#### Quy trình")
    st.graphviz_chart("""
    digraph G { rankdir=LR; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#e0e7ff", fontname="Arial", fontsize=11];
      raw [label="Dữ liệu bẩn\\n(mất cân bằng)", fillcolor="#fee2e2"];
      dedup [label="Loại trùng"]; rules [label="Chuẩn hoá nhãn\\nSửa đơn vị\\nLuật miền → NaN"];
      split [label="Tách train/test\\n(phân tầng)"];
      iqr [label="IQR×3 (trừ arrival)\\nĐiền khuyết\\nMã hoá"];
      samp [label="Chọn mẫu\\n(random/phân tầng/\\ncân bằng/bootstrap)"];
      feat [label="Chọn thuộc tính\\n+ chuẩn hoá"]; km [label="K-Means k=3", fillcolor="#dcfce7"];
      gs [label="Grid Search\\nStratified K-fold CV", shape=ellipse, fillcolor="#fef3c7"];
      pkl [label="learnt.pkl", shape=note, fillcolor="#fef9c3"];
      ev [label="Đánh giá: test, ổn định\\nbootstrap, so sánh mô hình", fillcolor="#dcfce7"];
      raw->dedup->rules->split->iqr->samp->feat->km->pkl->ev; gs->samp [style=dashed]; gs->feat [style=dashed];
    }""")
    st.markdown("#### Các giả thuyết sẽ được kiểm chứng bằng số liệu (không chỉ nói suông)")
    st.markdown(
        "- **H1.** Dữ liệu mất cân bằng → mẫu ngẫu nhiên nhỏ có thể làm méo tỉ lệ nhóm *trễ*; mẫu phân tầng giữ đúng tỉ lệ.\n"
        "- **H2.** Xoá ngoại lai bằng IQR một cách máy móc sẽ **xoá luôn nhóm thiểu số** (sinh viên trễ).\n"
        "- **H3.** Thuộc tính dư thừa/không liên quan (commute, gpa, age) làm K-Means kém đi → cần trích chọn.\n"
        "- **H4.** K-fold CV phù hợp hơn Bootstrap để *chọn mô hình*; Bootstrap phù hợp để *đo độ ổn định*.\n"
        "- **H5.** \"DBSCAN chính xác hơn nhưng chậm\" — có đúng với bài toán này không?")
    st.info("Vì dữ liệu là **giả lập**, ta biết nhãn thật (bị ẩn khỏi mô hình). Đây là cách duy nhất để trả lời "
            "câu hỏi *“mô hình có thực sự học đúng không?”* — điều không thể làm với dữ liệu thật không nhãn.")


def page_data():
    st.title("1. Dữ liệu giả lập & mất cân bằng")
    raw, y = D["raw"], D["y"]
    stage("idea", "Mô phỏng nhật ký đến lớp của sinh viên trong một học kỳ. Có **3 nhóm hành vi ẩn** với tỉ lệ "
                  "lệch nhau, rồi cố ý làm bẩn dữ liệu như ngoài đời: thiếu, nhiễu, ngoại lai, sai đơn vị, nhập trùng.")
    stage("lead")
    st.markdown("**Thuộc tính** (phân loại theo Chap 2) và vai trò thật mà ta cố tình thiết kế để kiểm tra bước trích chọn:")
    st.dataframe(dg.FEATURE_INFO, hide_index=True, use_container_width=True)
    st.markdown(f"**Dữ liệu thô** ({len(raw):,} dòng, gồm cả bản trùng). Cột `_true_group` là nhãn ẩn:")
    st.dataframe(raw.head(15), use_container_width=True)
    st.download_button("⬇️ Tải dữ liệu thô (CSV)", raw.to_csv(index=False).encode("utf-8-sig"), "raw_students.csv")

    st.markdown("#### Dữ liệu có mất cân bằng không?")
    vc = y.value_counts().reindex(dg.GROUPS)
    strata = pd.Series(D["strata"]).value_counts().reindex(pp.STRATA_ORDER).fillna(0)
    c1, c2 = st.columns(2)
    f = px.bar(x=[VI[g] for g in vc.index], y=vc.values, color=[VI[g] for g in vc.index], color_discrete_map=COLORS,
               text=[f"{v} ({100 * v / vc.sum():.1f}%)" for v in vc.values],
               labels={"x": "", "y": "Số sinh viên"}, title="Nhãn thật (ẩn) — chỉ biết vì là dữ liệu giả lập")
    c1.plotly_chart(f.update_layout(showlegend=False), use_container_width=True)
    f = px.bar(x=strata.index, y=strata.values, text=[f"{int(v)}" for v in strata.values],
               labels={"x": "", "y": "Số sinh viên"}, title="Tầng theo giờ đến (luật ±5 phút) — quan sát được")
    c2.plotly_chart(f, use_container_width=True)
    ir = vc.max() / vc.min()
    m1, m2, m3 = st.columns(3)
    m1.metric("Tỉ lệ mất cân bằng (lớn nhất / nhỏ nhất)", f"{ir:.1f} : 1")
    m2.metric("Accuracy nếu đoán tất cả 'Đúng giờ'", f"{100 * vc['on_time'] / vc.sum():.1f}%")
    p = vc / vc.sum()
    m3.metric("Độ cân bằng Shannon (1 = cân bằng hoàn toàn)", fmt(float(-(p * np.log(p)).sum() / np.log(3)), 2))
    st.success(f"**Kết luận:** dữ liệu **mất cân bằng** — nhóm *đi trễ* chỉ chiếm {100 * p['late']:.1f}%, "
               f"ít hơn nhóm đúng giờ {ir:.1f} lần. Tầng giờ đến (cột phải) cho ra cùng bức tranh, nên ngoài đời "
               "(không có nhãn) ta vẫn phát hiện được sự mất cân bằng.")

    stage("critique")
    debate("Mô hình đạt accuracy 72% là tốt.",
           "Một mô hình **không học gì** (đoán tất cả *đúng giờ*) cũng đạt ~72%. Với dữ liệu mất cân bằng phải dùng "
           "**F1-macro** (mỗi nhóm có trọng số bằng nhau) và **recall của nhóm trễ** — nhóm ta quan tâm nhất.")
    debate("Gom cụm là học không giám sát, không có nhãn thì sao biết mất cân bằng?",
           "Dùng **tri thức miền**: đến sau giờ học > 5 phút là trễ. Tầng giờ đến này vừa để *phát hiện* mất cân bằng, "
           "vừa để **chọn mẫu phân tầng** và **chia fold** — không động đến nhãn ẩn.")
    debate("K-Means tự tìm được nhóm nhỏ.",
           "K-Means tối thiểu SSE nên có **“hiệu ứng đồng đều”**: hay chẻ nhóm lớn và nuốt nhóm nhỏ. "
           "Nhóm thiểu số chỉ được tìm thấy nếu nó **tách biệt đủ rõ**, điều này sẽ được kiểm chứng ở trang 3 và 7.")

    stage("proof", "**Báo cáo chất lượng dữ liệu thô** (đối chiếu với nhật ký lỗi đã chèn):")
    c1, c2 = st.columns([3, 2])
    c1.dataframe(pp.quality_report(raw), hide_index=True, use_container_width=True)
    c2.dataframe(pd.DataFrame(list(D["log"].items()), columns=["Lỗi đã chèn", "Số lượng"]), hide_index=True,
                 use_container_width=True)
    c2.metric("Bản ghi trùng phát hiện được (theo student_id)", D["ndup"])
    col = st.selectbox("Xem phân phối thô của", dg.NUMERIC_COLS, index=0)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.histogram(raw, x=col, nbins=80, title=f"Histogram {col} (thô)"), use_container_width=True)
    c2.plotly_chart(px.box(raw, y=col, points="outliers", title=f"Boxplot {col} (thô)"), use_container_width=True)
    st.caption("Histogram cho thấy các giá trị vô lý (vd 999 phút, -300 phút) kéo dãn trục — K-Means rất nhạy với chúng vì dùng trung bình.")


def page_preprocess():
    st.title("2. Tiền xử lý — làm gì, vì sao, và có hiệu quả không?")
    X, y, clean = D["X"], D["y"], D["clean"]
    stage("idea", "K-Means dùng **khoảng cách Euclid** và **trung bình**, nên rất nhạy với: giá trị thiếu (không tính "
                  "được khoảng cách), ngoại lai (kéo tâm cụm), thang đo khác nhau (thuộc tính có đơn vị lớn lấn át), "
                  "và thuộc tính phân loại. Mỗi bước tiền xử lý dưới đây xử lý đúng một vấn đề như vậy.")
    stage("lead")
    R, rep = pp.apply_rules(X, return_report=True)
    steps = pd.DataFrame([
        ("1", "Loại bản ghi trùng", "drop_duplicates(student_id)", f"{D['ndup']} dòng bị xoá", "Data integration — redundancy"),
        ("2", "Chuẩn hoá nhãn phân loại", "strip + lower + từ điển đồng nghĩa (xe máy → motorbike)",
         f"{rep['transport_variants_before']} biến thể → {rep['transport_variants_after']} nhãn", "Inconsistent data"),
        ("3", "Sửa sai đơn vị", "distance > 60 km ⇒ hiểu là mét, chia 1000", f"{rep['unit_fixed']} giá trị", "Data discrepancy"),
        ("4", "Luật miền → coi như thiếu", "giá trị ngoài khoảng hợp lý (vd ngủ 22h, tuổi 150) → NaN",
         f"{sum(rep['invalid_to_nan'].values())} giá trị", "Noisy / outliers"),
        ("5", "Kẹp ngoại lai thống kê", "IQR×3 (winsorize) cho arrival_std, distance, commute, sleep, gpa — KHÔNG cho arrival",
         "học ngưỡng trên train", "Outlier handling"),
        ("6", "Điền khuyết", "median (số) / mode (phân loại); so sánh với mean, KNN bên dưới", "học trên train", "Missing data"),
        ("7", "Mã hoá", "one-hot cho transport", "4 cột tr_*", "Data transformation"),
        ("8", "Chuẩn hoá thang đo", "z-score (Standard) hoặc Robust (median/IQR) — chọn bằng Grid Search", "trong pipeline", "Normalization"),
    ], columns=["#", "Bước", "Cách làm", "Tác động trên dữ liệu này", "Khái niệm Chap 2"])
    st.dataframe(steps, hide_index=True, use_container_width=True)
    st.markdown("**Bước 4 chi tiết — số giá trị vô lý bị chuyển thành NaN theo cột:**")
    st.bar_chart(pd.Series(rep["invalid_to_nan"]), horizontal=True)

    c1, c2 = st.columns(2)
    before = X["transport"].astype(str).value_counts().head(20)
    c1.markdown("**Bước 2 — transport trước khi chuẩn hoá**")
    c1.dataframe(before.rename("số dòng"), use_container_width=True, height=300)
    c2.markdown("**… và sau khi chuẩn hoá**")
    c2.dataframe(R["transport"].value_counts(dropna=False).rename("số dòng"), use_container_width=True)

    stage("critique")
    lo, hi = pp.iqr_bounds(R["arrival_minutes"].dropna(), 1.5)
    flagged = (R["arrival_minutes"] < lo) | (R["arrival_minutes"] > hi)
    late_flagged = int((flagged & (y == "late")).sum())
    n_late = int((y == "late").sum())
    debate("Cứ dùng IQR×1.5 để loại bỏ ngoại lai cho mọi cột là chuẩn.",
           f"Với `arrival_minutes`, hàng rào IQR×1.5 là [{lo:.1f}, {hi:.1f}] phút. Nó đánh dấu **{int(flagged.sum())}** "
           f"sinh viên là ngoại lai, trong đó **{late_flagged}/{n_late} ({100 * late_flagged / n_late:.0f}%) là sinh viên "
           "trễ thật**. Nhóm thiểu số **chính là đuôi phân phối**, xoá nó là xoá luôn câu trả lời. Vì thế: giá trị "
           "*vô lý* xử lý bằng **luật miền**, còn IQR chỉ dùng (với k=3, kẹp chứ không xoá) cho các cột khác.")
    debate("Tiền xử lý trên toàn bộ dữ liệu rồi mới chia train/test cho tiện.",
           "Như vậy median, ngưỡng IQR… được học **có cả dữ liệu test** → rò rỉ thông tin, đánh giá lạc quan. "
           "Ở đây `StudentCleaner` là transformer sklearn: `fit` trên fold train, chỉ `transform` fold validation/test, "
           "và được **lưu cùng mô hình** trong learnt.pkl. Chỉ những bước *không học tham số* (loại trùng, luật miền) mới làm trước.")
    debate("Xoá hết dòng thiếu là sạch nhất.",
           "Xoá dòng làm mất dữ liệu (xem phương án A bên dưới) và **thiên lệch** nếu thiếu không ngẫu nhiên — ở đây "
           "sinh viên trễ hay bỏ trống giờ dậy (MNAR), xoá dòng thiếu sẽ xoá thêm nhóm trễ.")

    stage("proof")
    st.markdown("**(a) Chọn cách điền khuyết** — vì có dữ liệu sạch gốc, ta đo được sai số của giá trị điền so với giá trị thật:")
    truth = clean.set_index("student_id").loc[X["student_id"]].reset_index(drop=True)
    miss_mask = R[dg.NUMERIC_COLS].isna()
    rows = []
    for method in ["mean", "median", "knn"]:
        filled = pp.StudentCleaner(impute=method).fit(X).transform(X)
        for c in dg.NUMERIC_COLS:
            m = miss_mask[c].to_numpy()
            if m.sum() == 0:
                continue
            err = filled.loc[m, c].to_numpy() - truth.loc[m, c].to_numpy()
            rows.append({"Phương pháp": method, "Cột": c, "NRMSE": np.sqrt(np.mean(err ** 2)) / truth[c].std()})
    imp = pd.DataFrame(rows)
    piv = imp.pivot(index="Cột", columns="Phương pháp", values="NRMSE")
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(px.bar(imp, x="Cột", y="NRMSE", color="Phương pháp", barmode="group",
                           title="Sai số điền khuyết (NRMSE, thấp hơn = tốt hơn)"), use_container_width=True)
    c2.dataframe(piv.style.format("{:.3f}").highlight_min(axis=1, color="#bbf7d0"), use_container_width=True)
    best = piv.mean().idxmin()
    c2.markdown(f"Trung bình NRMSE: " + ", ".join(f"**{k}** {v:.3f}" for k, v in piv.mean().items()) +
                f". Tốt nhất: **{best}**. KNN tận dụng tương quan giữa các cột (vd distance ↔ commute) nên thường "
                "điền tốt hơn ở các cột có quan hệ; median đơn giản, bền với ngoại lai và nhanh. "
                "Bạn có thể thêm `knn` vào lưới Grid Search ở trang 3 để xem nó có cải thiện việc gom cụm không.")

    st.markdown("**(b) Tiền xử lý có thực sự giúp mô hình không?** — cùng K-Means k=3, cùng bộ thuộc tính *behavior*:")
    res, removed_late, removed = md.evaluate_preprocessing(D["raw"], X, y, seed)
    st.dataframe(res.style.format({"% dòng giữ lại": "{:.1f}", "ARI": "{:.3f}", "F1_macro": "{:.3f}",
                                   "Late_recall": "{:.3f}", "Late_precision": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    a, b, c = res.iloc[0], res.iloc[1], res.iloc[2]
    st.markdown(
        f"- **A** (không tiền xử lý): kích thước cụm {a['Kích thước cụm']} — các giá trị cực đoan (999 phút, khoảng cách "
        f"tính bằng mét…) chiếm riêng 1–2 cụm, F1 chỉ **{a['F1_macro']:.2f}**.\n"
        f"- **B** (xoá ngoại lai IQR×1.5 ở arrival): xoá {removed} dòng, trong đó **{removed_late} sinh viên trễ thật**. "
        f"Thiếu nhóm trễ nhưng vẫn bị ép k = 3, K-Means **chẻ đôi nhóm đúng giờ** (kích thước cụm {b['Kích thước cụm']}). "
        f"Recall nhóm trễ trông cao ({b['Late_recall']:.2f}) chỉ vì cụm \"trễ\" phình ra, gom cả SV đúng giờ: precision "
        f"nhóm trễ chỉ **{b['Late_precision']:.2f}**, F1 **{b['F1_macro']:.2f}** → không được chỉ nhìn một chỉ số.\n"
        f"- **C** (pipeline đầy đủ): F1 **{c['F1_macro']:.2f}**, recall nhóm trễ **{c['Late_recall']:.2f}**, "
        f"precision nhóm trễ **{c['Late_precision']:.2f}**, giữ 100% dòng.")
    Xc = pp.StudentCleaner().fit(X).transform(X)
    st.markdown("**(c) Phân phối sau tiền xử lý** (so với thô ở trang 1):")
    col = st.selectbox("Thuộc tính", dg.NUMERIC_COLS, index=0, key="pp_col")
    fig = go.Figure()
    fig.add_histogram(x=R[col], name="sau luật miền (còn NaN)", opacity=0.5, nbinsx=60)
    fig.add_histogram(x=Xc[col], name="sau toàn bộ pipeline", opacity=0.5, nbinsx=60)
    st.plotly_chart(fig.update_layout(barmode="overlay", title=col), use_container_width=True)
    st.caption(f"Sau pipeline: {int(Xc[dg.NUMERIC_COLS].isna().sum().sum())} ô thiếu, transport còn "
               f"{Xc['transport'].nunique()} nhãn. Cột dồn ở median là dấu vết của việc điền khuyết — một hạn chế cần biết.")


def page_grid():
    st.title("3. Chọn mẫu, trích chọn thuộc tính & Grid Search Cross-Validation")
    Xtr, ytr = D["Xtr"], D["ytr"]
    stage("idea", "Mẫu tốt phải **phản ánh đủ đặc tính bài toán**: giữ được nhóm thiểu số *đi trễ* và hình dạng phân "
                  "phối giờ đến. Thuộc tính tốt phải mang thông tin về *giờ giấc*, không phải nhiễu. Thay vì đoán, ta "
                  "để **Grid Search + Stratified K-fold CV** thử mọi tổ hợp (cách chọn mẫu × tỉ lệ mẫu × bộ thuộc tính × "
                  "cách chuẩn hoá) và chấm điểm trên dữ liệu chưa thấy.")
    stage("lead")
    st.markdown("#### (a) Các cách chọn mẫu (Chap 2 — Sampling)")
    st.table(pd.DataFrame([(k, v) for k, v in md.SAMPLING_VI.items()], columns=["Tham số", "Ý nghĩa"]))
    st.markdown("Tầng (strata) = sớm / đúng giờ / trễ / chưa rõ theo luật ±5 phút trên `arrival_minutes` "
                "— **không dùng nhãn ẩn**. Thử rút 1 mẫu nhiều lần để xem mẫu có *đại diện* không:")
    c1, c2 = st.columns([1, 3])
    frac = c1.slider("Tỉ lệ mẫu", 0.02, 1.0, 0.10, 0.02)
    reps = c1.slider("Số lần rút", 50, 400, 200, 50)
    rep = md.representativeness(Xtr, ytr, frac, reps, seed)
    true_late = (ytr == "late").mean()
    fig = px.box(rep, x="method", y="late_share_true", color="method", points=False,
                 labels={"late_share_true": "Tỉ lệ SV trễ thật trong mẫu", "method": ""},
                 title=f"Tỉ lệ nhóm trễ trong mẫu (đường đỏ = tổng thể {100 * true_late:.1f}%)")
    fig.add_hline(y=true_late, line_color="red", line_dash="dash")
    c2.plotly_chart(fig.update_layout(showlegend=False), use_container_width=True)
    agg = rep.groupby("method").agg(late_mean=("late_share_true", "mean"), late_std=("late_share_true", "std"),
                                    ks_mean=("ks", "mean"), unique_ratio=("unique_ratio", "mean"))
    st.dataframe(agg.rename(columns={"late_mean": "Tỉ lệ trễ TB", "late_std": "Độ lệch chuẩn tỉ lệ trễ",
                                     "ks_mean": "KS (phân phối giờ đến vs tổng thể)",
                                     "unique_ratio": "Tỉ lệ điểm không trùng"}).style.format("{:.3f}"),
                 use_container_width=True)
    st.caption("Phân tầng: tỉ lệ đúng và dao động ít nhất. Ngẫu nhiên/Bootstrap: đúng *trung bình* nhưng dao động — "
               "mẫu nhỏ có thể thiếu hẳn nhóm trễ. Cân bằng: cố ý **khuếch đại** nhóm trễ (lệch tổng thể, KS lớn). "
               "Bootstrap: một phần điểm bị lặp.")

    st.markdown("#### (b) Trích chọn thuộc tính")
    Xc = pp.StudentCleaner().fit(Xtr).transform(Xtr)
    num = dg.NUMERIC_COLS
    c1, c2 = st.columns(2)
    corr = Xc[num].corr()
    c1.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                              title="Tương quan Pearson (sau làm sạch)"), use_container_width=True)
    strata_tr = pp.arrival_strata(pp.apply_rules(Xtr)["arrival_minutes"])
    known = strata_tr != pp.UNKNOWN
    F, _ = f_classif(Xc.loc[known, num], strata_tr[known])
    fs = pd.Series(F, index=num).sort_values()
    c2.plotly_chart(px.bar(fs, orientation="h", log_x=True, title="Mức liên quan với tầng giờ đến (ANOVA F, thang log)",
                           labels={"value": "F", "index": ""}).update_layout(showlegend=False), use_container_width=True)
    r_dc = corr.loc["distance_km", "commute_min"]
    st.markdown(
        f"- **Dư thừa:** `commute_min` tương quan **{r_dc:.2f}** với `distance_km` (được suy ra từ nó) → giữ cả hai là "
        "đếm *khoảng cách đi lại* hai lần trong khoảng cách Euclid.\n"
        "- **Không liên quan:** `gpa`, `age` có F rất thấp → chỉ thêm nhiễu vào khoảng cách, làm các cụm mờ đi (curse of dimensionality).\n"
        "- `arrival_minutes` có F lớn nhất — hiển nhiên, vì tầng được tạo từ chính nó (nên chỉ dùng F để *xếp hạng sơ bộ*).\n"
        "- **3 bộ ứng viên** đưa vào Grid Search: " + "; ".join(f"`{k}` = {', '.join(v)}" for k, v in md.FEATURE_SETS.items()))

    st.markdown("#### (c) Grid Search CV")
    g = st.session_state.grid
    with st.form("grid_form"):
        c1, c2, c3 = st.columns(3)
        s_sampling = c1.multiselect("Cách chọn mẫu", list(md.SAMPLING_VI), g["sampling"])
        s_frac = c1.multiselect("Tỉ lệ mẫu", [0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0], g["frac"])
        s_feat = c2.multiselect("Bộ thuộc tính", list(md.FEATURE_SETS), g["features"])
        s_scaler = c2.multiselect("Chuẩn hoá", list(md.SCALERS), g["scaler"])
        s_imp = c3.multiselect("Điền khuyết", ["median", "mean", "knn"], g["impute"])
        refit = c3.selectbox("Tiêu chí chọn cấu hình", list(md.SCORERS), list(md.SCORERS).index(st.session_state.refit))
        n_splits = c1.slider("Số fold K (Stratified K-fold)", 3, 10, st.session_state.n_splits)
        one_se = c2.toggle("Quy tắc 1-SE (chọn cấu hình đơn giản nhất trong 1 sai số chuẩn)", st.session_state.one_se)
        n_cfg = max(1, len(s_sampling) * len(s_frac) * len(s_feat) * len(s_scaler) * len(s_imp))
        c3.caption(f"{n_cfg} cấu hình × {n_splits} fold = {n_cfg * n_splits} lần huấn luyện")
        if st.form_submit_button("▶️ Chạy Grid Search CV", type="primary"):
            if not all([s_sampling, s_frac, s_feat, s_scaler, s_imp]):
                st.error("Mỗi tham số cần ít nhất 1 giá trị.")
            else:
                st.session_state.grid = {"sampling": s_sampling, "frac": sorted(s_frac), "features": s_feat,
                                         "scaler": s_scaler, "impute": s_imp}
                st.session_state.refit, st.session_state.n_splits, st.session_state.one_se = refit, n_splits, one_se
                st.rerun()

    gs, elapsed = get_grid(gkey())
    metric = gs.metric_
    res = md.cv_results_frame(gs)
    chosen = res[res["chosen"]].iloc[0]
    best_params = gs.best_params_
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Thời gian Grid Search", f"{elapsed:.1f} s")
    m2.metric(f"{metric} CV của cấu hình chọn", f"{chosen[metric]:.3f} ± {chosen[metric + '_std']:.3f}")
    m3.metric(f"{metric} trên TEST (chưa từng thấy)", fmt(md.SCORERS[metric](gs.best_estimator_, D["Xte"], D["yte"])))
    m4.metric("Số cấu hình", len(res))
    st.success("**Cấu hình được chọn:** " + ", ".join(f"`{k}` = **{v}**" for k, v in best_params.items()))
    show = res.head(12).copy()
    cols = ["chosen", "sampling", "frac", "features", "scaler", "impute", "F1_macro", "F1_macro_std", "ARI",
            "Late_recall", "Silhouette", "fit_time_s"]
    st.dataframe(show[cols].style.format({c: "{:.3f}" for c in cols[6:]}).apply(
        lambda r: ["background-color:#dcfce7" if r["chosen"] else "" for _ in r], axis=1),
        hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    piv = res.groupby(["sampling", "frac"])[metric].max().unstack()
    c1.plotly_chart(px.imshow(piv, text_auto=".3f", color_continuous_scale="Greens", aspect="auto",
                              title=f"{metric} tốt nhất theo cách chọn mẫu × tỉ lệ mẫu"), use_container_width=True)
    fb = res.loc[res.groupby("features")[metric].idxmax()]
    c2.plotly_chart(px.bar(fb, x="features", y=metric, error_y=f"{metric}_std", color="features", text_auto=".3f",
                           title=f"{metric} tốt nhất theo bộ thuộc tính (± độ lệch chuẩn CV)").update_layout(
        showlegend=False, yaxis_range=[max(0, fb[metric].min() - 0.15), 1]), use_container_width=True)
    lr = res.groupby("sampling")["Late_recall"].agg(["mean", "max"])
    best_row = res.iloc[0]
    st.markdown(
        f"**Đọc kết quả:**\n"
        f"- Bộ thuộc tính tốt nhất: **{fb.sort_values(metric).iloc[-1]['features']}**; bộ `all` "
        f"({fb.set_index('features')[metric].get('all', np.nan):.3f}) kém hơn → kiểm chứng **H3**: thuộc tính dư thừa / không liên quan làm hại K-Means.\n"
        f"- Recall nhóm trễ trung bình theo cách chọn mẫu: " +
        ", ".join(f"{k} {v:.3f}" for k, v in lr["mean"].items()) + ".\n"
        f"- Cấu hình top-1 theo {metric}: {best_row['sampling']}/{best_row['frac']}/{best_row['features']}/"
        f"{best_row['scaler']} = {best_row[metric]:.3f} ± {best_row[metric + '_std']:.3f}. Nhiều cấu hình chênh nhau "
        "**ít hơn độ lệch chuẩn CV** → khác biệt đó có thể chỉ do may rủi chia fold. Bật *quy tắc 1-SE* để chọn "
        "cấu hình đơn giản nhất (ít thuộc tính, mẫu phân tầng, mẫu nhỏ) trong vùng đó.")
    if "balanced" in res["sampling"].values:
        bal = res[res["sampling"] == "balanced"][metric].max()
        oth = res[res["sampling"] != "balanced"][metric].max()
        if bal < oth:
            st.info(f"**Một phát hiện ngược trực giác:** mẫu *cân bằng* ({bal:.3f}) **không** tốt hơn ({oth:.3f}). "
                    "Vì nhóm trễ ở đây đã tách biệt khá rõ, K-Means tìm được nó mà không cần khuếch đại; cân bằng "
                    "lại còn **làm lệch tâm cụm đúng giờ** (nhóm lớn bị lấy mẫu thiếu) và lặp lại điểm nhóm trễ. "
                    "Mẫu *phản ánh đúng đặc tính* ≠ mẫu *cân bằng*.")

    stage("critique")
    st.markdown("#### Vì sao dùng K-fold CV mà không dùng Bootstrap để chọn cấu hình?")
    debate("Bootstrap tốt hơn vì lặp được nhiều lần (B = 100–1000) nên ước lượng mượt hơn, và phương pháp .632 "
           "còn sửa được độ chệch.",
           "1. **Không trùng lặp:** trong K-fold mỗi điểm được kiểm định **đúng 1 lần**, train và validation tách rời. "
           "Mẫu bootstrap chỉ chứa ~**63,2%** điểm khác nhau, phần còn lại là bản **lặp** — với K-Means, điểm lặp = "
           "trọng số gấp đôi, **kéo lệch tâm cụm**.\n"
           "2. **Giữ tỉ lệ nhóm thiểu số:** *Stratified* K-fold giữ đúng tỉ lệ nhóm trễ trong mọi fold; "
           "tập OOB của bootstrap có kích thước và tỉ lệ nhóm **ngẫu nhiên**.\n"
           "3. **Chi phí:** Grid Search × CV = (số cấu hình × K) lần huấn luyện; × Bootstrap = (số cấu hình × B). "
           f"Với {len(res)} cấu hình: {len(res) * gs.n_splits_} so với {len(res) * 100:,} (B = 100).\n"
           "4. **Độ chệch:** OOB thường **bi quan** (train ít điểm khác nhau); .632 bù bằng điểm resubstitution vốn **lạc quan** — "
           "sửa lỗi này bằng lỗi khác.\n\n"
           "**Nhưng công bằng mà nói:** Bootstrap rất tốt để đo **độ bất định / độ ổn định** (khoảng tin cậy, mô hình "
           "có đổi khi dữ liệu đổi không) và khi dữ liệu **rất ít**. Vì vậy ở trang 7 ta dùng bootstrap đúng vai trò đó: "
           "**CV để chọn mô hình, Bootstrap để đo độ tin cậy.**")
    stage("proof", "Thực nghiệm: lấy 8 cấu hình trải đều bảng xếp hạng, ước lượng điểm bằng K-fold CV, Bootstrap OOB, "
                   "Bootstrap .632, rồi so với điểm **test thật**. Ước lượng tốt = gần test, ít dao động, xếp hạng đúng, rẻ.")
    B = st.slider("Số lần bootstrap B", 10, 100, 30, 10)
    if st.button("▶️ Chạy so sánh CV vs Bootstrap"):
        st.session_state["cvboot_B"] = B
    if "cvboot_B" in st.session_state:
        df, summ = cvboot_cached(gkey(), st.session_state["cvboot_B"])
        c1, c2 = st.columns([3, 2])
        long = df.melt(id_vars=["config"], value_vars=["CV_mean", "Boot_OOB_mean", "Boot_632", "Test"],
                       var_name="Ước lượng", value_name=metric)
        c1.plotly_chart(px.line(long, x="config", y=metric, color="Ước lượng", markers=True,
                                title="Ước lượng của từng phương pháp vs điểm test thật"), use_container_width=True)
        c2.dataframe(summ.style.format("{:.4f}"), use_container_width=True)
        best_mae = summ["Sai lệch TB so với test (MAE)"].idxmin()
        ratio = summ.loc["Bootstrap OOB", "Thời gian tổng (s)"] / summ.loc["K-fold CV", "Thời gian tổng (s)"]
        c2.markdown(
            f"- Gần điểm test nhất: **{best_mae}**.\n"
            f"- Bootstrap tốn **{ratio:.1f}×** thời gian của CV (B = {st.session_state['cvboot_B']} vs K = {gs.n_splits_}).\n"
            f"- Spearman (xếp hạng đúng thứ tự test): CV {summ.loc['K-fold CV', 'Spearman(xếp hạng, test)']:.2f}, "
            f"OOB {summ.loc['Bootstrap OOB', 'Spearman(xếp hạng, test)']:.2f}.\n\n"
            "**Kết luận trung thực:** nếu độ chính xác ước lượng của hai cách xấp xỉ nhau, CV thắng vì **rẻ hơn nhiều lần** "
            "và giữ được phân tầng; nếu Bootstrap chính xác hơn rõ rệt trên dữ liệu của bạn, đó là cái giá phải trả bằng thời gian. "
            "Đây là đánh đổi, không phải chân lý tuyệt đối.")
        with st.expander("Bảng chi tiết"):
            st.dataframe(df.round(4), hide_index=True, use_container_width=True)


@st.cache_data(show_spinner="Đang so sánh CV vs Bootstrap…")
def cvboot_cached(gk, B):
    gs, _ = get_grid(gk)
    res = md.cv_results_frame(gs)
    idx = np.unique(np.linspace(0, len(res) - 1, min(8, len(res))).round().astype(int))
    params = res.iloc[idx][["sampling", "frac", "features", "scaler", "impute"]].to_dict("records")
    for p in params:
        p["frac"] = float(p["frac"])
    D_ = load_data(gk[0])
    return md.cv_vs_bootstrap(params, D_["Xtr"], D_["ytr"], D_["s_tr"], D_["Xte"], D_["yte"],
                              metric=gs.metric_, n_splits=gs.n_splits_, B=B, seed=dict(gk[0])["seed"])


@st.cache_data(show_spinner="Đang quét k…")
def kscan_cached(gk):
    gs, _ = get_grid(gk)
    return md.k_scan(gs.best_estimator_.transform_features(load_data(gk[0])["Xtr"]))


def page_kmeans():
    st.title("4. K-Means với k = 3 & lưu kết quả learnt.pkl")
    gs, _ = get_grid(gkey())
    est = gs.best_estimator_
    Xtr, Xte, ytr, yte = D["Xtr"], D["Xte"], D["ytr"], D["yte"]
    stage("idea", "Đề bài cho trước **k = 3** (sớm / đúng giờ / trễ). Mô hình chỉ cho ra *số hiệu cụm* 0, 1, 2 — "
                  "ta phải **gán nghĩa** cho từng cụm mà không nhìn nhãn: sắp các cụm theo *giờ đến trung bình* "
                  "(thấp nhất = đi sớm, cao nhất = đi trễ).")
    stage("lead", "Huấn luyện lại cấu hình tốt nhất từ Grid Search trên **toàn bộ tập train**: "
                  + ", ".join(f"`{k}`=**{v}**" for k, v in gs.best_params_.items())
                  + f". Mẫu huấn luyện: {len(est.sample_idx_):,} dòng ({len(np.unique(est.sample_idx_)):,} không trùng).")
    st.markdown("#### Tâm cụm (đã đổi ngược về đơn vị gốc) — *“chân dung”* mỗi nhóm")
    cent = est.centroids_original()
    st.dataframe(cent.style.format({c: "{:.2f}" for c in est.features_}), hide_index=True, use_container_width=True)
    pred = pd.Series(group_names(est.predict_group(Xtr)))
    sizes = pred.value_counts().reindex(GROUP_ORDER).fillna(0).astype(int)
    st.markdown("Số SV mỗi cụm (train): " + ", ".join(f"**{k}** {v}" for k, v in sizes.items()))

    stage("critique")
    debate("Đề bảo k = 3 thì cứ dùng k = 3.",
           "k = 3 là **giả thiết nghiệp vụ** — cần kiểm tra dữ liệu có ủng hộ hay không (Elbow, Silhouette). Nếu dữ "
           "liệu thật ra có 2 hay 5 nhóm tự nhiên, ép k = 3 sẽ cho các cụm vô nghĩa.")
    ks = kscan_cached(gkey())
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.line(ks, x="k", y="Inertia (SSE)", markers=True, title="Elbow (SSE)"), use_container_width=True)
    c2.plotly_chart(px.line(ks, x="k", y="Silhouette", markers=True, title="Silhouette theo k (cao hơn = tốt hơn)"),
                    use_container_width=True)
    best_k = int(ks.loc[ks["Silhouette"].idxmax(), "k"])
    (st.success if best_k == 3 else st.warning)(
        f"Silhouette cao nhất tại **k = {best_k}**" +
        (" → dữ liệu **ủng hộ** giả thiết 3 nhóm." if best_k == 3 else
         " → dữ liệu chưa hoàn toàn ủng hộ 3 nhóm; cần thận trọng khi diễn giải nhóm nhỏ nhất."))
    debate("Gán nhãn cho cụm bằng cách so với nhãn thật là được.",
           "Ngoài đời không có nhãn. Ta gán nghĩa bằng **thứ tự giờ đến trung bình** của tâm cụm — một quy tắc "
           "*không giám sát* được lưu luôn trong learnt.pkl (`cluster_to_group_`).")

    stage("proof", "**Mô hình học được gì?** Huấn luyện một cây quyết định nông (depth 3) bắt chước kết quả gom cụm "
                   "(*surrogate model*) để đọc ra quy tắc dạng IF–THEN:")
    rules, fidelity, imp = md.surrogate_rules(est, Xtr)
    c1, c2 = st.columns([3, 2])
    c1.code(rules, language="text")
    c2.metric("Độ trung thành của cây với K-Means", f"{100 * fidelity:.1f}%")
    c2.plotly_chart(px.bar(imp[::-1], orientation="h", title="Độ quan trọng thuộc tính (theo cây)",
                           labels={"value": "", "index": ""}).update_layout(showlegend=False), use_container_width=True)
    e, o, l = cent.iloc[0], cent.iloc[1], cent.iloc[2]

    def desc(row):
        parts = [f"đến {row['arrival_minutes']:+.1f} phút so với giờ học"]
        if "wake_up_hour" in row:
            h = row["wake_up_hour"]
            parts.append(f"dậy lúc ~{int(h)}h{int(round((h % 1) * 60)):02d}")
        if "arrival_std" in row:
            parts.append(f"dao động ±{row['arrival_std']:.1f} phút")
        if "distance_km" in row:
            parts.append(f"nhà cách {row['distance_km']:.1f} km")
        if "sleep_hours" in row:
            parts.append(f"ngủ {row['sleep_hours']:.1f} giờ")
        return ", ".join(parts)

    st.markdown(f"- **Đi sớm:** {desc(e)}.\n- **Đúng giờ:** {desc(o)}.\n- **Đi trễ:** {desc(l)}.\n\n"
                f"Thuộc tính quyết định nhất là **{imp.index[0]}** ({100 * imp.iloc[0]:.0f}% độ quan trọng). "
                "Đối chiếu với hồ sơ ta dùng để sinh dữ liệu (sớm ≈ −15', đúng giờ ≈ 0', trễ ≈ +14'; dậy 5h36 / 6h09 / 6h54): "
                "mô hình **khôi phục gần đúng cấu trúc thật** dù chưa từng thấy nhãn.")
    color_by = st.radio("Tô màu theo", ["Cụm dự đoán", "Nhãn thật (ẩn)"], horizontal=True)
    Xc = est.cleaner_.transform(Xtr)
    plot_df = Xc[est.features_].copy()
    plot_df["Cụm dự đoán"] = pred.values
    plot_df["Nhãn thật (ẩn)"] = [VI[g] for g in ytr]
    yax = "wake_up_hour" if "wake_up_hour" in est.features_ else est.features_[1]
    fig = px.scatter(plot_df, x="arrival_minutes", y=yax, color=color_by, color_discrete_map=COLORS, opacity=0.6,
                     category_orders={color_by: GROUP_ORDER}, title=f"arrival_minutes × {yax} (train)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 💾 Lưu kết quả học được: `learnt.pkl`")
    payload = {
        "model": est, "best_params": gs.best_params_, "selection_metric": gs.metric_,
        "one_se_rule": st.session_state.one_se, "cv_score": md.best_cv_score(gs),
        "label_map": {int(c): dg.GROUPS[g] for c, g in est.cluster_to_group_.items()},
        "features": est.features_, "centroids_original_units": cent,
        "surrogate_rules": rules, "feature_importance": imp.to_dict(),
        "test_metrics": {k: float(f(est, Xte, yte)) for k, f in md.SCORERS.items()},
        "data_config": dict(CFG),
    }
    c1, c2 = st.columns([1, 2])
    if c1.button("💾 Lưu learnt.pkl", type="primary"):
        blob = md.save_learnt(LEARNT_PATH, payload)
        st.session_state["pkl_blob"] = blob
        c1.success(f"Đã lưu {LEARNT_PATH.name} ({len(blob) / 1024:.0f} KB)")
    if "pkl_blob" in st.session_state:
        c1.download_button("⬇️ Tải learnt.pkl", st.session_state["pkl_blob"], "learnt.pkl")
    c2.markdown("**Nội dung file:** " + ", ".join(f"`{k}`" for k in [*payload, "created_at", "versions"]))
    c2.code("import pickle\nlearnt = pickle.load(open('learnt.pkl', 'rb'))   # cần thư mục core/ cạnh file\n"
            "model = learnt['model']            # gồm cả bộ làm sạch + scaler + K-Means\n"
            "groups = model.predict_group(df_moi)  # 0 = sớm, 1 = đúng giờ, 2 = trễ", language="python")
    c2.caption("Lưu **cả pipeline** (bộ làm sạch đã học median/ngưỡng IQR + scaler + K-Means + quy tắc gán nghĩa), "
               "không chỉ tâm cụm — để dữ liệu mới đi qua đúng các bước giống lúc huấn luyện.")


DB_LEARNT_PATH = Path(__file__).parent / "learnt_dbscan.pkl"
st.session_state.setdefault("dbgrid", dbm.DB_GRID_DEFAULT)


def db_gkey():
    g = st.session_state.dbgrid
    return CFG, tuple((k, tuple(v)) for k, v in g.items()), st.session_state.n_splits


@st.cache_resource(show_spinner="Đang chạy Grid Search CV cho DBSCAN (≈30 giây cho lưới mặc định)…")
def get_db_grid(dk):
    cfg, grid, n_splits = dk
    D_ = load_data(cfg)
    return dbm.run_dbscan_grid(D_["Xtr"], D_["ytr"], D_["s_tr"], {k: list(v) for k, v in grid},
                               n_splits=n_splits, seed=dict(cfg)["seed"])


@st.cache_data(show_spinner=False)
def db_kdist(cfg, features, scaler, min_samples):
    D_ = load_data(cfg)
    cleaner = pp.StudentCleaner().fit(D_["X"])
    Z = md.SCALERS[scaler]().fit_transform(cleaner.transform(D_["X"])[md.FEATURE_SETS[features]])
    knee, kdist = md.kdistance_eps(Z, min_samples)
    return knee, kdist


@st.cache_data(show_spinner="Đang phát hiện ngoại lai…")
def outlier_cached(cfg, q):
    D_ = load_data(cfg)
    return dbm.outlier_detection(D_["X"], D_["y"], q)


def db_label_names(est, labels):
    return ["Nhiễu" if c == -1 else f"Cụm {c} → {VI[dg.GROUPS[est.cluster_to_group_[int(c)]]]}" for c in labels]


def page_dbscan():
    st.title("5. DBSCAN — gom cụm dựa trên mật độ")
    X, y = D["X"], D["y"]
    stage("idea", "DBSCAN không tìm *tâm cụm* mà tìm **vùng dày đặc**: một điểm là *core* nếu trong bán kính **eps** "
                  "có ≥ **minPts** điểm; các core nối được với nhau tạo thành một cụm; điểm không thuộc vùng dày nào là "
                  "**nhiễu**. Ưu điểm hấp dẫn: không cần chọn k, cụm hình dạng bất kỳ, tự loại nhiễu. "
                  "Câu hỏi: những ưu điểm đó có giúp ích cho bài toán *sớm / đúng giờ / trễ* không?")
    st.table(pd.DataFrame([
        ("Core point", "≥ minPts điểm trong bán kính eps", "Nằm trong lõi của cụm"),
        ("Border point", "< minPts láng giềng nhưng nằm trong eps của một core", "Rìa cụm"),
        ("Noise (-1)", "Không phải core, không gần core nào", "Ngoại lai — không thuộc nhóm nào"),
    ], columns=["Loại điểm", "Điều kiện", "Ý nghĩa"]))

    stage("lead")
    st.markdown("#### (a) Thử tay: eps và minPts thay đổi kết quả thế nào?")
    st.caption("Cùng bộ làm sạch như K-Means, chạy trên toàn bộ dữ liệu. Gán nghĩa cụm **không dùng nhãn**: cụm có giờ "
               "đến trung bình ≤ −5' là *sớm*, > 5' là *trễ*, còn lại *đúng giờ*. Điểm nhiễu bị tính là **sai** khi tính F1.")
    c1, c2, c3, c4 = st.columns(4)
    feat = c1.selectbox("Bộ thuộc tính", ["core", "behavior"], key="db_feat")
    scal = c2.selectbox("Chuẩn hoá", ["standard", "robust"], key="db_scal")
    ms = c3.slider("minPts", 3, 80, 2 * len(md.FEATURE_SETS[feat]), key=f"db_ms_{feat}")
    knee, kdist = db_kdist(CFG, feat, scal, ms)
    eps = c4.slider("eps", 0.05, 3.0, float(round(knee, 2)), 0.01, key=f"db_eps_{feat}_{scal}_{ms}")
    est = dbm.SampledDBSCAN("random", 1.0, feat, scal, min_samples=ms, eps=eps, random_state=seed).fit(X)
    sc = dbm.db_scorer(est, X, y)
    m = st.columns(5)
    m[0].metric("Số cụm", int(sc["N_clusters"]))
    m[1].metric("% nhiễu", f"{sc['Noise_pct']:.1f}%")
    m[2].metric("F1-macro", fmt(sc["F1_macro"]))
    m[3].metric("Recall nhóm trễ", fmt(sc["Late_recall"]))
    m[4].metric("ARI", fmt(sc["ARI"]))
    c1, c2 = st.columns(2)
    fig = px.line(y=kdist, labels={"x": "điểm (sắp xếp tăng dần)", "y": f"khoảng cách tới láng giềng thứ {ms}"},
                  title="Đồ thị k-distance: khuỷu ≈ eps gợi ý")
    fig.add_hline(y=knee, line_dash="dot", line_color="gray", annotation_text=f"khuỷu {knee:.2f}")
    fig.add_hline(y=eps, line_dash="dash", line_color="red", annotation_text=f"eps đang chọn {eps:.2f}")
    c1.plotly_chart(fig, use_container_width=True)
    Xc = est.cleaner_.transform(X)
    names = db_label_names(est, est.labels_)
    cmap = {n: ("#9ca3af" if n == "Nhiễu" else COLORS[n.split("→ ")[1]]) for n in set(names)}
    c2.plotly_chart(px.scatter(x=Xc["arrival_minutes"], y=Xc["wake_up_hour"], color=names, color_discrete_map=cmap,
                               opacity=0.6, labels={"x": "arrival_minutes", "y": "wake_up_hour", "color": ""},
                               title="Kết quả DBSCAN"), use_container_width=True)
    st.caption("Thử: eps nhỏ → nhóm trễ (thưa) biến thành nhiễu xám; eps lớn → sớm/đúng giờ/trễ dính thành 1–2 cụm. "
               "Rất khó tìm được một eps cho ra đúng 3 nhóm.")

    st.markdown("#### (b) Grid Search CV cho DBSCAN (cùng Stratified K-fold như K-Means)")
    st.markdown("`eps = eps_scale × eps khuỷu` (tính lại trên mỗi fold train) để eps so sánh được giữa các bộ thuộc tính. "
                "Điểm của fold validation được gán theo **core point gần nhất trong bán kính eps**, xa hơn là nhiễu.")
    g = st.session_state.dbgrid
    with st.form("db_grid_form"):
        c1, c2, c3 = st.columns(3)
        s_feat = c1.multiselect("Bộ thuộc tính", list(md.FEATURE_SETS), g["features"])
        s_scal = c1.multiselect("Chuẩn hoá", list(md.SCALERS), g["scaler"])
        s_ms = c2.multiselect("minPts", [3, 5, 8, 10, 15, 20, 30, 40, 60], g["min_samples"])
        s_es = c2.multiselect("eps_scale", [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0], g["eps_scale"])
        s_samp = c3.multiselect("Cách chọn mẫu", list(md.SAMPLING_VI), g["sampling"])
        s_frac = c3.multiselect("Tỉ lệ mẫu", [0.3, 0.6, 1.0], g["frac"])
        n_cfg = max(1, len(s_feat) * len(s_scal) * len(s_ms) * len(s_es) * len(s_samp) * len(s_frac))
        c3.caption(f"{n_cfg} cấu hình × {st.session_state.n_splits} fold")
        if st.form_submit_button("▶️ Chạy Grid Search CV (DBSCAN)", type="primary"):
            if not all([s_feat, s_scal, s_ms, s_es, s_samp, s_frac]):
                st.error("Mỗi tham số cần ít nhất 1 giá trị.")
            else:
                st.session_state.dbgrid = {"features": s_feat, "scaler": s_scal, "min_samples": sorted(s_ms),
                                           "eps_scale": sorted(s_es), "sampling": s_samp, "frac": sorted(s_frac)}
                st.rerun()
    dgs, db_elapsed = get_db_grid(db_gkey())
    res = dbm.db_results_frame(dgs)
    best = dgs.best_estimator_
    st.success("**Cấu hình DBSCAN tốt nhất (theo F1-macro CV):** " +
               ", ".join(f"`{k}` = **{v}**" for k, v in dgs.best_params_.items()) +
               f" → eps = {best.eps_:.3f}, {best.n_clusters_} cụm trên train.")
    cols = ["chosen", "features", "scaler", "min_samples", "eps_scale", "F1_macro", "F1_macro_std", "ARI",
            "Late_recall", "Noise_pct", "N_clusters"]
    st.dataframe(res.head(10)[cols].style.format({c: "{:.3f}" for c in cols[5:]}).apply(
        lambda r: ["background-color:#dcfce7" if r["chosen"] else "" for _ in r], axis=1),
        hide_index=True, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    for col, metric, scale, title in [(c1, "F1_macro", "Greens", "F1-macro (max)"),
                                      (c2, "N_clusters", "Blues", "Số cụm (TB)"),
                                      (c3, "Noise_pct", "Reds", "% nhiễu (TB)")]:
        agg = "max" if metric == "F1_macro" else "mean"
        piv = res.groupby(["min_samples", "eps_scale"])[metric].agg(agg).unstack()
        col.plotly_chart(px.imshow(piv, text_auto=".2f", color_continuous_scale=scale, aspect="auto",
                                   title=f"{title}: minPts × eps_scale"), use_container_width=True)

    stage("critique")
    debate("DBSCAN không cần chọn k nên khách quan hơn K-Means.",
           f"Không chọn k nhưng phải chọn **eps và minPts**, và kết quả **rất nhạy** với chúng: trong lưới trên, "
           f"F1 dao động từ **{res['F1_macro'].min():.2f} đến {res['F1_macro'].max():.2f}**, số cụm từ "
           f"**{res['N_clusters'].min():.0f} đến {res['N_clusters'].max():.0f}**. Đề lại yêu cầu **đúng 3 nhóm** — "
           "với DBSCAN ta không ép được điều đó, chỉ có thể dò tham số và hy vọng.")
    Zb = best.transform_features(X)
    dd = dbm.density_by_group(Zb, [VI[g_] for g_ in y], k=best.min_samples)
    med = dd.groupby("group")["kdist"].median()
    ratio = med["Đi trễ"] / med["Đúng giờ"]
    debate("DBSCAN chính xác hơn K-Means, chỉ là chậm hơn.",
           f"DBSCAN giả định các cụm có **mật độ tương đương**. Ở đây khoảng cách tới láng giềng thứ {best.min_samples} "
           f"(trung vị) của nhóm trễ lớn gấp **{ratio:.1f} lần** nhóm đúng giờ — nhóm trễ **thưa hơn nhiều**. Một eps duy nhất "
           "không thể vừa đủ nhỏ để tách sớm/đúng giờ (vốn liền nhau), vừa đủ lớn để giữ nhóm trễ: hoặc nhóm trễ thành "
           "nhiễu, hoặc mọi thứ dính vào nhau (xem biểu đồ bên dưới).")
    fig = px.box(dd, x="group", y="kdist", color="group", color_discrete_map=COLORS, log_y=True, points=False,
                 category_orders={"group": GROUP_ORDER},
                 labels={"kdist": f"khoảng cách tới láng giềng thứ {best.min_samples} (log)", "group": ""},
                 title="Mật độ cục bộ theo nhóm thật — đường đỏ là eps tốt nhất")
    fig.add_hline(y=best.eps_, line_dash="dash", line_color="red")
    st.plotly_chart(fig.update_layout(showlegend=False), use_container_width=True)
    st.caption("Điểm có hộp nằm phía trên đường đỏ → phần lớn không đủ láng giềng trong eps → không phải core → dễ thành nhiễu.")

    debate("DBSCAN chống nhiễu tốt nên luôn dùng được.",
           "**Phần này đúng** — đây là thế mạnh thật của DBSCAN, nhưng ở vai trò **phát hiện ngoại lai** chứ không phải "
           "gom 3 nhóm. Minh chứng: chạy DBSCAN trên **dữ liệu thô chưa làm sạch** và xem điểm nhiễu có trùng với các dòng "
           "chứa giá trị vô lý/sai đơn vị không (ta biết chính xác vì tự chèn lỗi).")
    q = st.slider("eps = phân vị của k-distance (tỉ lệ dữ liệu coi là 'bình thường')", 0.60, 0.98, 0.85, 0.01)
    od, n_bad, t_od, knee_raw, eps_raw = outlier_cached(CFG, q)
    st.dataframe(od.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}"}), hide_index=True, use_container_width=True)
    dbr, iqr = od.iloc[0], od.iloc[1]
    st.markdown(
        f"Có **{n_bad}** dòng thật sự chứa giá trị vô lý. DBSCAN: precision **{dbr['Precision']:.2f}**, recall "
        f"**{dbr['Recall']:.2f}**, đánh dấu nhầm **{dbr['SV trễ hợp lệ bị đánh dấu nhầm']}** SV trễ hợp lệ. "
        f"IQR×1.5: precision **{iqr['Precision']:.2f}**, recall **{iqr['Recall']:.2f}**, nhưng đánh dấu nhầm "
        f"**{iqr['SV trễ hợp lệ bị đánh dấu nhầm']}** SV trễ hợp lệ — đúng vấn đề đã thấy ở trang 2. DBSCAN xét "
        "**nhiều chiều cùng lúc** nên phân biệt được *\"đến trễ nhưng mọi thứ khác hợp lý\"* với *\"dữ liệu hỏng\"*.\n\n"
        f"Lưu ý: ở đây **không** dùng điểm khuỷu ({knee_raw:.1f}) vì các giá trị cực đoan (999 phút…) kéo dài đuôi "
        f"k-distance làm khuỷu lệch rất xa; dùng phân vị cho eps = {eps_raw:.2f}.")

    stage("proof", "**Đối đầu trên tập test** — cả hai đều là cấu hình tốt nhất từ Grid Search CV của chính nó:")
    kgs, k_elapsed = get_grid(gkey())
    km = kgs.best_estimator_
    Xte, yte = D["Xte"], D["yte"]
    ks = {k: f(km, Xte, yte) for k, f in md.SCORERS.items()}
    ds = dbm.db_scorer(best, Xte, yte)
    h2h = pd.DataFrame([
        {"Mô hình": "K-Means (k=3)", "F1_macro": ks["F1_macro"], "ARI": ks["ARI"], "Late_recall": ks["Late_recall"],
         "% nhiễu": 0.0, "Số cụm": 3, "Thời gian Grid Search (s)": k_elapsed, "Thời gian 1 lần huấn luyện (s)": kgs.refit_time_},
        {"Mô hình": "DBSCAN", "F1_macro": ds["F1_macro"], "ARI": ds["ARI"], "Late_recall": ds["Late_recall"],
         "% nhiễu": ds["Noise_pct"], "Số cụm": int(ds["N_clusters"]), "Thời gian Grid Search (s)": db_elapsed,
         "Thời gian 1 lần huấn luyện (s)": dgs.refit_time_},
    ])
    c1, c2 = st.columns([3, 2])
    c1.dataframe(h2h.style.format({"F1_macro": "{:.3f}", "ARI": "{:.3f}", "Late_recall": "{:.3f}", "% nhiễu": "{:.1f}",
                                   "Thời gian Grid Search (s)": "{:.1f}", "Thời gian 1 lần huấn luyện (s)": "{:.3f}"}),
                 hide_index=True, use_container_width=True)
    long = h2h.melt(id_vars="Mô hình", value_vars=["F1_macro", "ARI", "Late_recall"], var_name="Chỉ số", value_name="Giá trị")
    c2.plotly_chart(px.bar(long, x="Chỉ số", y="Giá trị", color="Mô hình", barmode="group", text_auto=".2f",
                           title="K-Means vs DBSCAN (test)"), use_container_width=True)
    km_better = ks["F1_macro"] > ds["F1_macro"]
    (st.success if km_better else st.warning)(
        ("**Kết luận:** với bài toán 3 nhóm liền nhau, mật độ khác nhau, **K-Means tốt hơn rõ rệt** "
         f"(F1 {ks['F1_macro']:.2f} vs {ds['F1_macro']:.2f}; recall nhóm trễ {ks['Late_recall']:.2f} vs {ds['Late_recall']:.2f}). "
         "\"DBSCAN chính xác hơn\" **không đúng ở đây**."
         if km_better else
         f"**Kết luận:** trên cấu hình dữ liệu hiện tại DBSCAN đạt F1 {ds['F1_macro']:.2f} ≥ K-Means {ks['F1_macro']:.2f} — "
         "hãy xem lại phần mật độ: có thể các nhóm đang tách biệt và đồng đều mật độ hơn.") +
        "\n\n**Đề xuất kết hợp:** dùng **DBSCAN để phát hiện/loại dữ liệu hỏng** (thế mạnh của nó), rồi dùng "
        "**K-Means để gom 3 nhóm** (đúng giả định của bài toán).")

    st.markdown("#### 💾 Lưu `learnt_dbscan.pkl`")
    payload = {"model": best, "best_params": dgs.best_params_, "eps": best.eps_, "n_clusters_train": best.n_clusters_,
               "label_map": {int(c): (dg.GROUPS[g_] if g_ >= 0 else "noise") for c, g_ in best.cluster_to_group_.items()},
               "features": best.features_, "test_metrics": ds, "data_config": dict(CFG)}
    c1, c2 = st.columns([1, 2])
    if c1.button("💾 Lưu learnt_dbscan.pkl"):
        st.session_state["db_pkl_blob"] = md.save_learnt(DB_LEARNT_PATH, payload)
        c1.success(f"Đã lưu {DB_LEARNT_PATH.name}")
    if "db_pkl_blob" in st.session_state:
        c1.download_button("⬇️ Tải learnt_dbscan.pkl", st.session_state["db_pkl_blob"], "learnt_dbscan.pkl")
    c2.caption("Lưu cả pipeline (bộ làm sạch + scaler + core points + eps) để dự đoán điểm mới: "
               "`learnt['model'].predict_group(df)` → 0 sớm / 1 đúng giờ / 2 trễ / -1 nhiễu.")


@st.cache_resource(show_spinner="Đang chạy các thuật toán gom cụm…")
def compare_cached(gk):
    gs, _ = get_grid(gk)
    D_ = load_data(gk[0])
    return md.compare_models(D_["X"], D_["y"], gs.best_estimator_, dict(gk[0])["seed"])


@st.cache_data(show_spinner="Đang đo thời gian theo kích thước dữ liệu…")
def scaling_cached(sizes, ratios, features, seed):
    return md.scaling_benchmark(list(sizes), ratios, features, seed)


def page_compare():
    st.title("6. So sánh với các mô hình khác")
    gs, _ = get_grid(gkey())
    stage("idea", "K-Means không phải lựa chọn duy nhất. So sánh trên **cùng dữ liệu, cùng tiền xử lý, cùng bộ thuộc "
                  "tính** với: MiniBatch K-Means, Gaussian Mixture (GMM), Agglomerative (Ward), DBSCAN, HDBSCAN.")
    stage("lead", "Với thuật toán không cho ra đúng 3 cụm (DBSCAN/HDBSCAN), F1 được tính bằng ghép cặp Hungarian "
                  "1–1 giữa cụm và nhãn; cụm thừa và điểm nhiễu (-1) bị tính là sai. *DBSCAN dò theo nhãn* được phép "
                  "nhìn đáp án để chọn eps/minPts — đó là **cận trên lạc quan**, ngoài đời không làm được.")
    res, labels, Z, kdist, eps = compare_cached(gkey())
    st.dataframe(res.style.format({"Thời gian (s)": "{:.3f}", "% nhiễu": "{:.1f}", "ARI": "{:.3f}", "NMI": "{:.3f}",
                                   "F1_macro": "{:.3f}", "Late_recall": "{:.3f}", "Silhouette": "{:.3f}"})
                 .highlight_max(subset=["ARI", "F1_macro", "Late_recall"], color="#bbf7d0"),
                 hide_index=True, use_container_width=True)
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.bar(res, x="F1_macro", y="Mô hình", orientation="h", text_auto=".3f", title="F1-macro (cao = tốt)")
                    .update_layout(yaxis={"categoryorder": "total ascending"}), use_container_width=True)
    c2.plotly_chart(px.scatter(res, x="Thời gian (s)", y="ARI", text="Mô hình", log_x=True,
                               title="Đánh đổi thời gian ↔ chất lượng (ARI)").update_traces(textposition="top center"),
                    use_container_width=True)
    pick = st.selectbox("Xem kết quả gom cụm của", list(labels))
    lab = labels[pick]
    Xc = pp.StudentCleaner().fit(D["X"]).transform(D["X"])
    dfp = pd.DataFrame({"arrival_minutes": Xc["arrival_minutes"], "wake_up_hour": Xc["wake_up_hour"],
                        "Cụm": np.where(lab == -1, "Nhiễu (-1)", lab.astype(str)),
                        "Nhãn thật": [VI[g] for g in D["y"]]})
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(dfp, x="arrival_minutes", y="wake_up_hour", color="Cụm", opacity=0.6, title=pick),
                    use_container_width=True)
    c2.plotly_chart(px.scatter(dfp, x="arrival_minutes", y="wake_up_hour", color="Nhãn thật", opacity=0.6,
                               color_discrete_map=COLORS, title="Nhãn thật (ẩn)"), use_container_width=True)

    stage("critique")
    km = res[res["Mô hình"].str.startswith("K-Means (toàn")].iloc[0]
    db = res[res["Mô hình"].str.startswith("DBSCAN (eps tự")].iloc[0]
    dbt = res[res["Mô hình"].str.startswith("DBSCAN dò")].iloc[0]
    debate("Ưu tiên **tốc độ** → K-Means; ưu tiên **độ chính xác** nhưng chấp nhận chậm → DBSCAN.",
           "Vế đầu **đúng**: K-Means O(n·k·d·số vòng lặp), gần tuyến tính; DBSCAN O(n log n) khi có chỉ mục tốt ở "
           "chiều thấp, xấu nhất O(n²).\n\n"
           "Vế sau **không đúng như một quy tắc chung**. DBSCAN không *chính xác hơn*, nó dựa trên **giả định khác**: "
           "cụm là vùng **mật độ đồng đều** ngăn cách bởi vùng thưa, hình dạng tuỳ ý. Bài toán này lại có:\n"
           "1. Các nhóm **chạm nhau** (sớm–đúng giờ–trễ là một dải liên tục) → không có vùng thưa để tách.\n"
           "2. **Mật độ khác nhau**: nhóm đúng giờ dày đặc, nhóm trễ thưa và dao động lớn → một eps duy nhất hoặc "
           "gộp tất cả, hoặc biến nhóm trễ thành *nhiễu*.\n"
           "3. Đề yêu cầu **đúng 3 nhóm**, DBSCAN không cho chọn k; số cụm phụ thuộc mạnh vào eps/minPts.\n"
           "4. Sau chuẩn hoá, các nhóm gần dạng cầu/Gauss → đúng giả định của K-Means/GMM.\n\n"
           "DBSCAN **có ưu điểm thật**: cụm hình dạng bất kỳ và tự đánh dấu ngoại lai (dùng được như bước làm sạch). "
           "**Chọn thuật toán theo hình dạng/mật độ dữ liệu và yêu cầu bài toán**, không theo khẩu hiệu \"nhanh vs chính xác\".")
    stage("proof")
    st.markdown(
        f"- K-Means: F1 **{km['F1_macro']:.3f}**, ARI {km['ARI']:.3f}, {km['Thời gian (s)']:.3f}s.\n"
        f"- DBSCAN eps tự động (k-distance): **{int(db['Số cụm'])} cụm**, {db['% nhiễu']:.1f}% nhiễu, F1 **{db['F1_macro']:.3f}**.\n"
        f"- Kể cả DBSCAN *được nhìn đáp án*: F1 {dbt['F1_macro']:.3f}, recall nhóm trễ {dbt['Late_recall']:.3f}, "
        f"mất {dbt['Thời gian (s)']:.2f}s để dò tham số.\n\n"
        + ("→ Trên dữ liệu này, DBSCAN **vừa chậm hơn vừa kém hơn** — giả thuyết H5 bị **bác bỏ** cho bài toán này."
           if dbt["F1_macro"] < km["F1_macro"] else
           "→ DBSCAN (khi dò đúng tham số) đạt chất lượng tương đương/hơn nhưng phải trả giá bằng thời gian dò tham số và cần nhãn để dò."))
    with st.expander("Đồ thị k-distance dùng để chọn eps cho DBSCAN"):
        st.plotly_chart(px.line(y=kdist, labels={"x": "điểm (đã sắp xếp)", "y": "khoảng cách tới láng giềng thứ minPts"},
                                title=f"k-distance — điểm khuỷu ≈ eps = {eps:.2f}").add_hline(y=eps, line_dash="dash"),
                        use_container_width=True)

    st.markdown("#### Tốc độ khi dữ liệu lớn dần")
    sizes = st.multiselect("Kích thước n", [1000, 2500, 5000, 10000, 20000, 40000], [1000, 2500, 5000, 10000, 20000])
    if st.button("▶️ Chạy benchmark tốc độ"):
        st.session_state["bench"] = tuple(sorted(sizes))
    if "bench" in st.session_state:
        bm = scaling_cached(st.session_state["bench"], dict(CFG)["ratios"], gs.best_params_["features"], seed)
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(bm, x="n", y="Thời gian (s)", color="Thuật toán", markers=True, log_x=True, log_y=True,
                                title="Thời gian huấn luyện (log–log)"), use_container_width=True)
        c2.plotly_chart(px.line(bm, x="n", y="ARI", color="Thuật toán", markers=True, log_x=True,
                                title="Chất lượng (ARI) theo n"), use_container_width=True)
        st.caption("Agglomerative bị bỏ qua khi n > 6 000 và HDBSCAN khi n > 12 000 vì chi phí O(n²) bộ nhớ/thời gian.")


@st.cache_data(show_spinner="Đang đo độ ổn định bằng bootstrap…")
def stability_cached(gk, B):
    gs, _ = get_grid(gk)
    D_ = load_data(gk[0])
    return md.stability(gs.best_params_, D_["Xtr"], D_["Xte"], B, dict(gk[0])["seed"])


def page_evaluate():
    st.title("7. Đánh giá: mô hình học được gì và có đáng tin cậy không?")
    gs, _ = get_grid(gkey())
    est = gs.best_estimator_
    Xte, yte = D["Xte"], D["yte"]
    stage("idea", "Một kết quả gom cụm *đáng tin* phải thoả **4 điều**: (1) đúng với cấu trúc thật trên dữ liệu "
                  "**chưa thấy** (test), (2) có cấu trúc cụm rõ ràng (chỉ số nội), (3) **ổn định** khi dữ liệu/seed "
                  "thay đổi, (4) **khái quát hoá** — điểm CV ≈ điểm test.")
    stage("lead", "Đánh giá ngoài (external, dùng nhãn ẩn): ARI, NMI, F1-macro, recall từng nhóm. Đánh giá nội "
                  "(internal, không cần nhãn — thứ duy nhất có ngoài đời): Silhouette, Davies–Bouldin. Độ ổn định: "
                  "huấn luyện lại trên các mẫu **bootstrap** của train và với các seed khác, đo ARI giữa các lần dự đoán trên test.")
    pred = est.predict_group(Xte)
    yc = md.y_codes(yte)
    scores = {k: f(est, Xte, yte) for k, f in md.SCORERS.items()}
    cv_score = md.best_cv_score(gs)
    cols = st.columns(5)
    for c, (k, v) in zip(cols, scores.items()):
        c.metric(f"{k} (test)", fmt(v))
    c1, c2 = st.columns(2)
    cm = confusion_matrix(yc, pred, labels=[0, 1, 2])
    c1.plotly_chart(px.imshow(cm, x=GROUP_ORDER, y=GROUP_ORDER, text_auto=True, color_continuous_scale="Blues",
                              labels={"x": "Dự đoán", "y": "Thật"}, title="Ma trận nhầm lẫn (test)"),
                    use_container_width=True)
    rep = pd.DataFrame(classification_report(yc, pred, labels=[0, 1, 2], target_names=GROUP_ORDER,
                                             output_dict=True, zero_division=0)).T
    c2.dataframe(rep.style.format("{:.3f}"), use_container_width=True)
    Z = est.transform_features(Xte)
    lab = est.predict(Xte)
    c2.metric("Davies–Bouldin (thấp = tốt)", fmt(davies_bouldin_score(Z, lab)))

    st.markdown("#### Độ ổn định (Bootstrap — đúng vai trò của nó)")
    B = st.slider("Số lần lặp", 10, 60, 20, 10)
    boot, seeds = stability_cached(gkey(), B)
    c1, c2 = st.columns(2)
    sd = pd.DataFrame({"ARI": np.r_[boot, seeds], "Nguồn dao động": ["Bootstrap dữ liệu train"] * len(boot) +
                       ["Đổi seed (khởi tạo + rút mẫu)"] * len(seeds)})
    c1.plotly_chart(px.box(sd, x="Nguồn dao động", y="ARI", points="all", title="ARI giữa mô hình gốc và mô hình huấn luyện lại"),
                    use_container_width=True)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    f1_boot = []
    rng = np.random.default_rng(seed)
    for _ in range(200):
        i = rng.integers(0, len(yc), len(yc))
        f1_boot.append(f1_score(yc[i], pred[i], average="macro", labels=[0, 1, 2], zero_division=0))
    f_lo, f_hi = np.percentile(f1_boot, [2.5, 97.5])
    c2.metric("ARI ổn định (bootstrap), TB", fmt(boot.mean()), f"KTC 95%: [{lo:.3f}, {hi:.3f}]", delta_color="off")
    c2.metric("ARI ổn định (đổi seed), TB", fmt(seeds.mean()))
    c2.metric("F1-macro test, KTC 95% (bootstrap tập test)", f"[{f_lo:.3f}, {f_hi:.3f}]")

    stage("critique")
    debate("Silhouette cao là mô hình tốt.",
           f"Silhouette đo độ *tách biệt hình học*, không đo *đúng nghĩa*. Ở trang 3, bộ `core` có Silhouette cao hơn "
           "`behavior` dù F1 tương đương/thấp hơn — vì ít chiều hơn thì khoảng cách \"đẹp\" hơn. Silhouette **không so "
           "sánh được giữa các không gian thuộc tính khác nhau**; vì thế Grid Search dùng F1/ARI (có được nhờ dữ liệu giả lập).")
    debate("Dữ liệu giả lập đạt F1 cao nên ra thực tế cũng vậy.",
           "Độ tin cậy ở đây là **so với chính bộ sinh dữ liệu** do ta thiết kế (nhóm tách khá rõ, phân phối Gauss). "
           "Dữ liệu thật có thể chồng lấn hơn, có thêm yếu tố (thời tiết, kẹt xe, môn học…), và **không có nhãn** để "
           "tính F1 — khi đó chỉ còn chỉ số nội + độ ổn định + đánh giá của chuyên gia trên một mẫu nhỏ được gán nhãn tay.")

    stage("proof", "**Bảng kiểm độ tin cậy** (ngưỡng do ta đặt trước, không chỉnh theo kết quả):")
    checks = [
        ("Đúng cấu trúc thật trên test", f"F1-macro = {scores['F1_macro']:.3f}", scores["F1_macro"] >= 0.80, "≥ 0.80"),
        ("Không bỏ rơi nhóm thiểu số", f"Recall nhóm trễ = {scores['Late_recall']:.3f}", scores["Late_recall"] >= 0.70, "≥ 0.70"),
        ("Khái quát hoá (CV ≈ test)", f"|CV − test| = {abs(cv_score - scores[gs.metric_]):.3f}",
         abs(cv_score - scores[gs.metric_]) <= 0.05, "≤ 0.05"),
        ("Ổn định khi dữ liệu đổi", f"ARI bootstrap TB = {boot.mean():.3f}", boot.mean() >= 0.80, "≥ 0.80"),
        ("Ổn định khi seed đổi", f"ARI seed TB = {seeds.mean():.3f}", seeds.mean() >= 0.90, "≥ 0.90"),
        ("Có cấu trúc cụm thật", f"Silhouette = {scores['Silhouette']:.3f}", scores["Silhouette"] >= 0.25, "≥ 0.25"),
    ]
    ck = pd.DataFrame(checks, columns=["Tiêu chí", "Giá trị", "Đạt", "Ngưỡng"])
    ck["Đạt"] = ck["Đạt"].map({True: "✅", False: "❌"})
    st.dataframe(ck, hide_index=True, use_container_width=True)
    passed = sum(c[2] for c in checks)
    (st.success if passed == len(checks) else st.warning)(
        f"**{passed}/{len(checks)} tiêu chí đạt.** " +
        ("Kết quả học được **đáng tin cậy trong phạm vi dữ liệu giả lập**: đúng, ổn định và khái quát hoá tốt."
         if passed == len(checks) else "Cần xem lại các tiêu chí chưa đạt trước khi tin vào kết quả."))

    st.markdown("#### Mô hình học được gì từ dữ liệu?")
    cent = est.centroids_original()
    st.markdown(
        f"1. **Giờ đến trung bình là trục chính** phân biệt 3 nhóm; tâm cụm ≈ "
        f"{cent['arrival_minutes'].iloc[0]:+.1f}' / {cent['arrival_minutes'].iloc[1]:+.1f}' / {cent['arrival_minutes'].iloc[2]:+.1f}' "
        "(thật: −15' / 0' / +14').\n"
        "2. **Độ thất thường** (`arrival_std`) và **giờ thức dậy** là tín hiệu phụ: nhóm trễ vừa dậy muộn vừa đến giờ thất thường — "
        "cây surrogate dùng chúng để phân xử vùng biên giữa đúng giờ và trễ.\n"
        "3. `gpa`, `age`, `commute_min` **không giúp** gom cụm (bộ `all` kém hơn) → mô hình không \"học\" quan hệ giả.\n"
        "4. Vùng sai chủ yếu ở **biên đúng giờ ↔ trễ / sớm** (xem ma trận nhầm lẫn) — nơi các phân phối thật sự chồng lấn, "
        "không phải lỗi của thuật toán.")
    st.markdown("#### Em học được gì?")
    st.markdown(
        "- Phải **kiểm tra mất cân bằng** trước, và chọn thước đo (F1-macro, recall nhóm nhỏ) cho phù hợp.\n"
        "- Tiền xử lý **dựa trên tri thức miền** (luật hợp lệ, không xoá đuôi phân phối) quan trọng hơn áp công thức máy móc.\n"
        "- Mẫu *đại diện* nghĩa là giữ đúng đặc tính tổng thể (phân tầng), không nhất thiết là *cân bằng*.\n"
        "- **CV để chọn mô hình, Bootstrap để đo độ bất định** — hai công cụ bổ sung, không thay thế nhau.\n"
        "- Không có thuật toán \"chính xác hơn\" tuyệt đối: chọn theo **giả định khớp với dữ liệu**.\n"
        "- Mọi kết luận phải có **minh chứng định lượng** và nêu rõ **giới hạn**.")

    st.markdown("#### 🔮 Dùng learnt.pkl dự đoán cho sinh viên mới")
    model, src = est, "mô hình trong phiên hiện tại (chưa lưu learnt.pkl)"
    if LEARNT_PATH.exists():
        try:
            with open(LEARNT_PATH, "rb") as f:
                learnt = pickle.load(f)
            model, src = learnt["model"], f"learnt.pkl (lưu lúc {learnt.get('created_at', '?')})"
        except Exception as ex:  # noqa: BLE001
            st.warning(f"Không đọc được learnt.pkl: {ex}")
    st.caption(f"Nguồn mô hình: {src}")
    with st.form("predict"):
        c1, c2, c3 = st.columns(3)
        row = {
            "student_id": "SV_NEW",
            "arrival_minutes": c1.number_input("Đến lớp sớm(−)/trễ(+) bao nhiêu phút", -60.0, 90.0, 8.0),
            "arrival_std": c1.number_input("Độ dao động giờ đến (phút)", 0.0, 30.0, 7.0),
            "wake_up_hour": c1.number_input("Giờ thức dậy", 3.0, 12.0, 6.8, 0.05),
            "distance_km": c2.number_input("Khoảng cách (km)", 0.0, 60.0, 6.0),
            "commute_min": c2.number_input("Thời gian đi (phút)", 0.0, 240.0, 24.0),
            "sleep_hours": c2.number_input("Giờ ngủ", 2.0, 14.0, 6.2),
            "gpa": c3.number_input("GPA", 0.0, 4.0, 2.9),
            "age": c3.number_input("Tuổi", 16.0, 45.0, 20.0),
            "transport": c3.selectbox("Phương tiện", dg.TRANSPORTS, index=3),
        }
        if st.form_submit_button("Dự đoán"):
            new = pd.DataFrame([row])
            g = int(model.predict_group(new)[0])
            Zn = model.transform_features(new)
            d = np.linalg.norm(model.kmeans_.cluster_centers_ - Zn, axis=1)
            st.success(f"Sinh viên này thuộc nhóm: **{VI[dg.GROUPS[g]]}**")
            st.dataframe(pd.DataFrame({"Nhóm": [VI[dg.GROUPS[model.cluster_to_group_[c]]] for c in range(len(d))],
                                       "Khoảng cách tới tâm (đã chuẩn hoá)": d}).sort_values(
                "Khoảng cách tới tâm (đã chuẩn hoá)"), hide_index=True)


{
    PAGES[0]: page_overview, PAGES[1]: page_data, PAGES[2]: page_preprocess, PAGES[3]: page_grid,
    PAGES[4]: page_kmeans, PAGES[5]: page_dbscan, PAGES[6]: page_compare, PAGES[7]: page_evaluate,
}[page]()
