"""EduPro - Student Segmentation and Personalized Course Recommendation System (Streamlit app)."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import silhouette_score

import utils as U

st.set_page_config(page_title="EduPro Learner Segmentation", page_icon="🎓", layout="wide")

PALETTE = ["#3B82F6", "#22D3EE", "#818CF8", "#60A5FA", "#2DD4BF", "#A78BFA", "#38BDF8", "#93C5FD",
           "#F59E0B", "#F472B6", "#34D399", "#FB7185"]
PAGES = ["📊 Dashboard Overview", "👤 Learner Profile Explorer", "🤖 ML Segmentation",
         "🎯 Course Recommendations", "📈 Learning Behaviour", "👩‍🏫 Teacher & Course Analysis"]
FEATURE_LABELS = {"total_courses": "Courses enrolled", "categories_explored": "Categories explored",
                  "free_ratio": "Free-course share", "depth_index": "Learning depth",
                  "avg_spend": "Avg spend", "category_focus": "Category focus", "avg_rating": "Avg course rating"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.kpi {background: linear-gradient(135deg, #14234f, #0e1a3b); border: 1px solid #1d3a8a;
      border-left: 4px solid #3B82F6; border-radius: 12px; padding: 14px 18px; margin-bottom: 10px;}
.kpi .l {font-size: 12px; color: #8fa6d9; text-transform: uppercase; letter-spacing: .6px;}
.kpi .v {font-size: 28px; font-weight: 700; color: #E6EDFF; line-height: 1.25;}
.kpi .s {font-size: 12px; color: #6f8bc4;}
.card {background: #111C3D; border: 1px solid #1d3a8a; border-radius: 12px; padding: 14px 18px; margin-bottom: 12px;}
.card h4 {margin: 0 0 6px 0; color: #E6EDFF;}
.tag {display: inline-block; background: #1e3a8a; color: #bfdbfe; border-radius: 10px; padding: 2px 10px;
      font-size: 12px; margin-right: 6px;}
.tag.fb {background: #78350f; color: #fde68a;}
.reason {color: #9db4e6; font-size: 14px; margin-top: 6px;}
h1, h2, h3 {color: #E6EDFF;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ small helpers
def show(fig, height=380):
    """Apply the dark-navy style to a Plotly figure and display it."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      height=height, margin=dict(l=10, r=10, t=50, b=10), colorway=PALETTE,
                      font=dict(color="#E6EDFF"), title_font_size=16)
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:  # newest Streamlit versions use width="stretch"
        st.plotly_chart(fig, width="stretch")


def table(df, height=None):
    try:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
    except TypeError:
        st.dataframe(df, width="stretch", hide_index=True, height=height)


def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div>'
                 f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def download(df, filename, label="⬇ Download CSV"):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"), file_name=filename, mime="text/csv")


def counts(series, name):
    """value_counts as a tidy DataFrame with columns [name, 'Count']."""
    return series.value_counts().rename_axis(name).reset_index(name="Count")


def money(x):
    return f"${x:,.2f}"


# ------------------------------------------------------------------ cached data + model
@st.cache_data(show_spinner="Loading and preparing the Excel data...")
def load_all():
    data, report = U.load_data()
    master = U.build_master(data)
    feats = U.build_learner_features(master, data["users"])
    X, cols = U.preprocess(feats)
    mask = U.fit_mask(feats)
    scan = U.scan_k(X[mask])
    return data, report, master, feats, X, mask, scan


@st.cache_data(show_spinner="Running K-Means clustering...")
def run_model(k):
    data, report, master, feats, X, mask, scan = load_all()
    labels, km = U.fit_kmeans(X, mask, k)
    names, prof, z = U.name_segments(feats, labels)
    names = {int(c): n for c, n in names.items()}
    seg_table = U.segment_summary(feats, labels, names)
    learners = feats.assign(cluster=labels)
    learners["Segment"] = learners["cluster"].map(names)
    hier = U.hierarchical_check(X[mask], k, labels[mask])
    pca, var = U.pca_2d(X)
    pca["Segment"] = learners["Segment"].values
    pca["UserID"] = learners["UserID"].values
    z = z.rename(index=names, columns=FEATURE_LABELS)
    return {"labels": labels, "names": names, "learners": learners, "seg_table": seg_table,
            "sil_fit": silhouette_score(X[mask], labels[mask]), "sil_all": silhouette_score(X, labels),
            "hier": hier, "intra": {names[c]: v for c, v in U.intra_cluster_similarity(X, labels).items()},
            "pca": pca, "pca_var": var, "z": z}


@st.cache_resource(show_spinner="Building the recommender...")
def get_recommender(k):
    data, report, master, feats, X, mask, scan = load_all()
    R = U.interaction_matrix(data["transactions"], feats["UserID"], data["courses"])
    return U.Recommender(R, run_model(k)["labels"], data["courses"])


@st.cache_data(show_spinner="Evaluating recommendations (Precision@K)...")
def run_eval(k):
    data, report, master, feats, X, mask, scan = load_all()
    return U.evaluate_recommender(data["transactions"], feats["UserID"], data["courses"], run_model(k)["labels"])


data, report, master, feats, X, mask, scan = load_all()
courses, teachers, users = data["courses"], data["teachers"], data["users"]
suggested_k = U.best_k(scan)

# ------------------------------------------------------------------ sidebar: navigation + filters
FILTER_KEYS = ["f_seg", "f_cat", "f_lvl", "f_type"]


def reset_filters():
    for key in FILTER_KEYS:
        st.session_state[key] = []
    st.session_state["k_sel"] = suggested_k


if "k_sel" not in st.session_state:
    st.session_state["k_sel"] = suggested_k

with st.sidebar:
    st.markdown("## 🎓 EduPro")
    st.caption("Learner Segmentation & Course Recommendation")
    page = st.radio("Navigation", PAGES, label_visibility="collapsed")
    st.divider()
    st.markdown("**Model**")
    k = st.slider("Number of segments (K)", 3, 8, key="k_sel",
                  help="Default = K with the best Silhouette Score between 4 and 8.")
    st.caption(f"Suggested K (best silhouette): **{suggested_k}**")
    model = run_model(k)
    if st.session_state.get("last_k") != k:  # segment names change with K, so clear the old selection
        st.session_state["f_seg"] = []
        st.session_state["last_k"] = k
    names = model["names"]
    learners = model["learners"]

    st.divider()
    st.markdown("**Filters**")
    sel_seg = st.multiselect("Learner segment", sorted(names.values()), key="f_seg")
    sel_cat = st.multiselect("Course category", sorted(courses["CourseCategory"].unique()), key="f_cat")
    sel_lvl = st.multiselect("Course level", U.LEVELS, key="f_lvl")
    sel_type = st.multiselect("Course type", sorted(courses["CourseType"].unique()), key="f_type")
    st.button("↺ Reset filters", on_click=reset_filters)
    st.caption("Country/region filters are not available because the dataset has no such field.")

    st.divider()
    pool = learners[learners["Segment"].isin(sel_seg)] if sel_seg else learners
    ids = pool["UserID"].tolist()
    if st.session_state.get("learner") not in ids:
        st.session_state["learner"] = ids[0]
    name_map = dict(zip(learners["UserID"], learners["UserName"]))
    uid = st.selectbox("Learner (UserID)", ids, key="learner", format_func=lambda u: f"{u} · {name_map[u]}")

# master table (one row per enrolment) with the learner's segment attached
master_seg = master.merge(learners[["UserID", "Segment"]], on="UserID", how="left")
any_filter = bool(sel_seg or sel_cat or sel_lvl or sel_type)


def course_filter(df):
    if sel_cat:
        df = df[df["CourseCategory"].isin(sel_cat)]
    if sel_lvl:
        df = df[df["CourseLevel"].isin(sel_lvl)]
    if sel_type:
        df = df[df["CourseType"].isin(sel_type)]
    return df


mf = course_filter(master_seg)
if sel_seg:
    mf = mf[mf["Segment"].isin(sel_seg)]
lf = learners[learners["UserID"].isin(mf["UserID"].unique())]  # learners that remain after filtering


def no_data():
    st.warning("No records match the selected filters. Use **Reset filters** in the sidebar.")


# ================================================================== PAGE 1: DASHBOARD
def page_dashboard():
    st.title("📊 Dashboard Overview")
    st.caption("Platform summary built from the Users, Teachers, Courses and Transactions sheets "
               "(sidebar filters apply to every chart).")
    if mf.empty:
        return no_data()
    c = st.columns(5)
    kpi(c[0], "Total learners", f"{len(lf):,}")
    kpi(c[1], "Total teachers", f"{mf['TeacherID'].nunique() if any_filter else len(teachers):,}")
    kpi(c[2], "Total courses", f"{mf['CourseID'].nunique() if any_filter else len(courses):,}")
    kpi(c[3], "Transactions", f"{len(mf):,}", f"Revenue {money(mf['Amount'].sum())}")
    kpi(c[4], "Learner segments", f"{mf['Segment'].nunique()}", f"K-Means, K = {k}")

    a, b = st.columns(2)
    with a:
        show(px.pie(counts(lf["Segment"], "Segment"), names="Segment", values="Count", hole=0.5,
                    title="Learner distribution by segment"))
    with b:
        show(px.histogram(lf, x="Age", color="Gender", nbins=20, barmode="group",
                          title="Learner age distribution by gender"))
    a, b = st.columns([3, 2])
    with a:
        cat = counts(mf["CourseCategory"], "Category").sort_values("Count")
        show(px.bar(cat, x="Count", y="Category", orientation="h", title="Enrolments by course category"), 420)
    with b:
        show(px.pie(counts(mf["CourseType"], "Type"), names="Type", values="Count", hole=0.5,
                    title="Free vs Paid enrolments"), 200)
        show(px.pie(counts(mf["CourseLevel"], "Level"), names="Level", values="Count", hole=0.5,
                    title="Enrolments by course level"), 200)

    st.subheader("💡 Platform insights")
    for line in platform_insights(mf):
        st.markdown(f"- {line}")
    with st.expander("Data quality report (original Excel file is never modified)"):
        st.write(f"Rows loaded: {report['rows']}")
        st.write(f"Duplicate records removed: {report['duplicate_rows_removed']}")
        st.write(f"Missing values found: {report['missing_values_found']}")
        st.write(f"Transactions with unmatched UserID/CourseID removed: {report['unmatched_transactions_removed']}")


def platform_insights(m):
    out = []
    cat = m["CourseCategory"].value_counts()
    out.append(f"**{cat.index[0]}** is the most popular category with {cat.iloc[0]:,} enrolments "
               f"({100 * cat.iloc[0] / len(m):.1f}% of all enrolments); **{cat.index[-1]}** is the least popular.")
    seg = m.groupby("Segment").agg(enrol=("TransactionID", "count"), learners=("UserID", "nunique"))
    top = seg["enrol"].idxmax()
    out.append(f"The most active segment is **{top}** with {seg.loc[top, 'enrol']:,} enrolments "
               f"({seg.loc[top, 'enrol'] / seg.loc[top, 'learners']:.1f} per learner).")
    free = (m["CourseType"] == "Free").mean()
    free_seg = m.groupby("Segment")["CourseType"].apply(lambda s: (s == "Free").mean())
    out.append(f"{100 * free:.1f}% of enrolments are free courses; **{free_seg.idxmax()}** has the highest free-course "
               f"share ({100 * free_seg.max():.1f}%).")
    rate = m.groupby("CourseCategory")["CourseRating"].mean()
    out.append(f"The best-rated category (average rating of enrolled courses) is **{rate.idxmax()}** "
               f"({rate.max():.2f}/5).")
    month = m.groupby(m["TransactionDate"].dt.to_period("M")).size()
    out.append(f"The busiest month was **{month.idxmax().strftime('%B %Y')}** with {month.max():,} enrolments.")
    pop = m["CourseName"].value_counts()
    out.append(f"The most enrolled course is **{pop.index[0]}** ({pop.iloc[0]:,} enrolments).")
    return out


# ================================================================== PAGE 2: LEARNER PROFILE
def spending_label(fr):
    return "Mostly free courses" if fr >= 0.9 else ("Mostly paid courses" if fr <= 0.1 else "Mixed free and paid")


def page_profile():
    st.title("👤 Learner Profile Explorer")
    row = learners[learners["UserID"] == uid].iloc[0]
    user = users[users["UserID"] == uid].iloc[0]
    hist_all = master_seg[master_seg["UserID"] == uid]
    hist = course_filter(hist_all)

    st.markdown(f'<div class="card"><h4>{row["UserName"]} <span class="tag">{uid}</span>'
                f'<span class="tag">{row["Segment"]}</span></h4>'
                f'Age {int(user["Age"])} · {user["Gender"]} · {user["Email"]}</div>', unsafe_allow_html=True)
    c = st.columns(4)
    kpi(c[0], "Courses enrolled", int(row["total_courses"]), f"{int(row['categories_explored'])} categories explored")
    kpi(c[1], "Preferred category", row["preferred_category"], f"Focus {100 * row['category_focus']:.0f}% of courses")
    kpi(c[2], "Preferred level", row["preferred_level"], f"Learning depth index {row['depth_index']:+.2f}")
    kpi(c[3], "Spending", money(row["total_spent"]),
        f"{money(row['avg_spend'])} per course · {spending_label(row['free_ratio'])}")
    st.caption(f"Average rating of enrolled courses: {row['avg_rating']:.2f}/5 · "
               f"Enrolment frequency: {row['enroll_frequency']:.2f} courses per active month · "
               f"First enrolment {row['first_date']:%d %b %Y}, latest {row['last_date']:%d %b %Y}")
    if hist.empty:
        st.info("This learner has no enrolments matching the course filters in the sidebar.")
        return
    st.caption("Charts and the history table below respect the sidebar course filters.")
    a, b, c3 = st.columns(3)
    with a:
        show(px.bar(counts(hist["CourseCategory"], "Category"), x="Category", y="Count",
                    title="Enrolments by category"), 320)
    with b:
        show(px.pie(counts(hist["CourseLevel"], "Level"), names="Level", values="Count", hole=0.5,
                    title="Course levels"), 320)
    with c3:
        sp = hist.assign(Month=hist["TransactionDate"].dt.to_period("M").astype(str)).groupby("Month")["Amount"].sum()
        show(px.bar(sp.reset_index(), x="Month", y="Amount", title="Spending by month"), 320)
    st.subheader("Enrolment history")
    cols = ["TransactionDate", "CourseName", "CourseCategory", "CourseLevel", "CourseType", "Amount",
            "CourseRating", "TeacherName", "PaymentMethod"]
    h = hist.sort_values("TransactionDate")[cols].copy()
    h["TransactionDate"] = h["TransactionDate"].dt.date
    table(h)
    download(h, f"{uid}_enrolment_history.csv")


# ================================================================== PAGE 3: ML SEGMENTATION
def page_ml():
    st.title("🤖 Machine Learning Segmentation")
    st.caption("K-Means clustering on learner-level features (numeric features scaled with StandardScaler, "
               "gender/preferred category/preferred level one-hot encoded).")
    c = st.columns(4)
    kpi(c[0], "Selected K", k, f"suggested K = {suggested_k}")
    kpi(c[1], "K-Means Silhouette", f"{model['sil_fit']:.3f}", f"all {len(learners):,} learners: {model['sil_all']:.3f}")
    kpi(c[2], "Hierarchical Silhouette", f"{model['hier']['silhouette']:.3f}", "Ward linkage, same K")
    kpi(c[3], "K-Means vs Hierarchical", f"{model['hier']['agreement_ARI']:.2f}", "agreement (Adjusted Rand Index)")
    st.info("**How it works:** learners with 2+ enrolments are used to fit the K-Means model (a learner with a single "
            "enrolment gives too little behaviour to learn from). Learners with 1 enrolment are then assigned to the "
            "nearest cluster centre, so every learner still gets a segment. Silhouette > 0.25 shows a reasonable "
            "structure; values near 0.5 or higher are strong.")

    a, b = st.columns(2)
    sc = scan.copy()
    with a:
        f1 = px.line(sc, x="K", y="Inertia", markers=True, title="Elbow Method (Inertia vs K)")
        f1.add_vline(x=k, line_dash="dash", line_color="#F59E0B")
        show(f1, 340)
    with b:
        f2 = px.line(sc, x="K", y="Silhouette", markers=True, title="Silhouette Score vs K")
        f2.add_vline(x=k, line_dash="dash", line_color="#F59E0B")
        show(f2, 340)
    st.caption("K = 2 or 3 scores higher but only splits 'heavy' and 'light' learners, which is too coarse for "
               "personalisation, so the suggested K is the best silhouette between 4 and 8 (orange line = selected K).")

    lv = learners[learners["Segment"].isin(sel_seg)] if sel_seg else learners
    a, b = st.columns(2)
    with a:
        show(px.bar(counts(lv["Segment"], "Segment"), x="Segment", y="Count", color="Segment",
                    title="Cluster distribution (learners per segment)"), 360)
    with b:
        pca = model["pca"]
        pv = pca[pca["Segment"].isin(sel_seg)] if sel_seg else pca
        show(px.scatter(pv, x="PC1", y="PC2", color="Segment", opacity=0.6, hover_data=["UserID"],
                        title=f"Clusters in 2D (PCA, {100 * model['pca_var'].sum():.0f}% of variance)"), 360)

    st.subheader("Segment profiles")
    st_tab = model["seg_table"]
    if sel_seg:
        st_tab = st_tab[st_tab["Segment"].isin(sel_seg)]
    for _, r in st_tab.iterrows():
        st.markdown(f'<div class="card"><h4>{r["Segment"]} <span class="tag">{r["Learners"]:,} learners · '
                    f'{r["Share %"]}%</span></h4>{r["Description"]}</div>', unsafe_allow_html=True)
    table(st_tab.drop(columns=["Description"]))
    z = model["z"]
    if sel_seg:
        z = z[z.index.isin(sel_seg)]
    show(px.imshow(z, text_auto=".1f", aspect="auto", color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
                   title="Segment behaviour vs platform average (red = above, blue = below, in std. units)"), 360)
    intra = pd.DataFrame({"Segment": list(model["intra"].keys()),
                          "Intra-cluster similarity": [round(v, 3) for v in model["intra"].values()]})
    st.markdown("**Intra-cluster similarity** (average cosine similarity of members to their cluster centre; "
                "higher = more consistent behaviour)")
    table(intra[intra["Segment"].isin(sel_seg)] if sel_seg else intra)
    out = lv[["UserID", "UserName", "Age", "Gender", "total_courses", "preferred_category", "preferred_level",
              "avg_spend", "free_ratio", "Segment"]]
    download(out, "learner_segments.csv", "⬇ Download learner segments (CSV)")


# ================================================================== PAGE 4: RECOMMENDATIONS
def page_recs():
    st.title("🎯 Personalized Course Recommendations")
    rec = get_recommender(k)
    row = learners[learners["UserID"] == uid].iloc[0]
    n_hist = int(row["total_courses"])
    st.markdown(f'<div class="card"><h4>{row["UserName"]} <span class="tag">{uid}</span>'
                f'<span class="tag">{row["Segment"]}</span></h4>'
                f'{n_hist} course(s) enrolled · prefers <b>{row["preferred_category"]}</b> at '
                f'<b>{row["preferred_level"]}</b> level · {spending_label(row["free_ratio"])} · '
                f'average spend {money(row["avg_spend"])} per course</div>', unsafe_allow_html=True)
    a, b = st.columns([2, 3])
    n = a.slider("Number of recommendations", 3, 10, 5)
    cold = b.checkbox("Preview fallback (cold-start) recommendations for this learner")
    res = rec.recommend(uid, names, n, sel_cat, sel_lvl, sel_type, force_fallback=cold)
    if n_hist == 1 and not cold:
        st.caption("This learner has only one enrolment, so recommendations rely mostly on the segment's behaviour.")
    if res.empty:
        st.info("No courses match the current category/level/type filters (or all of them are already taken). "
                "Try **Reset filters**.")
    else:
        if res["Type"].iloc[0] == "Fallback":
            st.warning("Fallback recommendations: popular, well-rated courses shown instead of personalised ones.")
        for i, (_, r) in enumerate(res.iterrows(), 1):
            tag_cls = "tag fb" if r["Type"] == "Fallback" else "tag"
            st.markdown(
                f'<div class="card"><h4>{i}. {r["CourseName"]}</h4>'
                f'<span class="tag">{r["CourseCategory"]}</span><span class="tag">{r["CourseLevel"]}</span>'
                f'<span class="tag">{r["CourseType"]}</span><span class="tag">{money(r["CoursePrice"])}</span>'
                f'<span class="tag">⭐ {r["CourseRating"]}</span><span class="{tag_cls}">{r["Type"]} · score {r["Score"]:.2f}</span>'
                f'<div class="reason">{r["Reason"]}</div></div>', unsafe_allow_html=True)
        out = res[["CourseID", "CourseName", "CourseCategory", "CourseLevel", "CourseType", "CoursePrice",
                   "CourseRating", "Score", "Type", "Reason"]]
        download(out, f"{uid}_recommendations.csv", "⬇ Download recommendations (CSV)")
    taken = master_seg[master_seg["UserID"] == uid]["CourseName"].tolist()
    with st.expander(f"Courses already taken by this learner (excluded from recommendations): {len(taken)}"):
        st.write(", ".join(taken))

    st.divider()
    st.subheader("Recommendation evaluation (Precision@K)")
    ev, n_eval = run_eval(k)
    st.markdown(f"**Method:** for each of the {n_eval:,} learners with 2+ enrolments, the **latest enrolment is hidden**, "
                "the recommender is rebuilt from the remaining history, and we check whether the hidden course appears "
                "in the top-K list (Hit Rate@K). Precision@K = Hit Rate@K / K because exactly one course is hidden.")
    show(px.bar(ev.assign(K=ev["K"].astype(str)), x="K", y="Hit Rate@K", color="Method", barmode="group",
                title="Hit Rate@K: hybrid model vs baselines"), 340)
    table(ev)
    st.warning("**Limitations:** precision is naturally low because only one hidden course is used as the 'correct' answer "
               "(the learner might also like other courses), the catalogue has only 60 courses, and segments were fitted "
               "on the full data. The results show only how well the model predicts a learner's *next enrolment*; "
               "no engagement or course-completion improvement has been measured.")


# ================================================================== PAGE 5: LEARNING BEHAVIOUR
def page_behaviour():
    st.title("📈 Learning Behaviour Analysis")
    if mf.empty:
        return no_data()
    a, b = st.columns(2)
    with a:
        cat = counts(mf["CourseCategory"], "Category").sort_values("Count")
        show(px.bar(cat, x="Count", y="Category", orientation="h", title="Popular course categories"), 420)
    with b:
        ct = pd.crosstab(mf["Segment"], mf["CourseCategory"])
        show(px.imshow(ct, text_auto=True, aspect="auto", color_continuous_scale="Blues",
                       title="Enrolments: segment × category"), 420)
    a, b = st.columns(2)
    with a:
        lv = pd.crosstab(mf["Segment"], mf["CourseLevel"], normalize="index").mul(100).round(1)
        lv = lv.reset_index().melt(id_vars="Segment", var_name="Level", value_name="Share %")
        show(px.bar(lv, x="Segment", y="Share %", color="Level", barmode="group",
                    category_orders={"Level": U.LEVELS}, title="Beginner vs Intermediate vs Advanced by segment"), 360)
    with b:
        ty = pd.crosstab(mf["Segment"], mf["CourseType"], normalize="index").mul(100).round(1)
        ty = ty.reset_index().melt(id_vars="Segment", var_name="Type", value_name="Share %")
        show(px.bar(ty, x="Segment", y="Share %", color="Type", barmode="stack",
                    title="Free vs Paid course preference by segment"), 360)
    a, b = st.columns(2)
    with a:
        sp = mf.groupby(["UserID", "Segment"], as_index=False)["Amount"].sum().rename(columns={"Amount": "Total spent"})
        show(px.box(sp, x="Segment", y="Total spent", color="Segment", points=False,
                    title="Learner spending by segment"), 360)
    with b:
        split = st.radio("Trend split by", ["Overall", "Segment", "Course Type"], horizontal=True)
        t = mf.assign(Month=mf["TransactionDate"].dt.to_period("M").dt.to_timestamp())
        key = {"Overall": None, "Segment": "Segment", "Course Type": "CourseType"}[split]
        tr = t.groupby(["Month"] + ([key] if key else []), as_index=False).size().rename(columns={"size": "Enrolments"})
        show(px.line(tr, x="Month", y="Enrolments", color=key, markers=True, title="Enrolment trend over time"), 320)

    st.subheader("Most popular courses inside each segment")
    top = mf.groupby(["Segment", "CourseName", "CourseCategory"]).size().reset_index(name="Enrolments")
    top = top.sort_values(["Segment", "Enrolments"], ascending=[True, False]).groupby("Segment").head(3)
    table(top)
    st.subheader("💡 Behaviour insights")
    free_seg = mf.groupby("Segment")["CourseType"].apply(lambda s: (s == "Free").mean())
    adv_seg = mf.groupby("Segment")["CourseLevel"].apply(lambda s: (s == "Advanced").mean())
    spend = sp.groupby("Segment")["Total spent"].mean()
    st.markdown(f"- **{free_seg.idxmax()}** prefers free courses the most ({100 * free_seg.max():.1f}% free), while "
                f"**{free_seg.idxmin()}** picks the fewest ({100 * free_seg.min():.1f}%).\n"
                f"- **{adv_seg.idxmax()}** has the largest share of advanced courses ({100 * adv_seg.max():.1f}%).\n"
                f"- **{spend.idxmax()}** spends the most per learner ({money(spend.max())} on average).")
    cols = ["TransactionID", "UserID", "Segment", "CourseID", "CourseName", "CourseCategory", "CourseLevel",
            "CourseType", "TransactionDate", "Amount", "TeacherID", "PaymentMethod"]
    download(mf[cols], "filtered_enrolments.csv", "⬇ Download filtered enrolment data (CSV)")


# ================================================================== PAGE 6: TEACHER & COURSE
def page_teacher_course():
    st.title("👩‍🏫 Teacher and Course Analysis")
    st.subheader("Teachers")
    a, b = st.columns(2)
    with a:
        show(px.bar(counts(teachers["Expertise"], "Expertise").sort_values("Count"), x="Count", y="Expertise",
                    orientation="h", title="Number of teachers by expertise"), 380)
    with b:
        show(px.scatter(teachers, x="YearsOfExperience", y="TeacherRating", color="Expertise",
                        hover_name="TeacherName", title="Teacher experience vs rating"), 380)
    avg_t = teachers.groupby("Expertise", as_index=False)["TeacherRating"].mean().sort_values("TeacherRating")
    show(px.bar(avg_t, x="TeacherRating", y="Expertise", orientation="h", title="Average teacher rating by expertise"), 360)

    st.subheader("Courses")
    cf = course_filter(courses)
    if cf.empty:
        no_data()
    else:
        a, b = st.columns(2)
        with a:
            avg_c = cf.groupby("CourseCategory", as_index=False)["CourseRating"].mean().sort_values("CourseRating")
            show(px.bar(avg_c, x="CourseRating", y="CourseCategory", orientation="h",
                        title="Average course rating by category"), 380)
        with b:
            show(px.box(cf, x="CourseCategory", y="CoursePrice", color="CourseType",
                        title="Course price by category"), 380)
        a, b = st.columns(2)
        with a:
            show(px.scatter(cf, x="CourseDuration", y="CoursePrice", color="CourseLevel", size="CourseRating",
                            hover_name="CourseName", title="Course duration vs price (size = rating)"), 360)
        with b:
            show(px.histogram(cf, x="CourseRating", nbins=12, color="CourseType", title="Course rating distribution"), 360)

    st.subheader("Teachers and courses connected through transactions")
    if mf.empty:
        return no_data()
    tp = mf.groupby(["TeacherID", "TeacherName", "Expertise"], as_index=False).agg(
        Enrolments=("TransactionID", "count"), Revenue=("Amount", "sum"), Avg_Course_Rating=("CourseRating", "mean"),
        Teacher_Rating=("TeacherRating", "first")).sort_values("Enrolments", ascending=False)
    tp["Revenue"] = tp["Revenue"].round(2)
    tp["Avg_Course_Rating"] = tp["Avg_Course_Rating"].round(2)
    a, b = st.columns(2)
    with a:
        show(px.bar(tp.head(10), x="TeacherName", y="Enrolments", color="Expertise",
                    title="Top 10 teachers by enrolments"), 380)
    with b:
        show(px.scatter(tp, x="Teacher_Rating", y="Enrolments", size="Avg_Course_Rating", color="Expertise",
                        hover_name="TeacherName", title="Teacher rating vs enrolments (size = avg course rating)"), 380)
    match = 100 * (mf["Expertise"] == mf["CourseCategory"]).mean()
    st.info(f"In {match:.1f}% of transactions the teacher's expertise equals the course category. "
            "The teacher is stored per transaction (not per course), so one course appears with many different teachers.")
    pc = mf.groupby(["CourseName", "CourseCategory"], as_index=False).agg(
        Enrolments=("TransactionID", "count"), Revenue=("Amount", "sum"), Course_Rating=("CourseRating", "first"),
        Teachers=("TeacherID", "nunique")).sort_values("Enrolments", ascending=False)
    st.markdown("**Top courses by enrolments**")
    table(pc.head(10))
    download(tp, "teacher_performance.csv", "⬇ Download teacher performance (CSV)")


# ------------------------------------------------------------------ router
if page == PAGES[0]:
    page_dashboard()
elif page == PAGES[1]:
    page_profile()
elif page == PAGES[2]:
    page_ml()
elif page == PAGES[3]:
    page_recs()
elif page == PAGES[4]:
    page_behaviour()
else:
    page_teacher_course()
