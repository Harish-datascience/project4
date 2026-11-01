"""utils.py - data loading, feature engineering, clustering and recommendation logic.
No Streamlit / Plotly code here, so every function can be tested on its own."""
import glob
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
LEVELS = ["Beginner", "Intermediate", "Advanced"]
RANDOM_STATE = 42


# ----------------------------------------------------------------------------
# 1. DATA LOADING AND CLEANING
# ----------------------------------------------------------------------------
def find_excel():
    """Use data/EduPro Online Platform.xlsx, or any .xlsx found inside data/."""
    preferred = os.path.join(DATA_DIR, "EduPro Online Platform.xlsx")
    if os.path.exists(preferred):
        return preferred
    files = glob.glob(os.path.join(DATA_DIR, "*.xlsx"))
    if not files:
        raise FileNotFoundError("Put the Excel file inside the 'data' folder.")
    return files[0]


def load_data(path=None):
    """Read the 4 sheets, clean them and return (dict of DataFrames, quality report)."""
    sheets = pd.read_excel(path or find_excel(), sheet_name=None, engine="openpyxl")
    users, teachers = sheets["Users"].copy(), sheets["Teachers"].copy()
    courses, trans = sheets["Courses"].copy(), sheets["Transactions"].copy()
    report = {}

    # strip spaces in text columns (IDs, names, categories)
    for df in (users, teachers, courses, trans):
        for col in df.select_dtypes(include=["object", "string"]).columns:
            df[col] = df[col].astype(str).str.strip()

    # correct data types
    trans["TransactionDate"] = pd.to_datetime(trans["TransactionDate"], errors="coerce")
    trans["Amount"] = pd.to_numeric(trans["Amount"], errors="coerce")
    for col in ["CoursePrice", "CourseDuration", "CourseRating"]:
        courses[col] = pd.to_numeric(courses[col], errors="coerce")
    users["Age"] = pd.to_numeric(users["Age"], errors="coerce")
    for col in ["Age", "YearsOfExperience", "TeacherRating"]:
        teachers[col] = pd.to_numeric(teachers[col], errors="coerce")

    # duplicates
    report["duplicate_rows_removed"] = {
        "Users": int(users.duplicated("UserID").sum()),
        "Teachers": int(teachers.duplicated("TeacherID").sum()),
        "Courses": int(courses.duplicated("CourseID").sum()),
        "Transactions": int(trans.duplicated("TransactionID").sum()
                            + trans.duplicated(["UserID", "CourseID"]).sum()),
    }
    users = users.drop_duplicates("UserID")
    teachers = teachers.drop_duplicates("TeacherID")
    courses = courses.drop_duplicates("CourseID")
    trans = trans.drop_duplicates("TransactionID").drop_duplicates(["UserID", "CourseID"])

    # missing values
    report["missing_values_found"] = int(sum(d.isna().sum().sum() for d in (users, teachers, courses, trans)))
    users["Age"] = users["Age"].fillna(users["Age"].median())
    users["Gender"] = users["Gender"].replace({"nan": "Unknown"}).fillna("Unknown")
    courses["CourseRating"] = courses["CourseRating"].fillna(courses["CourseRating"].mean())
    courses["CoursePrice"] = courses["CoursePrice"].fillna(0)
    trans = trans.dropna(subset=["UserID", "CourseID", "TransactionDate"])
    trans["Amount"] = trans["Amount"].fillna(0)

    # unmatched references (transactions pointing to IDs that do not exist)
    before = len(trans)
    trans = trans[trans["UserID"].isin(users["UserID"]) & trans["CourseID"].isin(courses["CourseID"])]
    report["unmatched_transactions_removed"] = before - len(trans)
    trans = trans.copy()
    trans["TeacherID"] = trans["TeacherID"].where(trans["TeacherID"].isin(teachers["TeacherID"]), np.nan)

    report["rows"] = {"Users": len(users), "Teachers": len(teachers),
                      "Courses": len(courses), "Transactions": len(trans)}
    data = {"users": users.reset_index(drop=True), "teachers": teachers.reset_index(drop=True),
            "courses": courses.reset_index(drop=True), "transactions": trans.reset_index(drop=True)}
    return data, report


def build_master(data):
    """Join Transactions + Courses + Users (+ Teachers) into one table (one row per enrollment)."""
    m = data["transactions"].merge(data["courses"], on="CourseID", how="left")
    m = m.merge(data["users"][["UserID", "Age", "Gender"]], on="UserID", how="left")
    t = data["teachers"][["TeacherID", "TeacherName", "Expertise", "TeacherRating", "YearsOfExperience"]]
    return m.merge(t, on="TeacherID", how="left")


# ----------------------------------------------------------------------------
# 2. LEARNER-LEVEL FEATURE ENGINEERING
# ----------------------------------------------------------------------------
def _preferred(master, col, rating_tiebreak=True):
    """Most frequent value of `col` per learner (ties broken by higher rating)."""
    g = master.groupby(["UserID", col]).agg(n=("CourseID", "count"), r=("CourseRating", "sum")).reset_index()
    g = g.sort_values(["UserID", "n", "r"], ascending=[True, False, False])
    return g.drop_duplicates("UserID").set_index("UserID")[col]


def build_learner_features(master, users):
    """Aggregate transactions to one row per learner and add demographics."""
    g = master.groupby("UserID")
    f = pd.DataFrame({
        "total_courses": g["CourseID"].count(),
        "total_spent": g["Amount"].sum(),
        "avg_spend": g["Amount"].mean(),
        "avg_rating": g["CourseRating"].mean(),
        "categories_explored": g["CourseCategory"].nunique(),
        "free_ratio": g["CourseType"].apply(lambda s: (s == "Free").mean()),
        "first_date": g["TransactionDate"].min(),
        "last_date": g["TransactionDate"].max(),
    })
    f["avg_courses_per_category"] = f["total_courses"] / f["categories_explored"]
    # enrollment frequency = courses per active month
    months = master.assign(m=master["TransactionDate"].dt.to_period("M")).groupby("UserID")["m"].nunique()
    f["enroll_frequency"] = f["total_courses"] / months
    # learning depth: share of each level and (advanced - beginner) index
    lv = pd.crosstab(master["UserID"], master["CourseLevel"]).reindex(columns=LEVELS, fill_value=0)
    lv = lv.div(lv.sum(axis=1), axis=0)
    f["beginner_ratio"], f["intermediate_ratio"], f["advanced_ratio"] = lv["Beginner"], lv["Intermediate"], lv["Advanced"]
    f["depth_index"] = f["advanced_ratio"] - f["beginner_ratio"]  # -1 = all beginner, +1 = all advanced
    f["preferred_category"] = _preferred(master, "CourseCategory")
    f["preferred_level"] = _preferred(master, "CourseLevel")
    # how focused the learner is on the favourite category
    top = master.groupby(["UserID", "CourseCategory"]).size().groupby("UserID").max()
    f["category_focus"] = top / f["total_courses"]
    f = f.reset_index().merge(users[["UserID", "UserName", "Age", "Gender"]], on="UserID", how="left")
    return f


# ----------------------------------------------------------------------------
# 3. PREPROCESSING + CLUSTERING
# ----------------------------------------------------------------------------
NUM_FEATURES = ["Age", "total_courses", "categories_explored", "avg_courses_per_category", "enroll_frequency",
                "avg_rating", "avg_spend", "free_ratio", "depth_index", "category_focus"]
CAT_FEATURES = ["Gender", "preferred_category", "preferred_level"]


def preprocess(features):
    """Scale numeric columns (StandardScaler) and one-hot encode categorical columns."""
    num = features[NUM_FEATURES].copy()
    num["total_courses"] = np.log1p(num["total_courses"])  # reduce skew from heavy learners
    num["enroll_frequency"] = np.log1p(num["enroll_frequency"])
    num_scaled = StandardScaler().fit_transform(num)
    cat = pd.get_dummies(features[CAT_FEATURES], dtype=float)
    # one-hot columns are 0/1, so give them half weight; otherwise 12 category columns dominate the distance
    X = np.hstack([num_scaled, cat.values * 0.5])
    return X, NUM_FEATURES + list(cat.columns)


def scan_k(X, k_min=2, k_max=10):
    """Elbow (inertia) and Silhouette score for each K."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        rows.append({"K": k, "Inertia": km.inertia_, "Silhouette": silhouette_score(X, km.labels_)})
    return pd.DataFrame(rows)


def best_k(scan, low=4, high=8):
    """Best silhouette between low and high. K=2/3 usually only split 'heavy' vs 'light' learners,
    which is too coarse for personalisation, so we search from 4 segments upward."""
    s = scan[(scan["K"] >= low) & (scan["K"] <= high)]
    return int(s.loc[s["Silhouette"].idxmax(), "K"])


MIN_FIT = 2  # learners with fewer enrolments are too sparse to shape the clusters


def fit_mask(features):
    """True for learners with enough history (>= MIN_FIT enrolments) to be used for fitting."""
    return (features["total_courses"] >= MIN_FIT).values


def fit_kmeans(X, mask, k):
    """Fit K-Means on learners with enough history, then assign every learner (incl. sparse ones)
    to the nearest cluster centre. This reduces noise from learners with a single enrolment."""
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X[mask])
    return km.predict(X), km


def hierarchical_check(X_fit, k, kmeans_fit_labels):
    """Ward hierarchical clustering as a secondary comparison with K-Means."""
    hl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_fit)
    return {"silhouette": silhouette_score(X_fit, hl), "agreement_ARI": adjusted_rand_score(kmeans_fit_labels, hl)}


def pca_2d(X):
    """Project the feature matrix to 2 dimensions (only used to draw the cluster scatter plot)."""
    p = PCA(n_components=2, random_state=RANDOM_STATE)
    xy = p.fit_transform(X)
    return pd.DataFrame(xy, columns=["PC1", "PC2"]), p.explained_variance_ratio_


def intra_cluster_similarity(X, labels):
    """Average cosine similarity of each member to its own cluster centre (higher = more consistent)."""
    out = {}
    for c in np.unique(labels):
        pts = X[labels == c]
        out[c] = float(cosine_similarity(pts, pts.mean(axis=0, keepdims=True)).mean())
    return out


def name_segments(features, labels):
    """Give each cluster a readable name using its own average behaviour vs. the overall average."""
    f = features.assign(cluster=labels)
    prof = f.groupby("cluster")[["total_courses", "categories_explored", "free_ratio", "depth_index",
                                 "avg_spend", "category_focus", "avg_rating"]].mean()
    overall = f[prof.columns].mean()
    std = f[prof.columns].std().replace(0, 1)
    z = (prof - overall) / std  # how far each cluster is from average, in std units
    names, notes = {}, {}
    for c in prof.index:
        r, p = z.loc[c], prof.loc[c]
        # rule order: the strongest / most distinctive behaviour decides the name
        if p["total_courses"] >= 1.5 * overall["total_courses"] and p["categories_explored"] >= overall["categories_explored"]:
            name, note = "Multi-Category Power Learners", "enrol in many courses across many categories"
        elif p["depth_index"] > 0.4:
            name, note = "Career-Focused Learners", "prefer advanced courses"
        elif p["total_courses"] >= 2 and p["category_focus"] >= 0.7:
            name, note = "Specialized Learners", "concentrate on one subject"
        elif p["free_ratio"] >= 0.85 and p["depth_index"] < -0.3:
            name, note = "Budget-Conscious Beginners", "mostly free, beginner-level courses"
        elif p["free_ratio"] >= 0.85:
            name, note = "Budget-Conscious Learners", "choose almost only free courses"
        elif r["avg_spend"] > 0.5 and r["free_ratio"] < -0.3:
            name, note = "Premium Paid-Course Learners", "pay for courses and spend above average"
        elif p["depth_index"] < -0.4:
            name, note = "Beginner Explorers", "start with beginner-level courses"
        elif r["category_focus"] > 0.5:
            name, note = "Specialized Learners", "concentrate on one subject"
        else:
            name, note = "Balanced Learners", "mix levels and course types with no strong extreme"
        names[c], notes[c] = name, note
    # make names unique (e.g. two clusters both called "Budget-Conscious Learners" become "... (A)" and "... (B)")
    base = dict(names)
    counter = {}
    for c in prof.index:
        if list(base.values()).count(base[c]) > 1:
            counter[base[c]] = counter.get(base[c], 0) + 1
            names[c] = f"{base[c]} ({'ABCDEFGH'[counter[base[c]] - 1]})"
    return names, prof, z


def segment_summary(features, labels, names):
    """One row per segment with simple averages and a plain-language description."""
    f = features.assign(cluster=labels)
    rows = []
    for c, g in f.groupby("cluster"):
        top_cat = g["preferred_category"].value_counts().index[0]
        top_lvl = g["preferred_level"].value_counts().index[0]
        d = {"Segment": names[c], "Learners": len(g), "Share %": round(100 * len(g) / len(f), 1),
             "Avg courses": round(g["total_courses"].mean(), 2),
             "Avg categories": round(g["categories_explored"].mean(), 2),
             "Free courses %": round(100 * g["free_ratio"].mean(), 1),
             "Avg spend / course": round(g["avg_spend"].mean(), 2),
             "Avg rating": round(g["avg_rating"].mean(), 2),
             "Depth index": round(g["depth_index"].mean(), 2),
             "Top category": top_cat, "Top level": top_lvl}
        d["Description"] = (f"Learners in this group enrol in about {d['Avg courses']} courses across "
                            f"{d['Avg categories']} categories. {d['Free courses %']}% of their courses are free and they "
                            f"spend about ${d['Avg spend / course']} per course. Their most common category is "
                            f"{top_cat} and most common level is {top_lvl}.")
        rows.append(d)
    return pd.DataFrame(rows).sort_values("Learners", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------------
# 4. RECOMMENDATION ENGINE (hybrid, cluster-aware)
# ----------------------------------------------------------------------------
WEIGHTS = {"content": 0.25, "similar": 0.25, "cluster": 0.20, "category": 0.10, "level": 0.10, "rating": 0.10}


def interaction_matrix(trans, user_ids, courses):
    """Learner x Course matrix: 1 if the learner enrolled in the course."""
    R = pd.crosstab(trans["UserID"], trans["CourseID"]).clip(upper=1)
    return R.reindex(index=user_ids, columns=courses["CourseID"], fill_value=0)


class Recommender:
    """Scores every course for every learner using 6 components, then combines them with WEIGHTS.
      content   - cosine similarity between the learner's taste profile and each course (category/level/type)
      similar   - courses taken by the most similar learners of the same segment
      cluster   - popularity of the course inside the learner's segment
      category  - how much the learner has used the course's category before
      level     - how much the learner has used the course's level before
      rating    - course rating / 5
    """

    def __init__(self, R, labels, courses, neighbours=30):
        self.R, self.labels = R, np.asarray(labels)
        self.courses = courses.reset_index(drop=True)
        self.user_index = {u: i for i, u in enumerate(R.index)}
        A = R.values.astype(float)
        self.A = A
        cat = pd.get_dummies(self.courses["CourseCategory"], dtype=float).values
        lvl = pd.get_dummies(self.courses["CourseLevel"], dtype=float).values
        typ = pd.get_dummies(self.courses["CourseType"], dtype=float).values
        F = np.hstack([cat, lvl, typ])

        def share(M):  # learner's share of enrolments per attribute value, mapped back to courses
            u = A @ M
            u = u / np.maximum(u.sum(axis=1, keepdims=True), 1)
            return u @ M.T

        comp = {}
        comp["content"] = cosine_similarity(A @ F, F)            # content-based filtering
        comp["category"] = share(cat)                             # preferred category
        comp["level"] = share(lvl)                                # preferred level
        pop = np.zeros_like(A)                                    # popularity inside the learner's cluster
        self.cluster_counts = {}
        for c in np.unique(self.labels):
            counts = A[self.labels == c].sum(axis=0)
            self.cluster_counts[c] = counts
            pop[self.labels == c] = counts / max(counts.max(), 1)
        comp["cluster"] = pop
        S = cosine_similarity(A)                                  # similar learners (same cluster only)
        S[self.labels[:, None] != self.labels[None, :]] = 0
        np.fill_diagonal(S, 0)
        k = min(neighbours, S.shape[1] - 1)
        thresh = np.partition(S, -k, axis=1)[:, -k][:, None]
        S[S < thresh] = 0
        sim = S @ A
        comp["similar"] = sim / np.maximum(sim.max(axis=1, keepdims=True), 1e-9)
        comp["rating"] = np.tile(self.courses["CourseRating"].values / 5.0, (A.shape[0], 1))
        self.comp = comp
        score = sum(WEIGHTS[n] * comp[n] for n in WEIGHTS)
        self.score = np.where(A > 0, -np.inf, score)              # never recommend already-enrolled courses

    # ---- recommendations for one learner -------------------------------------------------
    def recommend(self, user_id, seg_names, top_n=5, categories=None, levels=None, types=None, force_fallback=False):
        i = self.user_index[user_id]
        history = int(self.A[i].sum())
        if history == 0 or force_fallback:
            return self.fallback(None if history == 0 else i, seg_names, top_n, categories, levels, types, history)
        out = self.courses.copy()
        out["Score"] = self.score[i]
        out = self._filter(out, categories, levels, types)
        out = out[np.isfinite(out["Score"])].sort_values("Score", ascending=False).head(top_n).copy()
        if out.empty:
            return out
        idx = [self.courses.index[self.courses["CourseID"] == cid][0] for cid in out["CourseID"]]
        out["Reason"] = [self._reason(i, j, seg_names) for j in idx]
        out["Type"] = "Personalized"
        return out

    def fallback(self, i, seg_names, top_n, categories, levels, types, history=0):
        """Popular, well-rated courses (segment-level if the learner has a segment, else platform-wide)."""
        if i is not None:
            counts = self.cluster_counts[self.labels[i]]
            scope = f"the '{seg_names[self.labels[i]]}' segment"
        else:
            counts, scope = self.A.sum(axis=0), "the whole platform"
        out = self.courses.copy()
        out["Score"] = 0.7 * counts / max(counts.max(), 1) + 0.3 * out["CourseRating"] / 5
        if i is not None:
            out.loc[self.A[i] > 0, "Score"] = -np.inf
        out = self._filter(out, categories, levels, types)
        out = out[np.isfinite(out["Score"])].sort_values("Score", ascending=False).head(top_n).copy()
        out["Reason"] = f"Fallback: popular and well-rated in {scope} (not enough history for personalisation)"
        out["Type"] = "Fallback"
        return out

    @staticmethod
    def _filter(df, categories, levels, types):
        if categories:
            df = df[df["CourseCategory"].isin(categories)]
        if levels:
            df = df[df["CourseLevel"].isin(levels)]
        if types:
            df = df[df["CourseType"].isin(types)]
        return df

    def _reason(self, i, j, seg_names):
        c = self.courses.iloc[j]
        seg = seg_names[self.labels[i]]
        texts = {
            "content": f"similar in topic/level/type to courses you already took",
            "similar": "learners with similar behaviour in your segment enrolled in it",
            "cluster": f"popular in your segment '{seg}' ({int(self.cluster_counts[self.labels[i]][j])} enrolments)",
            "category": f"matches your interest in {c['CourseCategory']}",
            "level": f"fits your usual {c['CourseLevel']} level",
            "rating": f"highly rated ({c['CourseRating']}/5)",
        }
        contrib = {n: WEIGHTS[n] * self.comp[n][i, j] for n in WEIGHTS}
        top = [n for n, v in sorted(contrib.items(), key=lambda x: -x[1])
               if self.comp[n][i, j] >= 0.4 and (n != "rating" or c["CourseRating"] >= 4.0)][:3]
        if not top:
            top = [max((n for n in contrib if n != "rating"), key=contrib.get)]
        return "Recommended because " + "; ".join(texts[n] for n in top)


# ----------------------------------------------------------------------------
# 5. EVALUATION: Precision@K with a leave-last-one-out split
# ----------------------------------------------------------------------------
def evaluate_recommender(trans, user_ids, courses, labels, ks=(3, 5, 10)):
    """For every learner with >= 2 enrolments, hide the LATEST enrolment, rebuild everything from the
    remaining history and check whether the hidden course appears in the top-K list."""
    t = trans.sort_values(["UserID", "TransactionDate", "TransactionID"])
    counts = t.groupby("UserID")["CourseID"].transform("count")
    held = t[counts >= 2].groupby("UserID").tail(1)
    train = t.drop(held.index)
    R_train = interaction_matrix(train, user_ids, courses)
    rec = Recommender(R_train, labels, courses)
    cidx = {c: j for j, c in enumerate(R_train.columns)}
    uidx = {u: i for i, u in enumerate(R_train.index)}
    rows_i = np.array([uidx[u] for u in held["UserID"]])
    cols_j = np.array([cidx[c] for c in held["CourseID"]])
    pop_score = np.where(R_train.values > 0, -np.inf, R_train.values.sum(axis=0)[None, :] + 0 * R_train.values)
    rng = np.random.RandomState(RANDOM_STATE)
    results = []
    for k in ks:
        hits = {}
        for name, sc in (("Hybrid (our model)", rec.score), ("Popularity baseline", pop_score)):
            s = sc[rows_i]
            target = s[np.arange(len(rows_i)), cols_j][:, None]
            rank = (s > target).sum(axis=1)  # number of courses scored higher than the hidden one
            hits[name] = float((rank < k).mean())
        # random baseline: expected hit rate = K / number of candidate courses
        cand = np.isfinite(rec.score[rows_i]).sum(axis=1)
        hits["Random baseline"] = float(np.minimum(k / cand, 1).mean())
        for name, hr in hits.items():
            results.append({"K": k, "Method": name, "Hit Rate@K": round(hr, 4), "Precision@K": round(hr / k, 4)})
    return pd.DataFrame(results), len(held)
