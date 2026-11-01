# EduPro – Student Segmentation and Personalized Course Recommendation System

Streamlit web app that segments online learners with **K-Means clustering** and recommends courses with a
**hybrid, cluster-aware recommender**. It uses only your real Excel data (`data/EduPro Online Platform.xlsx`);
the file is read-only and never modified.

## Project structure
```
EduPro_Project/
├── app.py              # Streamlit interface (6 pages, filters, charts, CSV downloads)
├── utils.py            # data cleaning, feature engineering, K-Means, recommender, Precision@K
├── data/EduPro Online Platform.xlsx
├── .streamlit/config.toml   # dark navy / blue theme
├── requirements.txt
└── README.md
```

## Install and run (VS Code, Windows / Mac / Linux)
1. Install Python 3.10+ and open the `EduPro_Project` folder in VS Code (File → Open Folder).
2. Open the terminal (Ctrl + `) and create a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate          # Windows      (Mac/Linux: source venv/bin/activate)
   ```
3. Install packages and start the app:
   ```
   pip install -r requirements.txt
   streamlit run app.py
   ```
4. The browser opens at http://localhost:8501. Keep the Excel file inside `data/`.

## How it works (for your presentation)
1. **Data loading** – 4 sheets are read; types fixed, duplicates and unmatched IDs removed (none were found in
   your file: 3,000 users, 60 teachers, 60 courses, 10,000 transactions). Sheets are joined with
   `UserID`, `CourseID`, `TeacherID`.
2. **Learner-level features** (one row per learner): courses enrolled, preferred category and level, average rating,
   average spend, categories explored (diversity), learning depth index (advanced share − beginner share),
   enrolment frequency (courses per active month), free-course share, category focus, plus age and gender.
3. **Preprocessing** – numeric features scaled with `StandardScaler`; gender, preferred category and preferred level
   one-hot encoded.
4. **Clustering** – K-Means fitted on learners with 2+ enrolments (a learner with one enrolment is too sparse to
   learn from); the remaining learners are assigned to the nearest cluster centre. K is chosen with the
   **Elbow method** and **Silhouette score** (best silhouette between K = 4 and 8; adjustable in the sidebar).
   **Hierarchical (Ward) clustering** is shown as a secondary comparison (silhouette + Adjusted Rand Index).
5. **Segment names** are assigned by simple rules from each cluster's actual averages – not forced.
6. **Recommender** – every course gets a score = weighted sum of: content-based similarity (0.25), similar learners in
   the same segment (0.25), popularity inside the segment (0.20), preferred category (0.10), preferred level (0.10),
   course rating (0.10). Courses the learner already took are excluded. Each recommendation shows its reasons.
   Learners with no history get **fallback** popular courses (tick the preview box on the page to see it).
7. **Evaluation** – leave-last-one-out: hide each learner's latest enrolment, rebuild the model, check if the hidden
   course is in the top-K (Hit Rate@K, Precision@K) and compare with popularity and random baselines.

## Results on your dataset (K = 4, Silhouette ≈ 0.26)
| Segment | Learners | What defines it |
|---|---|---|
| Budget-Conscious Learners | 1,475 | ~97% free courses, average spend ≈ $3 per course |
| Premium Paid-Course Learners | 978 | mostly paid courses, ≈ $217 per course |
| Multi-Category Power Learners | 456 | ≈ 13 courses across ≈ 9 categories |
| Specialized Learners | 91 | few courses, concentrated in one category |

No "Career-Focused" or "Beginner Explorer" group appeared: learners' beginner/advanced mix is spread evenly,
so the data does not support those segments.

## Limitations (mention these honestly)
- 54% of learners have only one enrolment, so their behaviour evidence is limited.
- Silhouette ≈ 0.26 means a moderate cluster structure, not sharply separated groups.
- Precision@K is low (hybrid Hit Rate@5 ≈ 12.7% vs ≈ 9.2% random and ≈ 11.7% popularity). The model beats the
  baselines slightly, but predicting one hidden next course among 60 is hard. Engagement or course-completion
  improvement is **not** measured.
- The dataset has no country/region field, so no region filter exists. Amounts are shown with "$" for readability
  (the file has no currency column).

## App pages
Dashboard Overview · Learner Profile Explorer · ML Segmentation · Course Recommendations · Learning Behaviour ·
Teacher & Course Analysis. Sidebar filters (segment, category, level, type, learner, K) update all charts;
**Reset filters** restores the defaults; every page with data has a CSV download button.
