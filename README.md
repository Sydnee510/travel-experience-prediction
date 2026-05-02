<div align="center">

# 🚄 Shinkansen Travel Experience Prediction

### Supervised Machine Learning · Transportation Analytics · Multi-Model Ensemble

*Predicting passenger satisfaction for Japan's Shinkansen bullet train using survey data, travel records, and a progression of boosting models — from Decision Trees to tuned CatBoost.*

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br>

![Passengers](https://img.shields.io/badge/Passengers-94%2C379-blueviolet?style=flat-square)
![Features](https://img.shields.io/badge/Features-25-blue?style=flat-square)
![Submissions](https://img.shields.io/badge/Submissions-18-orange?style=flat-square)
![Leaderboard](https://img.shields.io/badge/Leaderboard-95.69%25-success?style=flat-square)
![Hackathon](https://img.shields.io/badge/Hackathon-2nd_Place-gold?style=flat-square)

<br>

</div>

---

## 📁 Project Assets

| Asset | Description |
|:---|:---|
| 📓 [Analysis Notebook](<Learner Notebook - Low Code Version - Travel Experience Prediction.ipynb>) | Full EDA, preprocessing, model progression (DT → RF → XGBoost → LightGBM → CatBoost), ensembling, and fine-tuning |
| 🗂️ [Data Dictionary](<Data_Dictionary_(1) (1).xlsx>) | Feature definitions and value descriptions for all 25 columns |
| 📤 [Final Submission](<TeamSynergistsTravelExperienceSubmission(16) (1).csv>) | CatBoost (Round 7) predictions — 95.6913% leaderboard accuracy · Team Synergists · 2nd place |

---

## 🧠 Business Context

The Shinkansen is one of Japan's most iconic transportation systems — famed for punctuality, speed, and service. But operational excellence alone doesn't guarantee passenger satisfaction. This hackathon challenged participants to answer: **what actually makes passengers happy or unhappy on the Shinkansen?** Team Synergists iterated through 18 submissions across 6+ model families to claim 2nd place on the leaderboard.

**Goal:** Predict whether a passenger was satisfied or not with their overall travel experience, and identify which service dimensions drive that outcome.

- 🎯 Build a model accurate enough to place on a competitive hackathon leaderboard
- 🔍 Identify the most influential factors — is it punctuality, comfort, entertainment, or digital experience?
- 🚄 Provide data-driven recommendations for improving the passenger experience

---

## 📊 Dataset

<div align="center">

**94,379 training records · 35,602 test records · 25 features · Two merged data sources**

</div>

Two datasets were joined on passenger ID: **Traveldata** (on-time performance, demographics, trip details) and **Surveydata** (14 service ratings + target variable).

<details>
<summary><strong>Feature Reference (click to expand)</strong></summary>
<br>

| Feature | Type | Description |
|---|---|---|
| `Gender` | Categorical | Female (50.7%) / Male (49.3%) — nearly even split |
| `Customer_Type` | Categorical | Loyal (74%) vs Disloyal (16.5%) — 9.5% missing |
| `Age` | Numerical | Range: 7–85, mean ~39 — bimodal peaks at ~27 and ~47 |
| `Type_Travel` | Categorical | Business (62%) vs Personal (28%) — 9.8% missing |
| `Travel_Class` | Categorical | Eco (52.3%) vs Business (47.7%) |
| `Seat_Class` | Categorical | Green Car (50.3%) vs Ordinary (49.7%) |
| `Travel_Distance` | Numerical | Range: 50–6,951 km, mean ~1,979 km — right-skewed with high-end outliers |
| `Departure_Delay_in_Mins` | Numerical | Median = 0, mean = 14.6 — heavily right-skewed, max 1,592 mins |
| `Arrival_Delay_in_Mins` | Numerical | Median = 0, correlated 0.97 with departure delay |
| `Onboard_Entertainment` | Ordinal (1–6) | Survey rating — **#1 driver of satisfaction across all models** |
| `Seat_Comfort` | Ordinal (1–6) | Survey rating — 2nd most important feature |
| `Ease_of_Online_Booking` | Ordinal (1–6) | Survey rating — 3rd most important feature |
| `Online_Support` | Ordinal (1–6) | Survey rating |
| `Onboard_Wifi_Service` | Ordinal (1–6) | Survey rating |
| `Onboard_Service` | Ordinal (1–6) | Survey rating |
| `Legroom` | Ordinal (1–6) | Survey rating |
| `Catering` | Ordinal (1–6) | Survey rating |
| `Baggage_Handling` | Ordinal (1–6) | Survey rating |
| `CheckIn_Service` | Ordinal (1–6) | Survey rating |
| `Cleanliness` | Ordinal (1–6) | Survey rating |
| `Online_Boarding` | Ordinal (1–6) | Survey rating |
| `Platform_Location` | Ordinal (1–6) | Survey rating: Very Inconvenient → Very Convenient |
| `Arrival_Time_Convenient` | Ordinal (1–6) | Survey rating |
| `Overall_Experience` | 🎯 Target | Satisfied (1) or Not Satisfied (0) — 54.7% / 45.3% |

</details>

---

## ⚙️ Methodology

<div align="center">

```
Data Merging  →  EDA  →  Preprocessing  →  Baseline Models  →  Boosting  →  Ensembling  →  Fine-Tuning
```

</div>

| Step | Action |
|:---:|:---|
| 1 | **Data Merging** — Inner join of Traveldata and Surveydata on passenger ID → 25 features, 94,379 rows |
| 2 | **EDA** — Univariate distributions, bivariate vs target, correlation heatmap, outlier detection |
| 3 | **Preprocessing** — Ordinal encoding for 14 survey cols (1–6 scale), one-hot encoding for 5 categoricals, median imputation for missing values |
| 4 | **Baseline Models** — Decision Tree and Random Forest; both overfit at 1.00 train accuracy, tuned down to 0.93 and 0.91 respectively |
| 5 | **Boosting Models** — XGBoost, LightGBM, CatBoost, and HistGradientBoosting; each tuned with GridSearchCV |
| 6 | **Ensembling** — Soft-vote VotingClassifier combining XGBoost + LightGBM + CatBoost |
| 7 | **Fine-Tuning** — 9 rounds of CatBoost depth, iteration, and learning rate optimization |

---

## 🎯 Results: Model Progression

| Model | Train Accuracy | Leaderboard |
|---|:---:|:---:|
| Decision Tree — tuned | 0.93 | — |
| Random Forest — tuned | 0.91 | 0.9041 |
| XGBoost — tuned (round 2) | 0.97 | 0.9529 |
| LightGBM — tuned | 0.97 | 0.9537 |
| XGB + LightGBM + CatBoost ensemble | 0.97 | 0.9541 |
| **CatBoost — tuned (round 7)** ✅ | **0.99** | **0.9569** 🥈 |

> Each jump in approach was driven by a real plateau — Random Forest maxed out at 90.41%; switching to sequential boosting unlocked the next tier. CatBoost's ordered boosting proved uniquely suited to this structured survey data, ultimately delivering the best leaderboard score.

**Final Model — CatBoost (Round 7):**
- Best parameters: `depth=8`, `iterations=10,000`, `learning_rate=0.01`
- Lower learning rate with more iterations generalized best — model peaked at 99% training accuracy before beginning to overfit at 100%

### Top Features Driving Satisfaction

| Rank | Feature | Decision Tree | Random Forest |
|:---:|---|:---:|:---:|
| 1 | Onboard_Entertainment | `0.502` | `0.260` |
| 2 | Seat_Comfort | `0.200` | `0.160` |
| 3 | Ease_of_Online_Booking | `0.079` | `0.120` |
| 4 | Online_Support | — | `0.090` |
| 5 | Travel_Class_Eco | `0.029` | `0.050` |

> Both models independently agreed on the same top 3 — giving high confidence these are the true drivers. Onboard_Entertainment was the root node of the Decision Tree, confirming it as the single most decisive factor the model asks about.

---

## 💡 Key Findings & Recommendations

### Satisfaction Rates by Segment

| Segment | Satisfaction Rate |
|---|:---:|
| Travel Class — **Business** | **70.8%** |
| Travel Class — Eco | 39.9% |
| Customer Type — **Loyal** | **61.6%** |
| Customer Type — Disloyal | 23.9% |
| Type of Travel — **Business** | **58.3%** |
| Type of Travel — Personal | 46.5% |
| Median age — Satisfied passengers | **43 years** |
| Median age — Dissatisfied passengers | 36 years |

### Business Recommendations

| # | Recommendation | Signal |
|:---:|---|---|
| 1 | **Prioritize Onboard Entertainment** — the single most important factor across every model built | DT: 50.2% importance · RF: 26% importance |
| 2 | **Improve Eco class seat comfort** — targeted upgrades could close the ~31pp satisfaction gap | Business 70.8% vs Eco 39.9% |
| 3 | **Streamline the online booking experience** — consistently top 3 across all models | DT: 7.9% · RF: 12% importance |
| 4 | **Invest in loyal customer retention** — loyal passengers are satisfied at ~3× the rate of disloyal ones | 61.6% vs 23.9% |
| 5 | **Tailor services for personal travelers** — entertainment and comfort improvements benefit leisure riders most | 46.5% satisfaction vs 58.3% for business |
| 6 | **De-emphasize delay reduction as a satisfaction lever** — punctuality barely affects satisfaction scores | Correlation: -0.07 · last-ranked feature in both tree models |

---

<div align="center">

*MIT Applied Data Science Program · Hackathon — 🥈 2nd Place*

*Built with Python · pandas · NumPy · matplotlib · seaborn · scikit-learn · XGBoost · LightGBM · CatBoost*

</div>
