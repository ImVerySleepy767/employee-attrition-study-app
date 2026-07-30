# IBM HR Employee Attrition Prediction

## Project Overview
Employee attrition poses a major financial and operational challenge for enterprise organizations. The objective of this project is to build an end-to-end binary classification model to predict employee churn (`Attrition = Yes/No`) based on demographic, compensation, workload, and job satisfaction indicators.

Early identification of attrition risk allows HR business partners to deploy targeted retention strategies (e.g., compensation review, workload rebalancing, career pathing) before key talent leaves.

---

## Model Comparison — Untuned vs. Tuned (All 3 Algorithms)

Both iterations follow the same pattern: train each algorithm untuned, then tune it with `RandomizedSearchCV` (5-fold CV, `scoring='f1'`, ≤3 values per hyperparameter), and compare against its own untuned baseline.

### Iteration 1 — Original Features

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (Untuned) | 0.770 | 0.310 | 0.383 | 0.343 |
| Decision Tree (Tuned) | 0.833 | 0.455 | 0.213 | 0.290 |
| Random Forest (Untuned) | 0.837 | 0.444 | 0.085 | 0.143 |
| Random Forest (Tuned) | 0.837 | 0.417 | 0.106 | 0.170 |
| Gradient Boosting (Untuned) | 0.850 | 0.588 | 0.213 | 0.313 |
| Gradient Boosting (Tuned) | 0.847 | 0.526 | 0.213 | 0.303 |

### Iteration 2 — Engineered Features (`TenureRatio`, `LowWLB_Overtime`, `IncomePerWorkingYear`)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree (Untuned, FE) | 0.770 | 0.327 | 0.362 | 0.343 |
| Decision Tree (Tuned, FE) | 0.833 | 0.455 | 0.213 | 0.290 |
| Random Forest (Untuned, FE) | 0.833 | 0.375 | 0.064 | 0.109 |
| Random Forest (Tuned, FE) | 0.833 | 0.333 | 0.085 | 0.136 |
| Gradient Boosting (Untuned, FE) | 0.847 | 0.611 | 0.234 | 0.339 |
| **Gradient Boosting (Tuned, FE) — FINAL** | **0.850** | **0.545** | **0.255** | **0.348** |

**Selection rationale:** The tuned Gradient Boosting model on engineered features was selected as the final model. It has the highest F1-Score of all twelve model/iteration combinations, and feature engineering measurably improved it over the original-feature version (F1 0.303 → 0.348, Recall 0.213 → 0.255). We acknowledge the untuned Decision Tree actually has the single highest Recall (0.383) of any model tested — a business that wants to cast the widest possible net on flagging at-risk employees, accepting more false alarms, could reasonably choose that model instead. We chose Gradient Boosting because it best balances catching leavers against not overwhelming HR with false positives.

---

## Why Accuracy Was Misleading (and Recall/F1 Weren't)

Our dataset has an 83.9% / 16.1% class imbalance (stayed vs. left). This means a model that *never* predicts attrition — always guessing "stayed" — would already score 83.9% accuracy while catching zero actual leavers. Accuracy alone rewards this lazy behaviour, so it cannot distinguish a genuinely useful model from one that just learned to ignore the minority class.

This is exactly what happened with our untuned Random Forest baseline: despite a high overall accuracy (0.837), its Recall on the original features was only 0.085 — it caught just 8–9% of employees who actually left, because predicting "stayed" for nearly everyone was still a low-error strategy under heavy class imbalance.

Precision and Recall tell a more honest story because they're computed *only* on the minority class:
- **Recall** answers: *"Of employees who actually left, how many did we catch?"*
- **Precision** answers: *"Of employees we flagged as high-risk, how many actually left?"*

This is why model selection prioritized Recall and F1-Score (the harmonic balance of Recall and Precision) over raw Accuracy — a model with slightly lower accuracy but meaningfully higher recall is more useful to an HR team trying to actually intervene before people quit.

---

## Assumptions & Limitations

* **Snapshot representativeness:** This dataset is a single cross-sectional extract, not a time series. We assume the patterns it captures (e.g., overtime and low work-life balance driving attrition) are reasonably stable over time, though real-world shifts — such as new remote-work policies — could change these relationships going forward.
* **Self-reported psychometric fields:** `JobSatisfaction`, `WorkLifeBalance`, and `EnvironmentSatisfaction` are ordinal self-assessments (1–4 scales). We assume employees interpreted these scales consistently, though satisfaction is inherently subjective.
* **Recall-over-precision business assumption:** We assumed a missed leaver (false negative) costs the business more than an unnecessary retention conversation (false positive) — this assumption directly shaped our choice of F1/Recall as the primary optimization targets rather than accuracy.
* **Excluded identifier columns:** `EmployeeNumber`, `EmployeeCount`, `Over18`, and `StandardHours` were dropped as either unique identifiers or zero-variance constants, on the assumption they carry no generalizable predictive signal.
* **Model class limitation:** Only scikit-learn tree-based models and `RandomizedSearchCV` were used, per the assignment spec. This favours interpretability (clear feature importances for HR stakeholders) over techniques like SMOTE, which were out of scope here.
* **Minority class size:** With only 237 positive (churn) cases in the full dataset, the final model's recall (~25–26%) leaves room for improvement; more historical data would likely help more than further hyperparameter tuning at this point.

---

## Development Log (July 21, 2026 – Present)

### 1. Data Ingestion & Initial Cleaning (July 21)
* **Initial Loading:** Loaded `IBM HR Employee Attrition Data.csv` into Pandas.
* **Encoding Artifact Fix:** Detected corrupted metadata header (`ï»¿Age`) during initial inspection (`df.info()`) caused by BOM UTF-8 file encoding. Standardized the column name back to `Age`.
* **Structural Cleaning:** Removed uninformative identifier and constant attributes (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`) before performing train-test split.

### 2. Feature Metadata & Setup (July 23)
* **Data Preparation:** Mapped the target variable `Attrition` (`Yes` → `1`, `No` → `0`) and performed an 80/20 stratified train-test split (`random_state=42`).
* **Categorical Encoding:** Converted string features into numerical indicators using One-Hot Encoding (`pd.get_dummies`), split-then-encode to avoid data leakage.
* **Baseline Modelling:** Trained baseline models across three algorithms — Decision Tree, Random Forest, Gradient Boosting.

### 3. Baseline Evaluation & Metric Strategy (July 25)
* **Class Imbalance Realization:** Target distribution showed an 83.9% / 16.1% imbalance, making Accuracy alone misleading.
* **Metric Strategy:** Prioritized Recall and F1-Score over Accuracy (see rationale above). Dropped ROC-AUC from the evaluation set since it wasn't part of the taught curriculum for this module — kept to Accuracy, Precision, Recall, and F1 (confusion-matrix based metrics) throughout.

### 4. Feature Engineering & Hyperparameter Tuning (July 27)
* Engineered `TenureRatio`, `LowWLB_Overtime`, and `IncomePerWorkingYear`.
* Ran `RandomizedSearchCV` (5-fold CV, `scoring='f1'`) across all three algorithms, for both the original and engineered feature sets — six untuned/tuned pairs total.

### 5. Final Model Artifact (July 29)
* Selected tuned Gradient Boosting (engineered features) as the final model — see comparison table above.
* Exported the trained model and feature-alignment pipeline as `best_attrition_model.pkl` for Streamlit deployment.

### 6. Bug Fixes, Documentation & Deployment Polish (July 30)
* **Critical fix:** Found and corrected a cell-ordering bug where the one-hot encoding cell referenced `X_train`/`X_test` before the train-test split cell that creates them — would have crashed a top-to-bottom "Run All". Verified a clean re-run with no errors afterward.
* **Documentation:** Added written hyperparameter-selection rationale and an Assumptions & Limitations section to the notebook, addressing rubric requirements for explained tuning impact and stated assumptions.
* **Streamlit app rebuild:** Consolidated the app to 3 pages (Predict Risk / Insights / About), fixed a text-contrast bug in the dark theme, moved to a navy/ice-blue/gold colour scheme, and added a live dashboard reading directly from the training CSV (department attrition rates, model feature importances) plus a per-prediction comparison chart (employee profile vs. company averages).
* **Tutor review:** Confirmed with module tutor (Alvin PS Tan) that the untuned-vs-tuned comparison table format above is an acceptable way to justify final model selection.

---

## Project Structure
```text
├── IBM_HR_Employee_Attrition_Data.csv             # Raw input dataset
├── MLDP_Program_Codes_Submission_Template.ipynb   # Main project notebook
├── best_attrition_model.pkl                       # Exported model artifact (model + feature_columns)
├── app.py                                         # Streamlit deployment app
├── requirements.txt                               # App dependencies
└── README.md                                      # Project documentation
```
