import joblib
import streamlit as st
import pandas as pd

# ==========================================
# Page Config
# ==========================================
st.set_page_config(page_title="Employee Attrition Predictor", page_icon="", layout="centered")

# ==========================================
# Load trained model artifact
# ==========================================
# NOTE: best_attrition_model.joblib must be in the SAME folder as this app.py,
# both locally and in your GitHub repo.
try:
    artifact = joblib.load("best_attrition_model.joblib")
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except FileNotFoundError:
    st.error("Model file 'best_attrition_model.joblib' not found. "
             "Make sure it's in the same folder as this app.")
    st.stop()

st.title("📊 Employee Attrition Risk Predictor")
st.markdown(
    "Estimate the risk that an employee will leave the company, based on key "
    "workplace and demographic factors. Fill in the details below and click **Predict**."
)

# ==========================================
# Default values for features NOT exposed in the UI
# (typical/median values from the IBM HR Attrition dataset)
# ==========================================
DEFAULTS = {
    "BusinessTravel": "Travel_Rarely",
    "DailyRate": 800,
    "Department": "Research & Development",
    "Education": 3,
    "EducationField": "Life Sciences",
    "EnvironmentSatisfaction": 3,
    "Gender": "Male",
    "HourlyRate": 65,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "MaritalStatus": "Married",
    "MonthlyRate": 14000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 3,
    "TrainingTimesLastYear": 3,
    "YearsInCurrentRole": 4,
    "YearsSinceLastPromotion": 2,
    "YearsWithCurrManager": 4,
}

# ==========================================
# Interactive Inputs (top drivers of attrition)
# ==========================================
st.subheader("Employee Profile")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", min_value=18, max_value=60, value=35)
    monthly_income = st.number_input("Monthly Income ($)", min_value=1000, max_value=25000, value=5000, step=100)
    job_role = st.selectbox("Job Role", [
        "Sales Executive", "Research Scientist", "Laboratory Technician",
        "Manufacturing Director", "Healthcare Representative", "Manager",
        "Sales Representative", "Research Director", "Human Resources"
    ])
    overtime = st.selectbox("Works Overtime?", ["No", "Yes"])
    distance_from_home = st.slider("Distance From Home (km)", min_value=1, max_value=30, value=9)

with col2:
    total_working_years = st.slider("Total Working Years (career)", min_value=0, max_value=40, value=10)
    years_at_company = st.slider("Years At This Company", min_value=0, max_value=40, value=5)
    work_life_balance = st.select_slider("Work-Life Balance", options=[1, 2, 3, 4],
                                          value=3, help="1 = Bad, 2 = Good, 3 = Better, 4 = Best")
    job_satisfaction = st.select_slider("Job Satisfaction", options=[1, 2, 3, 4],
                                         value=3, help="1 = Low, 2 = Medium, 3 = High, 4 = Very High")
    stock_option_level = st.select_slider("Stock Option Level", options=[0, 1, 2, 3], value=0)

# ==========================================
# Input Validation
# ==========================================
errors = []
if years_at_company > total_working_years:
    errors.append("**Years At Company** cannot be greater than **Total Working Years**.")
if monthly_income < 1000:
    errors.append("**Monthly Income** must be at least $1,000.")
if age < 18:
    errors.append("**Age** must be 18 or above.")

if errors:
    for e in errors:
        st.error(e)

# ==========================================
# Predict
# ==========================================
if st.button("🔮 Predict Attrition Risk", type="primary", disabled=bool(errors)):

    # 1. Assemble the full feature row (interactive inputs + sensible defaults)
    row = {
        "Age": age,
        "MonthlyIncome": monthly_income,
        "JobRole": job_role,
        "OverTime": overtime,
        "DistanceFromHome": distance_from_home,
        "TotalWorkingYears": total_working_years,
        "YearsAtCompany": years_at_company,
        "WorkLifeBalance": work_life_balance,
        "JobSatisfaction": job_satisfaction,
        "StockOptionLevel": stock_option_level,
        **DEFAULTS,
    }
    df_input = pd.DataFrame([row])

    # 2. Recreate the engineered features exactly as in the notebook
    df_input["TenureRatio"] = df_input["YearsAtCompany"] / (df_input["TotalWorkingYears"] + 1e-5)
    df_input["LowWLB_Overtime"] = (
        (df_input["WorkLifeBalance"] <= 2) & (df_input["OverTime"] == "Yes")
    ).astype(int)
    df_input["IncomePerWorkingYear"] = df_input["MonthlyIncome"] / (df_input["TotalWorkingYears"] + 1)

    # 3. One-hot encode categorical columns (same as training)
    categorical_cols = ["BusinessTravel", "Department", "EducationField",
                         "Gender", "JobRole", "MaritalStatus", "OverTime"]
    df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

    # 4. Align to the exact columns the model was trained on
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)

    # 5. Predict
    prediction = model.predict(df_input)[0]
    probability = model.predict_proba(df_input)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f" **High Risk of Attrition** — Predicted probability: {probability:.1%}")
    else:
        st.success(f" **Low Risk of Attrition** — Predicted probability: {probability:.1%}")

    st.progress(float(probability))
    st.caption(
        "This estimate is based on the employee's profile compared against historical "
        "attrition patterns. Use it as a decision-support signal, not a sole determinant."
    )

st.markdown("---")
st.caption("Model: Tuned Gradient Boosting Classifier · Trained on IBM HR Employee Attrition dataset")