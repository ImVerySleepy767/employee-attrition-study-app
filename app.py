import joblib
import streamlit as st
import pandas as pd
import time

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS — Fonts, Colors, Animations
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
    }

    /* App background gradient */
    .stApp {
        background: linear-gradient(160deg, #0f172a 0%, #1e293b 45%, #312e81 100%);
        color: #f1f5f9;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #0f172a 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    /* Top nav bar */
    .navbar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(90deg, #7c3aed 0%, #db2777 100%);
        box-shadow: 0 8px 24px rgba(124, 58, 237, 0.35);
        margin-bottom: 1.6rem;
        animation: fadeInDown 0.6s ease;
    }
    .navbar h1 {
        color: white;
        font-size: 1.4rem;
        margin: 0;
    }
    .navbar span {
        color: rgba(255,255,255,0.85);
        font-size: 0.85rem;
    }

    /* Hero / title card */
    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem 1rem;
        animation: fadeIn 0.8s ease;
    }
    .hero h1 {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #f472b6, #fb923c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #cbd5e1;
        font-size: 1.02rem;
        max-width: 640px;
        margin: 0 auto;
    }

    /* Section card wrapper */
    .card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(6px);
        animation: fadeInUp 0.6s ease;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(124, 58, 237, 0.25);
    }
    .card h3 {
        color: #f472b6;
        margin-top: 0;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #7c3aed, #db2777);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        font-weight: 600;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 18px rgba(219, 39, 119, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px rgba(219, 39, 119, 0.5);
    }

    /* Result badges */
    .risk-high {
        background: linear-gradient(90deg, #b91c1c, #ef4444);
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        animation: pulse 1.6s infinite, fadeIn 0.5s ease;
        box-shadow: 0 8px 22px rgba(239, 68, 68, 0.4);
    }
    .risk-low {
        background: linear-gradient(90deg, #15803d, #22c55e);
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
        animation: fadeIn 0.5s ease;
        box-shadow: 0 8px 22px rgba(34, 197, 94, 0.35);
    }

    /* Metric-like chips */
    .chip {
        display: inline-block;
        background: rgba(124, 58, 237, 0.18);
        border: 1px solid rgba(167, 139, 250, 0.4);
        color: #ddd6fe;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }

    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-14px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.15);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Load trained model artifact
# ==========================================
try:
    artifact = joblib.load("best_attrition_model.pkl")
    model = artifact["model"]
    feature_columns = artifact["feature_columns"]
except FileNotFoundError:
    st.error("Model file 'best_attrition_model.pkl' not found. "
             "Make sure it's in the same folder as this app.")
    st.stop()

# ==========================================
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Go to",
        ["🔮 Predict Risk", "ℹ️ About This Tool", "📈 Model Info"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### 🎯 Quick Facts")
    st.markdown("""
    <div class="chip">Gradient Boosting</div>
    <div class="chip">HR Analytics</div>
    <div class="chip">Recall-Focused</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit · Powered by scikit-learn")

# ==========================================
# Top Navbar
# ==========================================
st.markdown("""
<div class="navbar">
    <h1>📊 Attrition Insights</h1>
    <span>Employee Retention Analytics Platform</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# Default values for features NOT exposed in the UI
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
# PAGE: Predict Risk
# ==========================================
if page == "🔮 Predict Risk":

    st.markdown("""
    <div class="hero">
        <h1>Employee Attrition Risk Predictor</h1>
        <p>Estimate the risk that an employee will leave the company, based on key
        workplace and demographic factors. Fill in the details below and click Predict.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 👤 Employee Profile")

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

    st.markdown('</div>', unsafe_allow_html=True)

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
    predict_clicked = st.button("🔮 Predict Attrition Risk", type="primary", disabled=bool(errors))

    if predict_clicked:
        with st.spinner("Analyzing employee profile..."):
            time.sleep(0.6)  # brief pause purely for perceived feedback

            # 1. Assemble the full feature row
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

            # 2. Recreate engineered features
            df_input["TenureRatio"] = df_input["YearsAtCompany"] / (df_input["TotalWorkingYears"] + 1e-5)
            df_input["LowWLB_Overtime"] = (
                (df_input["WorkLifeBalance"] <= 2) & (df_input["OverTime"] == "Yes")
            ).astype(int)
            df_input["IncomePerWorkingYear"] = df_input["MonthlyIncome"] / (df_input["TotalWorkingYears"] + 1)

            # 3. One-hot encode
            categorical_cols = ["BusinessTravel", "Department", "EducationField",
                                 "Gender", "JobRole", "MaritalStatus", "OverTime"]
            df_input = pd.get_dummies(df_input, columns=categorical_cols, drop_first=True)

            # 4. Align columns
            df_input = df_input.reindex(columns=feature_columns, fill_value=0)

            # 5. Predict
            prediction = model.predict(df_input)[0]
            probability = model.predict_proba(df_input)[0][1]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 🎯 Result")

        if prediction == 1:
            st.markdown(f"""
            <div class="risk-high">
                ⚠️ High Risk of Attrition &nbsp;|&nbsp; Predicted probability: {probability:.1%}
            </div>
            """, unsafe_allow_html=True)
            st.toast("High attrition risk detected — consider a retention check-in.", icon="⚠️")
        else:
            st.markdown(f"""
            <div class="risk-low">
                ✅ Low Risk of Attrition &nbsp;|&nbsp; Predicted probability: {probability:.1%}
            </div>
            """, unsafe_allow_html=True)
            st.toast("Low attrition risk — employee looks stable.", icon="✅")
            st.balloons()

        st.write("")
        st.progress(float(probability), text=f"Risk score: {probability:.1%}")
        st.caption(
            "This estimate is based on the employee's profile compared against historical "
            "attrition patterns. Use it as a decision-support signal, not a sole determinant."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# PAGE: About
# ==========================================
elif page == "ℹ️ About This Tool":
    st.markdown("""
    <div class="hero">
        <h1>About This Tool</h1>
        <p>Understand how and why this predictor works.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🎯 Purpose</h3>
        <p>This tool estimates the likelihood that an employee will leave the company
        (attrition), based on workplace and demographic factors drawn from the IBM HR
        Employee Attrition dataset. It's designed to help HR business partners identify
        at-risk employees early and act with targeted retention strategies.</p>
    </div>

    <div class="card">
        <h3>⚙️ How It Works</h3>
        <p>You provide 10 key inputs (age, income, job role, overtime, tenure, etc.).
        Less-influential fields are filled with sensible dataset averages behind the
        scenes so the form stays quick to use. The trained model then returns a
        probability of attrition, alongside a High/Low risk label.</p>
    </div>

    <div class="card">
        <h3>⚠️ Limitations</h3>
        <p>This is a decision-support signal, not a diagnosis. Predictions reflect
        historical patterns in the training data and should always be combined with
        human judgment and context before any HR action is taken.</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE: Model Info
# ==========================================
elif page == "📈 Model Info":
    st.markdown("""
    <div class="hero">
        <h1>Model Information</h1>
        <p>A quick look under the hood.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🤖 Algorithm</h3>
            <p>Tuned Gradient Boosting Classifier (scikit-learn), selected after
            comparing against Decision Tree and Random Forest baselines, both
            untuned and hyperparameter-tuned via RandomizedSearchCV.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>🧪 Evaluation Metrics</h3>
            <p>Accuracy, Precision, Recall, and F1-Score, evaluated on a held-out
            test set. Given the dataset's class imbalance (~84% stayed / 16% left),
            Recall and F1 were prioritized over raw accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🛠️ Feature Engineering</h3>
        <p>Three engineered features were added to help the model: <b>TenureRatio</b>
        (company tenure vs. total career length), <b>LowWLB_Overtime</b> (poor work-life
        balance combined with overtime), and <b>IncomePerWorkingYear</b> (income scaled
        by career experience).</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# Footer
# ==========================================
st.markdown("""
<div class="footer">
    Model: Tuned Gradient Boosting Classifier · Trained on IBM HR Employee Attrition dataset
</div>
""", unsafe_allow_html=True)
