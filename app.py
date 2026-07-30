import joblib
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import time
import io
import base64

# ==========================================
# Page Config
# ==========================================
st.set_page_config(
    page_title="Employee Attrition Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# Custom CSS — Fonts, Colors, Contrast Fix, Animations
# ==========================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Manrope:wght@400;500;600;700&display=swap');

    :root {
        --bg-deep: #060e1f;
        --bg-mid: #0c1a35;
        --navy: #16305c;
        --ice-blue: #9fc0e8;
        --gold: #e0b04b;
        --text-light: #f4f7fc;
        --muted: #8ea3c4;
        --risk-red: #ef4444;
        --risk-green: #22c55e;
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif;
    }

    /* App background */
    .stApp {
        background: radial-gradient(circle at 15% 0%, #1a3a6e 0%, var(--bg-mid) 40%, var(--bg-deep) 100%);
        color: var(--text-light);
    }

    /* ============ CONTRAST FIX ============ */
    /* Streamlit renders widget labels & slider tick numbers with its own
       light-theme defaults. Without forcing color here, they render dark-on-dark
       and become unreadable — this block is the actual fix for that bug. */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: var(--text-light) !important;
    }
    [data-testid="stTickBarMin"],
    [data-testid="stTickBarMax"],
    [data-testid="stCaptionContainer"] p,
    [data-testid="stCaptionContainer"] {
        color: var(--muted) !important;
        opacity: 1 !important;
    }
    [data-testid="stSliderThumbValue"] {
        color: var(--gold) !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1a38 0%, #050d1c 100%);
        border-right: 1px solid rgba(159, 192, 232, 0.2);
    }
    section[data-testid="stSidebar"] * {
        color: var(--text-light) !important;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label {
        border-radius: 10px;
        padding: 0.35rem 0.6rem;
        transition: background 0.2s ease;
    }
    section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
        background: rgba(159, 192, 232, 0.12);
    }

    /* Sticky top nav bar */
    .navbar {
        position: sticky;
        top: 0;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.9rem 1.6rem;
        border-radius: 16px;
        background: linear-gradient(90deg, #0e2447 0%, #16305c 100%);
        border: 1px solid rgba(159, 192, 232, 0.3);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        margin-bottom: 1.6rem;
        animation: fadeInDown 0.6s ease;
    }
    .navbar h1 { color: white; font-size: 1.4rem; margin: 0; }
    .navbar span { color: var(--muted); font-size: 0.85rem; }

    /* Hero */
    .hero {
        text-align: center;
        padding: 1.8rem 1rem 1.2rem 1rem;
        animation: fadeIn 0.8s ease;
    }
    .hero h1 {
        font-size: 2.3rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--ice-blue), var(--gold));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: var(--muted);
        font-size: 1.02rem;
        max-width: 640px;
        margin: 0 auto;
    }

    /* Section cards */
    .card {
        background: rgba(22, 48, 92, 0.45);
        border: 1px solid rgba(159, 192, 232, 0.2);
        border-left: 4px solid var(--ice-blue);
        border-radius: 18px;
        padding: 1.4rem 1.6rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(6px);
        animation: fadeInUp 0.6s ease;
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 28px rgba(159, 192, 232, 0.18);
    }
    .card h3 { color: var(--gold); margin-top: 0; }

    /* Color variants so cards don't all look identical, kept within the navy/blue/gold family */
    .card-amber { border-left-color: var(--gold); }
    .card-amber h3 { color: var(--gold); }
    .card-steel { border-left-color: #6ea8e6; }
    .card-steel h3 { color: #6ea8e6; }
    .card-violet { border-left-color: #3b6bb5; }
    .card-violet h3 { color: #7ba7de; }
    .card-sky { border-left-color: var(--ice-blue); }
    .card-sky h3 { color: var(--ice-blue); }

    /* Risk score bar — deliberately bold so it reads at a glance */
    .risk-track {
        position: relative;
        width: 100%;
        height: 26px;
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(159, 192, 232, 0.25);
        border-radius: 999px;
        overflow: hidden;
    }
    .risk-fill {
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: var(--target-width);
        border-radius: 999px;
        animation: growBar 1.1s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        box-shadow: 0 0 16px rgba(224, 176, 75, 0.35);
    }
    @keyframes growBar {
        from { width: 0%; }
        to { width: var(--target-width); }
    }
    .risk-threshold {
        position: absolute;
        top: 0;
        left: 50%;
        width: 2px;
        height: 100%;
        background: var(--gold);
        z-index: 2;
    }

    /* Native bordered containers (st.container(border=True)) — used where real
       interactive widgets need to sit inside a card, since raw HTML divs can't
       wrap widgets rendered in separate Streamlit calls. */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(22, 48, 92, 0.45) !important;
        border: 1px solid rgba(159, 192, 232, 0.25) !important;
        border-left: 4px solid var(--ice-blue) !important;
        border-radius: 18px !important;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #1d4488, #16305c);
        color: white;
        border: 1px solid var(--gold);
        border-radius: 12px;
        padding: 0.7rem 1.4rem;
        font-weight: 700;
        font-size: 1rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 18px rgba(22, 48, 92, 0.5);
    }
    .stButton>button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 10px 24px rgba(224, 176, 75, 0.35);
    }

    /* Result badges — kept semantic red/green regardless of theme, since these carry meaning */
    .risk-high {
        background: linear-gradient(90deg, #7f1d1d, var(--risk-red));
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        animation: pulse 1.6s infinite, fadeIn 0.5s ease;
        box-shadow: 0 8px 22px rgba(239, 68, 68, 0.4);
    }
    .risk-low {
        background: linear-gradient(90deg, #14532d, var(--risk-green));
        padding: 1.1rem 1.4rem;
        border-radius: 14px;
        color: white;
        font-weight: 700;
        font-size: 1.1rem;
        animation: fadeIn 0.5s ease;
        box-shadow: 0 8px 22px rgba(34, 197, 94, 0.35);
    }

    /* Chips */
    .chip {
        display: inline-block;
        background: rgba(224, 176, 75, 0.15);
        border: 1px solid rgba(224, 176, 75, 0.4);
        color: #f2d38a;
        border-radius: 999px;
        padding: 0.25rem 0.8rem;
        font-size: 0.8rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .chip-iceblue {
        background: rgba(159, 192, 232, 0.15);
        border-color: rgba(159, 192, 232, 0.4);
        color: var(--ice-blue);
    }
    .chip-sky {
        background: rgba(110, 168, 230, 0.15);
        border-color: rgba(110, 168, 230, 0.4);
        color: #bcd7f5;
    }

    /* Animations */
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes fadeInUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes fadeInDown { from { opacity: 0; transform: translateY(-14px); } to { opacity: 1; transform: translateY(0); } }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.5); }
        70% { box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }
        100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
    }

    .footer {
        text-align: center;
        color: var(--muted);
        font-size: 0.8rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(159, 192, 232, 0.15);
    }

    /* Toast fix — Streamlit's toast renders in its own light box and was
       never covered by the rules above, so its text defaulted to a washed-out
       grey. Force it to match the app's navy theme with high-contrast text. */
    [data-testid="stToast"] {
        background: linear-gradient(90deg, #0e2447, #16305c) !important;
        border: 1px solid rgba(224, 176, 75, 0.45) !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
    }
    [data-testid="stToast"] * {
        color: var(--text-light) !important;
        opacity: 1 !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# Chart embedding helper
# ==========================================
def fig_to_html_img(fig):
    """Render a matplotlib figure as a base64-encoded <img> tag string so it can
    be embedded inside a single st.markdown HTML block. This keeps the chart
    properly nested inside its card (st.pyplot in a separate call breaks the
    surrounding div), and it also has no built-in fullscreen-expand button, so
    it can't trigger Streamlit's white fullscreen overlay that clashed with our
    light-on-dark chart text."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", transparent=True, dpi=150)
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f'<img src="data:image/png;base64,{encoded}" style="width:100%; height:auto;" />'

# ==========================================
# Cached loaders (model + dataset load once, not on every rerun)
# ==========================================
@st.cache_resource
def load_model():
    artifact = joblib.load("best_attrition_model.pkl")
    return artifact["model"], artifact["feature_columns"]

@st.cache_data
def load_dataset():
    # Your repo's file has been named a couple different ways across pushes
    # (spaces vs underscores) — try the common variants instead of failing outright.
    candidate_names = [
        "IBM_HR_Employee_Attrition_Data.csv",
        "IBM HR Employee Attrition Data.csv",
        "IBM_HR_Employee_Attrition_Data (1).csv",
    ]
    for name in candidate_names:
        try:
            return pd.read_csv(name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("Dataset CSV not found under any known filename.")

try:
    model, feature_columns = load_model()
except FileNotFoundError:
    st.error("Model file 'best_attrition_model.pkl' not found. "
             "Make sure it's in the same folder as this app.")
    st.stop()

try:
    hr_data = load_dataset()
except FileNotFoundError:
    hr_data = None  # Insights page will show a friendly message instead of crashing

# ==========================================
# Sidebar Navigation
# ==========================================
with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio(
        "Go to",
        ["Predict Risk", "Insights", "About"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### Quick Facts")
    st.markdown("""
    <div class="chip chip-iceblue">Gradient Boosting</div>
    <div class="chip chip-sky">HR Analytics</div>
    <div class="chip">Recall-Focused</div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Built with Streamlit · Powered by scikit-learn")

# ==========================================
# Top Navbar
# ==========================================
st.markdown("""
<div class="navbar">
    <h1>Attrition Insights</h1>
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
if page == "Predict Risk":

    st.markdown("""
    <div class="hero">
        <h1>Employee Attrition Risk Predictor</h1>
        <p>Estimate the risk that an employee will leave the company, based on key
        workplace and demographic factors. Fill in the details below and click Predict.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### Employee Profile")

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
    predict_clicked = st.button("Predict Attrition Risk", type="primary", disabled=bool(errors))

    if predict_clicked:
        with st.spinner("Analyzing employee profile..."):
            time.sleep(0.6)  # brief pause purely for perceived feedback

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

        risk_label = "High Risk of Attrition" if prediction == 1 else "Low Risk of Attrition"
        risk_class = "risk-high" if prediction == 1 else "risk-low"
        bar_color_start = "#7f1d1d" if prediction == 1 else "#14532d"
        bar_color_end = "#ef4444" if prediction == 1 else "#22c55e"
        pct = probability * 100

        st.markdown(f"""
        <div class="card">
            <h3 style="color: var(--gold);">Result</h3>
            <div class="{risk_class}">
                {risk_label} &nbsp;|&nbsp; Predicted probability: {probability:.1%}
            </div>
            <div style="margin-top: 1.2rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:0.35rem;">
                    <span style="color: var(--muted); font-size:0.85rem;">Risk score</span>
                    <span style="color: var(--text-light); font-weight:700; font-size:0.95rem;">{probability:.1%}</span>
                </div>
                <div class="risk-track">
                    <div class="risk-threshold"></div>
                    <div class="risk-fill" style="--target-width: {pct:.1f}%; background: linear-gradient(90deg, {bar_color_start}, {bar_color_end});"></div>
                </div>
                <div style="display:flex; justify-content:space-between; margin-top:0.3rem;">
                    <span style="color: var(--muted); font-size:0.75rem;">0%</span>
                    <span style="color: var(--muted); font-size:0.75rem;">50% threshold</span>
                    <span style="color: var(--muted); font-size:0.75rem;">100%</span>
                </div>
            </div>
            <p style="color: var(--muted); font-size:0.85rem; margin-top:1.1rem; margin-bottom:0;">
                This estimate is based on the employee's profile compared against historical
                attrition patterns. Use it as a decision-support signal, not a sole determinant.
            </p>
        </div>
        """, unsafe_allow_html=True)

        if prediction == 1:
            st.toast("High attrition risk detected — consider a retention check-in.")
        else:
            st.toast("Low attrition risk — employee looks stable.")
            st.balloons()

# ==========================================
# PAGE: Insights (live data charts + model info, combined)
# ==========================================
elif page == "Insights":
    st.markdown("""
    <div class="hero">
        <h1>Attrition Insights</h1>
        <p>Real figures from the training dataset, and how the model itself was built.</p>
    </div>
    """, unsafe_allow_html=True)

    if hr_data is None:
        with st.container(border=True):
            st.warning(
                "Dataset file 'IBM_HR_Employee_Attrition_Data.csv' was not found alongside "
                "this app, so the live charts can't load. Make sure it's in the same folder "
                "(and pushed to your GitHub repo) as app.py."
            )
    else:
        overall_rate = (hr_data["Attrition"] == "Yes").mean()
        total_employees = len(hr_data)
        left_count = (hr_data["Attrition"] == "Yes").sum()

        st.markdown(f"""
        <div class="card">
            <div style="display:flex; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
                <div>
                    <div style="color: var(--muted); font-size:0.85rem;">Employees in dataset</div>
                    <div style="color: var(--text-light); font-size:1.8rem; font-weight:700;">{total_employees:,}</div>
                </div>
                <div>
                    <div style="color: var(--muted); font-size:0.85rem;">Left the company</div>
                    <div style="color: var(--text-light); font-size:1.8rem; font-weight:700;">{left_count:,}</div>
                </div>
                <div>
                    <div style="color: var(--muted); font-size:0.85rem;">Overall attrition rate</div>
                    <div style="color: var(--gold); font-size:1.8rem; font-weight:700;">{overall_rate:.1%}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            dept_rate = (
                hr_data.groupby("Department")["Attrition"]
                .apply(lambda s: (s == "Yes").mean())
                .sort_values(ascending=False)
            )
            palette = ["#e0b04b", "#6ea8e6", "#9fc0e8", "#3b6bb5", "#c9922f"]
            bar_colors = [palette[i % len(palette)] for i in range(len(dept_rate))]
            fig, ax = plt.subplots(figsize=(4.2, 3.2))
            fig.patch.set_alpha(0)
            ax.set_facecolor("none")
            bars = ax.barh(dept_rate.index, dept_rate.values, color=bar_colors)
            ax.set_xlabel("Attrition rate", color="#f4f7fc")
            ax.tick_params(colors="#f4f7fc")
            for spine in ax.spines.values():
                spine.set_color("#8ea3c4")
            ax.xaxis.label.set_color("#f4f7fc")
            for bar, val in zip(bars, dept_rate.values):
                ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val:.1%}", va="center", color="#f4f7fc", fontsize=9)
            dept_chart_html = fig_to_html_img(fig)

            st.markdown(f"""
            <div class="card card-amber">
                <h4 style="color: var(--gold); margin-top:0;">Attrition rate by department</h4>
                {dept_chart_html}
            </div>
            """, unsafe_allow_html=True)

        with col2:
            importances = pd.Series(model.feature_importances_, index=feature_columns)
            top_importances = importances.sort_values(ascending=False).head(8)
            palette2 = ["#9fc0e8", "#6ea8e6", "#3b6bb5", "#16305c", "#e0b04b"]
            bar_colors2 = [palette2[i % len(palette2)] for i in range(len(top_importances))][::-1]
            fig2, ax2 = plt.subplots(figsize=(4.2, 3.2))
            fig2.patch.set_alpha(0)
            ax2.set_facecolor("none")
            ax2.barh(top_importances.index[::-1], top_importances.values[::-1], color=bar_colors2)
            ax2.tick_params(colors="#f4f7fc", labelsize=8)
            for spine in ax2.spines.values():
                spine.set_color("#8ea3c4")
            importance_chart_html = fig_to_html_img(fig2)

            st.markdown(f"""
            <div class="card card-sky">
                <h4 style="color: var(--ice-blue); margin-top:0;">Top predictors (model feature importance)</h4>
                {importance_chart_html}
            </div>
            """, unsafe_allow_html=True)

        st.caption(
            "Department attrition rate is computed live from the training CSV. "
            "Feature importance comes directly from the trained Gradient Boosting model."
        )

    st.markdown("#### Under the hood: how the model was built")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card card-sky">
            <h3>Algorithm</h3>
            <p>Tuned Gradient Boosting Classifier (scikit-learn), selected after
            comparing against Decision Tree and Random Forest baselines, both
            untuned and hyperparameter-tuned via RandomizedSearchCV.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card card-violet">
            <h3>Evaluation Metrics</h3>
            <p>Accuracy, Precision, Recall, and F1-Score, evaluated on a held-out
            test set. Given the dataset's class imbalance (~84% stayed / 16% left),
            Recall and F1 were prioritized over raw accuracy.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-amber">
        <h3>Feature Engineering</h3>
        <p>Three engineered features were added to help the model: <b>TenureRatio</b>
        (company tenure vs. total career length), <b>LowWLB_Overtime</b> (poor work-life
        balance combined with overtime), and <b>IncomePerWorkingYear</b> (income scaled
        by career experience).</p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PAGE: About
# ==========================================
elif page == "About":
    st.markdown("""
    <div class="hero">
        <h1>About This Tool</h1>
        <p>Understand how and why this predictor works.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card card-sky">
        <h3>Purpose</h3>
        <p>This tool estimates the likelihood that an employee will leave the company
        (attrition), based on workplace and demographic factors drawn from the IBM HR
        Employee Attrition dataset. It's designed to help HR business partners identify
        at-risk employees early and act with targeted retention strategies.</p>
    </div>

    <div class="card card-violet">
        <h3>How It Works</h3>
        <p>You provide 10 key inputs (age, income, job role, overtime, tenure, etc.).
        Less-influential fields are filled with sensible dataset averages behind the
        scenes so the form stays quick to use. The trained model then returns a
        probability of attrition, alongside a High/Low risk label.</p>
    </div>

    <div class="card card-steel">
        <h3>Limitations</h3>
        <p>This is a decision-support signal, not a diagnosis. Predictions reflect
        historical patterns in the training data and should always be combined with
        human judgment and context before any HR action is taken.</p>
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
