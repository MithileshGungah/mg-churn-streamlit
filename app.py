import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Churn Prediction Engine",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root{
        --bg-0:#0b0c10;
        --bg-1:#0f1117;
        --panel:#121528;
        --panel-2:#0f1224;
        --border:rgba(192,132,252,0.22);
        --border-soft:rgba(255,255,255,0.10);
        --text:#e9e9f3;
        --muted:rgba(233,233,243,0.72);
        --accent:#C084FC;
        --accent-2:#8B5CF6;
        --green:#2ecc71;
        --red:#ff4b4b;
        --amber:#ffb020;
        --cyan:#32d0ff;
    }

    .stApp {
        background:
            radial-gradient(900px 500px at 15% 0%, rgba(192,132,252,0.14), transparent 55%),
            radial-gradient(800px 600px at 85% 10%, rgba(46,204,113,0.10), transparent 60%),
            linear-gradient(180deg, var(--bg-0), var(--bg-1));
        color: var(--text);
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 1150px;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0.4rem;
    }

    [data-testid="stSidebarCollapseButton"] {
        display: flex !important;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(16,16,22,1) 0%, rgba(10,10,14,1) 100%);
        border-right: 1px solid rgba(192,132,252,0.16);
    }

    [data-testid="stSidebarContent"] {
        padding-top: 0.75rem;
        padding-bottom: 0.75rem;
    }

    .sidebar-card {
        background: transparent;
        border: none;
        padding: 0;
        margin-bottom: 16px;
        box-shadow: none;
        backdrop-filter: none;
    }

    .card-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(192,132,252,0.28), transparent);
        margin: 8px 0 12px 0;
        border-radius: 999px;
    }

    .card-title {
        font-size: 24px;
        font-weight: 800;
        margin: 2px 0 6px 0;
        letter-spacing: 0.2px;
        color: var(--accent);
    }

    .stSelectbox div[data-baseweb="select"],
    .stNumberInput div[data-baseweb="input"],
    .stTextInput div[data-baseweb="input"] {
        background: rgba(10,12,20,0.60) !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.20) !important;
        border-radius: 12px !important;
    }

    .stSelectbox div[data-baseweb="select"]:hover,
    .stNumberInput div[data-baseweb="input"]:hover,
    .stTextInput div[data-baseweb="input"]:hover {
        border: 1px solid rgba(192,132,252,0.28) !important;
    }

    .stSelectbox div[data-baseweb="select"] > div,
    .stNumberInput input,
    .stTextInput input,
    textarea {
        color: var(--text) !important;
    }

    label, .stMarkdown, .stText, p, span, div {
        color: var(--text);
    }

    div.stButton > button,
    div[data-testid="stDownloadButton"] > button {
        width: 100%;
        border-radius: 12px;
        font-weight: 800;
        letter-spacing: 0.2px;
        padding: 0.75rem 1rem;
        color: var(--text);
        background: linear-gradient(90deg, rgba(192,132,252,0.18), rgba(139,92,246,0.14));
        border: 1px solid rgba(192,132,252,0.35);
        box-shadow: 0 14px 30px rgba(0,0,0,0.25);
    }

    div[data-testid="stAlert"] {
    margin-top: 14px;
    }

    div.stButton > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(90deg, rgba(192,132,252,0.22), rgba(139,92,246,0.18));
        border: 1px solid rgba(192,132,252,0.50);
        transform: translateY(-1px);
    }

    [data-testid="stMetric"] {
        background: rgba(10,12,20,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.18);
    }

    .result-card{
        background: linear-gradient(180deg, rgba(18,21,40,0.72), rgba(10,12,20,0.55));
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 18px;
        padding: 14px 12px;
        box-shadow: 0 16px 36px rgba(0,0,0,0.28);
    }

    .badge{
        display:inline-flex;
        align-items:center;
        gap:8px;
        font-weight:800;
        padding:7px 12px;
        border-radius:999px;
        border:1px solid rgba(255,255,255,0.12);
        background: rgba(10,12,20,0.55);
        color: var(--text);
        font-size: 12px;
        line-height: 1;
    }

    .badge-low{ border-color: rgba(46,204,113,0.35); }
    .badge-med{ border-color: rgba(255,176,32,0.40); }
    .badge-high{ border-color: rgba(255,75,75,0.45); }

    .badge-prio-1{ border-color: rgba(255,75,75,0.55); }
    .badge-prio-2{ border-color: rgba(255,176,32,0.55); }
    .badge-prio-3{ border-color: rgba(192,132,252,0.55); }
    .badge-prio-none{ border-color: rgba(255,255,255,0.12); }

    .badge-decision-churn{
        border-color: rgba(255,75,75,0.45);
        background: rgba(255,75,75,0.14);
        box-shadow: inset 0 0 0 1px rgba(255,75,75,0.10);
        padding: 18px 28px;
    }

    .badge-decision-stay{
        border-color: rgba(46,204,113,0.40);
        background: rgba(46,204,113,0.14);
        box-shadow: inset 0 0 0 1px rgba(46,204,113,0.10);
        padding: 18px 28px;
    }

    .result-top{
        display:flex;
        justify-content:space-between;
        align-items:flex-start;
        gap:12px;
        flex-wrap:wrap;
        width:100%;
        margin-top: 10px;
        padding-top: 6px;
    }

    .result-left{
        display:flex;
        gap:12px;
        flex-wrap:wrap;
        align-items:center;
    }

    .result-right{
        margin-left:auto;
        font-weight:800;
        color: rgba(233,233,243,0.70);
        white-space:nowrap;
        padding-left: 10px;
    }

    .kpi-grid{
        display:flex;
        gap:10px;
        flex-wrap:wrap;
        margin-top:14px;
    }

    .kpi{
        flex: 1 1 160px;
        min-width: 160px;
        background: rgba(10,12,20,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 10px 22px rgba(0,0,0,0.18);
    }

    .kpi .label{
        color: rgba(233,233,243,0.70);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.2px;
    }

    .kpi .value{
        font-size: 20px;
        font-weight: 900;
        margin-top: 2px;
    }

    .prob-wrap{
        margin-top: 12px;
        padding: 12px 12px;
        background: rgba(10,12,20,0.30);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
    }

    .prob-top{
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap: 10px;
        margin-bottom: 10px;
        flex-wrap: wrap;
    }

    .prob-title{
        font-weight: 900;
        letter-spacing: 0.2px;
        color: rgba(233,233,243,0.88);
    }

    .prob-num{
        font-weight: 900;
        color: var(--accent);
    }

    .prob-bar{
        height: 10px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        overflow: hidden;
    }

    .prob-fill{
        height: 100%;
        width: 0%;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(46,204,113,0.55), rgba(192,132,252,0.55), rgba(255,75,75,0.60));
    }

    .section-h{
        font-size: 16px;
        font-weight: 900;
        color: rgba(233,233,243,0.92);
        margin: 14px 0 8px 0;
    }

    .pill{
        display:inline-block;
        padding: 6px 10px;
        margin: 4px 6px 0 0;
        border-radius: 999px;
        border: 1px solid rgba(192,132,252,0.25);
        background: rgba(192,132,252,0.08);
        color: rgba(233,233,243,0.88);
        font-weight: 800;
        font-size: 12px;
    }

    .small-muted{
        color: rgba(233,233,243,0.70);
        font-size: 12px;
        margin-top: 8px;
        line-height: 1.45;
    }

    [data-testid="stDataFrame"] {
        border-radius: 14px;
        overflow: hidden;
    }

    @media (max-width: 992px) {
        .block-container {
            max-width: 100%;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .card-title {
            font-size: 22px;
        }

        .result-right {
            margin-left: 0;
            padding-left: 0;
            white-space: normal;
            width: 100%;
        }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 0.75rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-bottom: 1.5rem;
        }

        h1 {
            font-size: 1.8rem !important;
            line-height: 1.2 !important;
        }

        .card-title {
            font-size: 20px;
        }

        .result-top,
        .result-left,
        .kpi-grid,
        .prob-top {
            flex-direction: column;
            align-items: stretch;
        }

        .result-left {
            gap: 8px;
        }

        .badge {
            width: fit-content;
            max-width: 100%;
        }

        .kpi {
            flex: 1 1 100%;
            min-width: 100%;
        }

        .kpi .value {
            font-size: 18px;
        }

        .result-card {
            padding: 8px 8px;
            border-radius: 16px;
        }

        .section-h {
            font-size: 15px;
        }

        .small-muted {
            font-size: 11.5px;
        }
    }

    @media (max-width: 480px) {
        .block-container {
            padding-left: 0.7rem;
            padding-right: 0.7rem;
        }

        h1 {
            font-size: 1.55rem !important;
        }

        .card-title {
            font-size: 18px;
        }

        .badge {
            font-size: 11px;
            padding: 7px 10px;
        }

        .kpi .value {
            font-size: 17px;
        }

        .pill {
            font-size: 11px;
            padding: 5px 9px;
        }

        .title-green {
        color: #2ecc71 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_artifacts():
    pipe = joblib.load("churn_xgb_pipeline.pkl")
    monthlycharges_median = joblib.load("monthlycharges_median.pkl")
    raw_input_columns = joblib.load("raw_input_columns.pkl")
    return pipe, monthlycharges_median, raw_input_columns


pipe, monthlycharges_median, raw_input_columns = load_artifacts()

EXPECTED_REMAINING_MONTHS_PROXY = 19.67
RETENTION_SUCCESS_RATE = 0.30

st.markdown(
    """
    <h1 class="title-green" style="margin-top:31px;">
        Churn Prediction Engine
    </h1>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
The **Churn Prediction Engine** predicts the likelihood that a customer will leave by analysing behavioural, service, and billing data.

Use the sidebar for a single prediction or upload a file below for batch predictions.
"""
)

st.divider()

def risk_badge(prob: float) -> tuple[str, str]:
    if prob < 0.33:
        return "Low", "badge-low"
    if prob < 0.66:
        return "Medium", "badge-med"
    return "High", "badge-high"


def priority_band(prob: float, pred: int) -> tuple[str, str]:
    if pred != 1:
        return "-", "badge-prio-none"
    if prob >= 0.80:
        return "P1", "badge-prio-1"
    if prob >= 0.60:
        return "P2", "badge-prio-2"
    return "P3", "badge-prio-3"


def model_confidence(prob: float) -> float:
    return min(1.0, abs(prob - 0.5) * 2)


def estimate_revenue_at_risk(monthly_charges: float) -> float:
    return float(monthly_charges) * EXPECTED_REMAINING_MONTHS_PROXY


def estimate_recoverable_value(monthly_charges: float) -> float:
    return estimate_revenue_at_risk(monthly_charges) * RETENTION_SUCCESS_RATE


def detected_drivers(inputs: dict) -> list[str]:
    drivers = []
    if inputs["Contract"] == "Month-to-month":
        drivers.append("Month-to-month contract")
    if inputs["InternetService"] == "Fiber optic":
        drivers.append("Fiber optic internet")
    if inputs["TechSupport"] == "No":
        drivers.append("No Tech Support")
    if inputs["OnlineSecurity"] == "No":
        drivers.append("No Online Security")
    if inputs["PaymentMethod"] == "Electronic check":
        drivers.append("Electronic check payments")
    if float(inputs["MonthlyCharges"]) > float(monthlycharges_median):
        drivers.append("High monthly charges")
    if int(inputs["tenure"]) < 6:
        drivers.append("Short tenure (<6 months)")
    return drivers[:6]


def recommended_actions(inputs: dict, pred: int, prob: float, prio: str) -> list[str]:
    actions = []

    if pred == 1:
        if prio == "P1":
            actions.append("P1: Escalate to retention outreach within 24–48 hours.")
        elif prio == "P2":
            actions.append("P2: Prioritise outreach within 3–5 days with targeted offer.")
        else:
            actions.append("P3: Add to retention nurture queue (light-touch intervention).")

    if inputs["Contract"] == "Month-to-month":
        actions.append("Offer a 12–24 month contract incentive to reduce churn risk.")
    if inputs["InternetService"] == "Fiber optic":
        actions.append("Check service quality; proactively handle service issues for fiber customers.")
    if inputs["TechSupport"] == "No":
        actions.append("Bundle or discount Tech Support for at-risk customers.")
    if inputs["OnlineSecurity"] == "No":
        actions.append("Add Online Security as a discounted retention add-on.")
    if inputs["PaymentMethod"] == "Electronic check":
        actions.append("Encourage autopay (bank/credit automatic) with a small monthly discount.")
    if float(inputs["MonthlyCharges"]) > float(monthlycharges_median):
        actions.append("Run a pricing review: propose a cheaper plan or loyalty discount.")
    if int(inputs["tenure"]) < 6:
        actions.append("Use an early-life retention touchpoint (welcome call / onboarding).")

    if not actions:
        actions.append("Maintain standard engagement; no strong churn indicators detected.")

    return actions[:8]


def build_features(inputs: dict) -> pd.DataFrame:
    tenure = int(inputs["tenure"])
    monthly = float(inputs["MonthlyCharges"])
    total = float(inputs["TotalCharges"])

    safe_tenure = tenure if tenure > 0 else 1

    row = {
        "gender": inputs["gender"],
        "SeniorCitizen": int(inputs["SeniorCitizen"]),
        "Partner": inputs["Partner"],
        "Dependents": inputs["Dependents"],
        "PhoneService": inputs["PhoneService"],
        "MultipleLines": inputs["MultipleLines"],
        "InternetService": inputs["InternetService"],
        "OnlineSecurity": inputs["OnlineSecurity"],
        "OnlineBackup": inputs["OnlineBackup"],
        "DeviceProtection": inputs["DeviceProtection"],
        "TechSupport": inputs["TechSupport"],
        "StreamingTV": inputs["StreamingTV"],
        "StreamingMovies": inputs["StreamingMovies"],
        "Contract": inputs["Contract"],
        "PaperlessBilling": inputs["PaperlessBilling"],
        "PaymentMethod": inputs["PaymentMethod"],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "AvgMonthlySpend": total / safe_tenure,
        "ShortTenure": int(tenure < 6),
        "HighMonthlyCharges": int(monthly > float(monthlycharges_median)),
    }

    df = pd.DataFrame([row])
    df = df.reindex(columns=raw_input_columns)
    return df


REQUIRED_INPUT_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]


def compute_priority_from_row(prob: float, pred: int) -> str:
    prio, _ = priority_band(prob, pred)
    return prio


def compute_risk_label(prob: float) -> str:
    if prob < 0.33:
        return "Low"
    if prob < 0.66:
        return "Medium"
    return "High"


def build_features_from_dataframe(df_in: pd.DataFrame) -> pd.DataFrame:
    df = df_in.copy()

    df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).astype(int)
    df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce").fillna(0.0).astype(float)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0).astype(float)

    safe_tenure = df["tenure"].clip(lower=1)
    df["AvgMonthlySpend"] = df["TotalCharges"] / safe_tenure
    df["ShortTenure"] = (df["tenure"] < 6).astype(int)
    df["HighMonthlyCharges"] = (df["MonthlyCharges"] > float(monthlycharges_median)).astype(int)

    df = df.reindex(columns=raw_input_columns)

    return df


with st.sidebar:
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Customer Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-divider'></div>", unsafe_allow_html=True)

    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_label = st.selectbox("Senior Citizen", ["No", "Yes"])
    senior = 1 if senior_label == "Yes" else 0
    partner = st.selectbox("Partner", ["No", "Yes"])
    dependents = st.selectbox("Dependents", ["No", "Yes"])

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Account Charges</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-divider'></div>", unsafe_allow_html=True)

    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    monthly_charges = st.number_input("Monthly Charges (£)", 0.0, 500.0, 70.0)
    total_charges = st.number_input("Total Charges (£)", 0.0, 20000.0, 1000.0)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Billing Information</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-divider'></div>", unsafe_allow_html=True)

    contract = st.selectbox("Contract Type", ["Month-to-Month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method = st.selectbox(
        "Payment Method",
        [
            "Bank Transfer (Automatic)",
            "Credit Card (Automatic)",
            "Electronic Check",
            "Mailed Check",
        ],
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.markdown("<div class='card-title'>Service Subscriptions</div>", unsafe_allow_html=True)
    st.markdown("<div class='card-divider'></div>", unsafe_allow_html=True)

    phone = st.selectbox("Phone Service", ["No", "Yes"])
    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    addon_choices = ["No internet service"] if internet == "No" else ["No", "Yes"]
    online_security = st.selectbox("Online Security", addon_choices)
    online_backup = st.selectbox("Online Backup", addon_choices)
    device_protection = st.selectbox("Device Protection", addon_choices)
    tech_support = st.selectbox("Tech Support", addon_choices)
    streaming_tv = st.selectbox("Streaming TV", addon_choices)
    streaming_movies = st.selectbox("Streaming Movies", addon_choices)

    st.markdown("</div>", unsafe_allow_html=True)

    run = st.button("Predict", use_container_width=True)

st.subheader("Prediction Result")

if run:
    user_inputs = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment_method,
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    X = build_features(user_inputs)
    pred = int(pipe.predict(X)[0])

    proba = None
    if hasattr(pipe, "predict_proba"):
        proba = float(pipe.predict_proba(X)[0][1])

    if proba is None:
        st.warning("Model probability output not available. Showing class prediction only.")
        if pred == 1:
            st.error("This customer is at risk of churning.")
        else:
            st.success("This customer is likely to remain with the company.")
    else:
        r_label, r_class = risk_badge(proba)
        prio, prio_class = priority_band(proba, pred)
        conf = model_confidence(proba)
        decision_class = "badge-decision-churn" if pred == 1 else "badge-decision-stay"

        st.markdown("<div class='result-card'>", unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="result-top">
                <div class="result-left">
                    <span class="badge {decision_class}">Decision: {"CHURN" if pred==1 else "STAY"}</span>
                    <span class="badge {r_class}">Risk: {r_label}</span>
                    <span class="badge {prio_class}">Priority: {prio}</span>
                </div>
                <div class="result-right">Recall-focused Screening Model</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="kpi-grid">
                <div class="kpi">
                    <div class="label">Churn Probability</div>
                    <div class="value">{proba*100:.1f}%</div>
                </div>
                <div class="kpi">
                    <div class="label">Model Confidence</div>
                    <div class="value">{conf*100:.0f}%</div>
                </div>
                <div class="kpi">
                    <div class="label">Priority Tier</div>
                    <div class="value">{prio}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if pred == 1:
            if prio == "P1":
                st.error("P1: Very high churn risk - Prioritise proactive retention outreach.")
            elif prio == "P2":
                st.warning("P2: High churn risk - Prioritise targeted retention actions.")
            else:
                st.info("P3: Moderate churn risk - Add to retention nurture pipeline.")
        else:
            st.success("This customer is likely to remain with the company.")

        st.markdown(
            f"""
            <div class="prob-wrap">
                <div class="prob-top">
                    <div class="prob-title">Risk Meter</div>
                    <div class="prob-num">{proba*100:.1f}%</div>
                </div>
                <div class="prob-bar">
                    <div class="prob-fill" style="width:{min(100, max(0, proba*100)):.1f}%"></div>
                </div>
                <div class="small-muted">
                    Confidence is derived from distance to the decision boundary (0.50).
                    Priority tiers (if predicted churn): P1 ≥ 80%, P2 ≥ 60%, otherwise P3.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if r_label != "Low":
            drivers = detected_drivers(user_inputs)
            st.markdown("<div class='section-h'>Detected Drivers</div>", unsafe_allow_html=True)
            if drivers:
                pills = "".join([f"<span class='pill'>{d}</span>" for d in drivers])
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown(
                    "<div class='small-muted'>No strong heuristic drivers detected from inputs.</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<div class='section-h'>Recommended Actions</div>", unsafe_allow_html=True)
            actions = recommended_actions(user_inputs, pred, proba, prio)
            for a in actions:
                st.markdown(f"- {a}")

        if pred == 1:
            monthly_revenue = float(user_inputs["MonthlyCharges"])
            revenue_at_risk = estimate_revenue_at_risk(monthly_revenue)
            recoverable_value = estimate_recoverable_value(monthly_revenue)

            st.markdown("<div class='section-h'>Estimated Revenue Impact</div>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="kpi-grid">
                    <div class="kpi">
                        <div class="label">Monthly Revenue</div>
                        <div class="value">£{monthly_revenue:,.2f}</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Revenue At Risk</div>
                        <div class="value">£{revenue_at_risk:,.0f}</div>
                    </div>
                    <div class="kpi">
                        <div class="label">Recoverable Value</div>
                        <div class="value">£{recoverable_value:,.0f}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="small-muted">
                    Revenue at risk is estimated as <b>Monthly Charges × {EXPECTED_REMAINING_MONTHS_PROXY:.2f}</b> months.
                    Recoverable value assumes a <b>{RETENTION_SUCCESS_RATE:.0%}</b> retention success rate.
                    This is a directional business estimate for single-customer review.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("")

else:
    st.info("Fill inputs in the sidebar and click Predict.")

st.divider()
st.subheader("Batch Prediction (CSV Upload)")

st.markdown(
    """
Upload a CSV containing **exactly these columns** (case-sensitive):

- gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup,
  DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
  tenure, MonthlyCharges, TotalCharges
"""
)

uploaded = st.file_uploader("Upload CSV", type=["csv"])

template_df = pd.DataFrame(columns=REQUIRED_INPUT_COLUMNS)
st.download_button(
    "Download CSV Template",
    data=template_df.to_csv(index=False).encode("utf-8"),
    file_name="churn_batch_template.csv",
    mime="text/csv",
)

if uploaded is not None:
    try:
        batch_raw = pd.read_csv(uploaded)

        missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in batch_raw.columns]
        extra = [c for c in batch_raw.columns if c not in REQUIRED_INPUT_COLUMNS]

        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            if extra:
                st.warning(f"Extra columns detected (they will be ignored): {extra}")

            batch_in = batch_raw[REQUIRED_INPUT_COLUMNS].copy()
            Xb = build_features_from_dataframe(batch_in)

            batch_pred = pipe.predict(Xb).astype(int)
            if hasattr(pipe, "predict_proba"):
                batch_prob = pipe.predict_proba(Xb)[:, 1].astype(float)
            else:
                batch_prob = pd.Series([None] * len(batch_in))

            out = batch_in.copy()
            out["Churn_Probability"] = batch_prob
            out["Decision"] = ["Churn" if p == 1 else "Stay" for p in batch_pred]
            out["Risk_Level"] = [compute_risk_label(p) if p is not None else None for p in batch_prob]
            out["Priority"] = [
                compute_priority_from_row(p, pr) if p is not None else ("-" if pr != 1 else "P3")
                for p, pr in zip(batch_prob, batch_pred)
            ]

            st.success(f"Batch predictions complete: {len(out)} rows")
            st.dataframe(out, use_container_width=True)

            st.download_button(
                "Download Results CSV",
                data=out.to_csv(index=False).encode("utf-8"),
                file_name="churn_batch_predictions.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Could not process file: {e}")