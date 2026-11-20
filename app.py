

import streamlit as st
import pandas as pd
import joblib
import json

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="CardioAI | Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_model.pkl")

model = load_model()

with open("model_threshold.json", "r") as file:
    threshold = json.load(file)["threshold"]

# ---------------- CUSTOM STYLING ----------------

st.markdown("""
<style>
    .stApp {
        background-color: #f4f7fb;
    }

    .main-title {
        font-size: 38px;
        font-weight: 750;
        color: #142d4e;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #142d4e;
        font-size: 21px;
        font-weight: 650;
        margin-top: 10px;
        margin-bottom: 15px;
    }

    div[data-testid="stForm"] {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
    }

    div.stButton > button,
    div[data-testid="stFormSubmitButton"] > button {
        background-color: #1769aa;
        color: white;
        border-radius: 9px;
        border: none;
        font-weight: 600;
        padding: 12px;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
    }

    .result-heading {
        color: #142d4e;
        font-size: 25px;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------

st.markdown(
    '<div class="main-title">❤️ CardioAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Heart Disease Risk Prediction Using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.info(
    "An educational machine learning application that estimates "
    "the likelihood of heart disease using selected health parameters."
)

st.divider()

# ---------------- PATIENT INPUT FORM ----------------

st.markdown(
    '<div class="section-title">Patient Information</div>',
    unsafe_allow_html=True
)

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Personal Information")

        age = st.number_input(
            "Age (years)",
            min_value=28,
            max_value=77,
            value=54
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        st.markdown("### Medical Measurements")

        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80.0,
            max_value=200.0,
            value=130.0
        )

        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=85.0,
            max_value=603.0,
            value=240.0
        )

        thalch = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60.0,
            max_value=202.0,
            value=140.0
        )

        oldpeak = st.number_input(
            "Oldpeak",
            min_value=-2.6,
            max_value=6.2,
            value=0.5
        )

    with col2:

        st.markdown("### Heart Test Results")

        cp = st.selectbox(
            "Chest Pain Type",
            [
                "typical angina",
                "asymptomatic",
                "non-anginal",
                "atypical angina"
            ]
        )

        fbs = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            [True, False],
            format_func=lambda x: "Yes" if x else "No"
        )

        restecg = st.selectbox(
            "Resting ECG",
            [
                "lv hypertrophy",
                "normal",
                "st-t abnormality"
            ]
        )

        exang = st.selectbox(
            "Exercise-Induced Angina",
            [True, False],
            format_func=lambda x: "Yes" if x else "No"
        )

        slope = st.selectbox(
            "ST Segment Slope",
            [
                "downsloping",
                "flat",
                "upsloping"
            ]
        )

        ca = st.selectbox(
            "Number of Major Vessels",
            [0.0, 1.0, 2.0, 3.0],
            format_func=lambda x: str(int(x))
        )

        thal = st.selectbox(
            "Thalassemia",
            [
                "fixed defect",
                "normal",
                "reversable defect"
            ]
        )

    st.markdown("---")

    submit = st.form_submit_button(
        "🔍 Predict Heart Disease",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if submit:

    input_data = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    st.divider()

    st.markdown(
        '<div class="result-heading">Prediction Results</div>',
        unsafe_allow_html=True
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Model Score for Heart Disease",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        st.metric(
            "Classification Threshold",
            f"{threshold * 100:.0f}%"
        )

    st.progress(float(probability))

    if prediction == 1:

        st.error(
            "The model flags this input as potentially indicating "
            "heart disease."
        )

    else:

        st.success(
            "The model does not flag this input as indicating "
            "heart disease."
        )

    st.caption(
        "The model score is an estimated probability, not a "
        "clinically validated assessment of individual risk."
    )

    with st.expander("View submitted patient parameters"):

        st.dataframe(
            input_data,
            use_container_width=True
        )

    st.warning(
        "Educational project only. This model is not clinically "
        "validated and must not be used for diagnosis or treatment. "
        "Consult a qualified healthcare professional."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "CardioAI | Machine Learning Project | "
    "Built using Python, Scikit-learn and Streamlit"
)
