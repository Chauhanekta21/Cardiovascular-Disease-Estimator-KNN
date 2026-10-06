from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "knn_model.joblib"
SCALER_PATH = APP_DIR / "models" / "minmax_scaler.joblib"
FEATURES = [
    "age",
    "height",
    "weight",
    "systolic_bp",
    "diastolic_bp",
    "cholesterol_level",
    "glucose_level",
    "smoking",
    "alcohol",
    "physical_activity",
]


st.set_page_config(
    page_title="Cardiovascular Disease Estimator",
    page_icon="♥",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
        --crimson: #a51f35;
        --crimson-dark: #7f1729;
        --ink: #202328;
        --muted: #687078;
        --line: #e5e5e3;
        --paper: #f7f7f5;
    }
    .stApp {
        background: radial-gradient(ellipse at 90% 0%, #f4e9e8 0, transparent 27rem), var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    .block-container { max-width: 1320px; padding-top: 3.5rem; padding-bottom: 2rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; color: var(--ink); }
    .hero { border-bottom: 1px solid var(--line); padding: 0 0 1.5rem; margin-bottom: 1.4rem; }
    .eyebrow { color: var(--crimson); display: block; font-size: .76rem; font-weight: 700; letter-spacing: .12em; line-height: 1.6; padding-top: .2rem; margin-bottom: .55rem; overflow: visible; }
    .hero-title { font: 800 2.35rem/1.15 'Manrope', sans-serif; margin: 0; letter-spacing: 0; }
    .hero-copy { color: var(--muted); font-size: 1rem; margin: .65rem 0 0; max-width: 700px; }
    .section-title { font: 700 1.05rem 'Manrope', sans-serif; margin: 1rem 0 .1rem; }
    .section-copy { color: var(--muted); font-size: .86rem; margin: 0 0 .8rem; }
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label,
    [data-testid="stWidgetLabel"] span {
        color: #202b36 !important;
        opacity: 1 !important;
    }
    [data-testid="stWidgetLabel"] p {
        font-family: 'DM Sans', sans-serif; font-size: .9rem;
        font-weight: 600; margin-bottom: .35rem;
    }
    div[data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {
        color: #4b402c !important;
    }
    div[data-testid="stNumberInput"] input, div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: #d7d8d7; border-radius: 5px;
    }
    div[data-testid="stFormSubmitButton"] button {
        background: var(--crimson); color: white; border: 1px solid var(--crimson);
        border-radius: 5px; font-weight: 700; min-height: 3rem;
    }
    div[data-testid="stFormSubmitButton"] button:hover { background: var(--crimson-dark); border-color: var(--crimson-dark); color: white; }
    div[data-testid="stVerticalBlockBorderWrapper"] { border-color: var(--line); border-radius: 7px; background: rgba(255,255,255,.78); }
    .panel-title { font: 700 1rem 'Manrope', sans-serif; margin: 0 0 .65rem; }
    .panel-copy { color: var(--muted); font-size: .9rem; line-height: 1.55; }
    .result-card { border-radius: 6px; padding: 1.15rem; border: 1px solid; margin-top: .4rem; }
    .result-card.positive { background: #fbefef; border-color: #e6b7bd; }
    .result-card.negative { background: #edf4f0; border-color: #bfd5c8; }
    .result-label { color: var(--muted); font-size: .77rem; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
    .result-value { font: 800 1.35rem 'Manrope', sans-serif; margin-top: .35rem; }
    .result-card.positive .result-value { color: var(--crimson-dark); }
    .result-card.negative .result-value { color: #285d43; }
    .disclaimer { border-top: 1px solid var(--line); color: var(--muted); font-size: .78rem; margin-top: 2rem; padding-top: .9rem; }
    @media (max-width: 700px) {
        .block-container { padding-top: 2.25rem; }
        .hero-title { font-size: 1.8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
        <header class="hero">
            <div class="eyebrow">MACHINE LEARNING MODEL · K-NEAREST NEIGHBORS ALGORITHM</div>
            <h1 class="hero-title">Cardiovascular Disease Estimator</h1>
            <p class="hero-copy">Explore a model-based estimate using health and lifestyle information.</p>
    </header>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    if list(scaler.feature_names_in_) != FEATURES:
        raise ValueError("The saved scaler feature order does not match the app inputs.")
    if model.n_features_in_ != len(FEATURES):
        raise ValueError("The saved model does not match the expected feature count.")
    return model, scaler


try:
    knn_model, scaler = load_artifacts()
except FileNotFoundError:
    st.error("The trained model files are missing. Run `python build_artifacts.py` from the streamlit_app folder first.")
    st.stop()
except (ValueError, OSError, AttributeError) as error:
    st.error(f"The saved model files could not be loaded: {error}")
    st.stop()


left, right = st.columns([1.75, 1], gap="large")

with left:
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Personal Information</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Enter age in years and select gender.</p>', unsafe_allow_html=True)
        age_col, gender_col = st.columns(2)
        with age_col:
            age = st.number_input("Age (years)", min_value=18, max_value=100, value=50, step=1)
        with gender_col:
            gender = st.selectbox("Gender", options=["Female", "Male"])

        st.divider()
        st.markdown('<div class="section-title">Physical Measurements</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Use centimetres for height and kilograms for weight.</p>', unsafe_allow_html=True)
        height_col, weight_col = st.columns(2)
        with height_col:
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=165, step=1)
        with weight_col:
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=200.0, value=72.0, step=0.5)

        st.divider()
        st.markdown('<div class="section-title">Blood Pressure</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Systolic pressure must be higher than diastolic pressure.</p>', unsafe_allow_html=True)
        systolic_col, diastolic_col = st.columns(2)
        with systolic_col:
            systolic_bp = st.number_input("Systolic Blood Pressure (mmHg)", min_value=1, max_value=250, value=120, step=1)
        with diastolic_col:
            diastolic_bp = st.number_input("Diastolic Blood Pressure (mmHg)", min_value=1, max_value=249, value=80, step=1)

        st.divider()
        st.markdown('<div class="section-title">Health &amp; Lifestyle</div>', unsafe_allow_html=True)
        st.markdown('<p class="section-copy">Select the category that best matches each health factor.</p>', unsafe_allow_html=True)
        cholesterol_col, glucose_col = st.columns(2)
        with cholesterol_col:
            cholesterol = st.selectbox(
                "Cholesterol Level",
                options=[1, 2, 3],
                format_func=lambda value: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[value],
            )
        with glucose_col:
            glucose = st.selectbox(
                "Glucose Level",
                options=[1, 2, 3],
                format_func=lambda value: {1: "Normal", 2: "Above normal", 3: "Well above normal"}[value],
            )
        smoke_col, alcohol_col = st.columns(2)
        with smoke_col:
            smoking = st.selectbox("Smoking", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
        with alcohol_col:
            alcohol = st.selectbox("Alcohol Consumption", options=[0, 1], format_func=lambda value: "No" if value == 0 else "Yes")
        activity_col, _ = st.columns(2)
        with activity_col:
            physical_activity = st.selectbox("Physical Activity", options=[1, 0], format_func=lambda value: "Yes" if value == 1 else "No")

        st.caption("Gender is collected for profile context. It is not used as a model feature in the training notebook.")
        submitted = st.form_submit_button("Estimate Cardiovascular Risk", use_container_width=True)

    if submitted and (age < 30 or age > 65):
        st.warning("⚠️ This age is outside the model's training range (30–65 years), so the prediction may be less reliable.")

    if submitted:
        if systolic_bp <= diastolic_bp:
            st.session_state.pop("prediction", None)
            st.error("Check the blood pressure values: systolic must be higher than diastolic.")
        else:
            input_values = {
                "age": age,
                "height": height,
                "weight": weight,
                "systolic_bp": systolic_bp,
                "diastolic_bp": diastolic_bp,
                "cholesterol_level": cholesterol,
                "glucose_level": glucose,
                "smoking": smoking,
                "alcohol": alcohol,
                "physical_activity": physical_activity,
            }
            input_frame = pd.DataFrame([input_values], columns=FEATURES)
            scaled_input = scaler.transform(input_frame)
            st.session_state["prediction"] = int(knn_model.predict(scaled_input)[0])

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-title">About this model</div>', unsafe_allow_html=True)
        st.markdown(
            '<p class="panel-copy">This project uses a trained KNN classifier. Inputs are scaled with the project Min-Max scaler and passed in the same feature order used during training.</p>',
            unsafe_allow_html=True,
        )
        st.caption("Trained KNN estimator · Personalized cardiovascular risk prediction")

    if "prediction" in st.session_state:
        result = st.session_state["prediction"]
        if result == 1:
            result_class = "positive"
            result_text = "Cardiovascular Disease Detected"
            detail = "The model classified this input in the cardiovascular disease group."
        else:
            result_class = "negative"
            result_text = "No Cardiovascular Disease Detected"
            detail = "The model classified this input in the no cardiovascular disease group."
        st.markdown(
            f'<div class="result-card {result_class}"><div class="result-label">Prediction result</div><div class="result-value">{result_text}</div><p class="panel-copy">{detail}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        with st.container(border=True):
            st.markdown('<div class="panel-title">Prediction result</div>', unsafe_allow_html=True)
            st.markdown('<p class="panel-copy">Complete the form and generate a prediction to see the model result here.</p>', unsafe_allow_html=True)

st.markdown(
    '<div class="disclaimer">Educational use only. This prediction is not a medical diagnosis and should not replace advice from a qualified healthcare professional.</div>',
    unsafe_allow_html=True,
)