import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from src import features as feat

st.set_page_config(
    page_title="Diabetes Health Indicators Dashboard",
    page_icon="\U0001fa7a",
    layout="wide",
)

pio.templates.default = "plotly_dark"
pio.templates["plotly_dark"].layout.font.family = "Inter, -apple-system, sans-serif"
pio.templates["plotly_dark"].layout.font.color = "#e2e8f0"
pio.templates["plotly_dark"].layout.paper_bgcolor = "#1e293b"
pio.templates["plotly_dark"].layout.plot_bgcolor = "#1e293b"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #312e81 0%, #4f46e5 55%, #0ea5e9 100%);
        padding: 1.75rem 2.25rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 28px rgba(79, 70, 229, 0.35);
    }
    .hero-banner h1 {
        color: #ffffff;
        margin: 0;
        font-weight: 800;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: rgba(255,255,255,0.88);
        margin: 0.35rem 0 0 0;
        font-size: 1.02rem;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    }
    [data-testid="stMetricLabel"] {
        font-weight: 600;
        color: #94a3b8;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 1px solid #334155;
    }
    .stTabs [data-baseweb="tab"] {
        height: 46px;
        background-color: transparent;
        border-radius: 10px 10px 0 0;
        padding: 0 1.25rem;
        font-weight: 600;
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background-color: #312e81;
        color: #a5b4fc;
    }

    /* Bordered containers (cards) */
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 12px;
    }

    /* Risk result card */
    .risk-card {
        border-radius: 12px;
        padding: 1.1rem 1.5rem;
        border-left: 6px solid;
        margin-bottom: 1rem;
    }
    .risk-card h3 {
        margin: 0;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

TARGET_COLORS = {0: "#2ecc71", 1: "#f1c40f", 2: "#e74c3c"}
TARGET_LABELS = {0: "No Diabetes", 1: "Prediabetes", 2: "Diabetes"}
TARGET_ORDER = [0, 1, 2]
CLASS_NAMES = [TARGET_LABELS[c] for c in TARGET_ORDER]
COLOR_MAP = {TARGET_LABELS[c]: TARGET_COLORS[c] for c in TARGET_ORDER}

AGE_MAP = {
    1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44", 6: "45-49",
    7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69", 11: "70-74", 12: "75-79", 13: "80+",
}
GENHLTH_MAP = {1: "Excellent", 2: "Very Good", 3: "Good", 4: "Fair", 5: "Poor"}
EDUCATION_MAP = {
    1: "None/Kindergarten", 2: "Elementary", 3: "Some High School",
    4: "High School Grad", 5: "Some College", 6: "College Grad",
}
INCOME_MAP = {
    1: "<$10k", 2: "$10-15k", 3: "$15-20k", 4: "$20-25k",
    5: "$25-35k", 6: "$35-50k", 7: "$50-75k", 8: "$75k+",
}
BMI_CAT_MAP = {0: "Underweight", 1: "Normal", 2: "Overweight", 3: "Obese"}
SEX_MAP = {0: "Female", 1: "Male"}

BINARY_FEATURES = {
    "High Blood Pressure": "HighBP",
    "High Cholesterol": "HighChol",
    "Smoker": "Smoker",
    "Stroke History": "Stroke",
    "Heart Disease / Attack": "HeartDiseaseorAttack",
    "Physical Activity": "PhysActivity",
    "Eats Fruit Daily": "Fruits",
    "Eats Vegetables Daily": "Veggies",
    "Heavy Alcohol Use": "HvyAlcoholConsump",
    "Difficulty Walking": "DiffWalk",
}
LABELED_CAT_FEATURES = {
    "Age Group": "AgeGroup",
    "Sex": "SexLabel",
    "General Health": "GenHealthLabel",
    "Education": "EducationLabel",
    "Income Level": "IncomeLabel",
    "BMI Category": "BMICategoryLabel",
}
NUMERIC_FEATURES = {
    "BMI": "BMI",
    "General Health (1=Excellent, 5=Poor)": "GenHlth",
    "Mental Health (poor days / 30)": "MentHlth",
    "Physical Health (poor days / 30)": "PhysHlth",
    "Comorbidity Score (0-4)": "ComorbidityScore",
    "Lifestyle Score (0-5)": "LifestyleScore",
}

FEATURE_DESCRIPTIONS = {
    "HighBP": "Diagnosed with high blood pressure",
    "HighChol": "Diagnosed with high cholesterol",
    "CholCheck": "Had a cholesterol check in the past 5 years",
    "BMI": "Body Mass Index",
    "Smoker": "Smoked 100+ cigarettes in lifetime",
    "Stroke": "Ever told had a stroke",
    "HeartDiseaseorAttack": "Coronary heart disease or myocardial infarction",
    "PhysActivity": "Physical activity in past 30 days (not job-related)",
    "Fruits": "Consumes fruit 1+ times per day",
    "Veggies": "Consumes vegetables 1+ times per day",
    "HvyAlcoholConsump": "Heavy alcohol consumption (>14 drinks/wk men, >7 women)",
    "AnyHealthcare": "Has any form of health insurance/coverage",
    "NoDocbcCost": "Could not see a doctor in past year due to cost",
    "GenHlth": "Self-rated general health (1=Excellent ... 5=Poor)",
    "MentHlth": "Days of poor mental health in past 30 days",
    "PhysHlth": "Days of poor physical health in past 30 days",
    "DiffWalk": "Serious difficulty walking or climbing stairs",
    "Sex": "0 = Female, 1 = Male",
    "Age": "13-level age category (1=18-24 ... 13=80+)",
    "Education": "6-level education category",
    "Income": "8-level household income category",
    "BMICategory": "Engineered: WHO BMI band (0=Underweight ... 3=Obese)",
    "LifestyleScore": "Engineered: 0-5, count of healthy behaviours present",
    "ComorbidityScore": "Engineered: 0-4, count of HighBP/HighChol/Stroke/HeartDisease",
    "Diabetes_012": "Target: 0=No Diabetes, 1=Prediabetes, 2=Diabetes",
}


@st.cache_data
def load_data():
    df = pd.read_csv(ROOT / "data/processed/diabetes_engineered.csv")
    df["DiabetesStatus"] = df["Diabetes_012"].astype(int).map(TARGET_LABELS)
    df["AgeGroup"] = df["Age"].astype(int).map(AGE_MAP)
    df["GenHealthLabel"] = df["GenHlth"].astype(int).map(GENHLTH_MAP)
    df["EducationLabel"] = df["Education"].astype(int).map(EDUCATION_MAP)
    df["IncomeLabel"] = df["Income"].astype(int).map(INCOME_MAP)
    df["SexLabel"] = df["Sex"].astype(int).map(SEX_MAP)
    df["BMICategoryLabel"] = df["BMICategory"].astype(int).map(BMI_CAT_MAP)
    return df


@st.cache_resource
def load_model():
    return joblib.load(ROOT / "models/lightgbm_diabetes_final.pkl")


@st.cache_data
def load_metadata():
    with open(ROOT / "models/model_metadata.json") as f:
        return json.load(f)


@st.cache_data
def load_shap_importance():
    s = pd.read_csv(ROOT / "models/shap_feature_importance.csv", index_col=0)
    return s.iloc[:, 0].sort_values(ascending=False)


@st.cache_data
def load_model_comparison():
    return pd.read_csv(ROOT / "models/model_comparison.csv", index_col=0)


@st.cache_data
def get_test_evaluation(_model, model_path):
    test_df = pd.read_csv(ROOT / "data/processed/test.csv")
    X_test = test_df[feat.FEATURES]
    y_test = test_df[feat.TARGET].astype(int)
    y_pred = _model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, normalize="true")
    report = classification_report(
        y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    return cm, report


df = load_data()
model = load_model()
metadata = load_metadata()
shap_importance = load_shap_importance()
comparison = load_model_comparison()
tm = metadata["test_metrics"]

with st.sidebar:
    st.markdown("## \U0001fa7a Diabetes Insights")
    st.caption("BRFSS 2015 · ML-powered risk screening demo")
    st.divider()
    st.markdown("**Final Model — Tuned LightGBM**")
    st.metric("Balanced Accuracy", f"{tm['Balanced Accuracy']:.1%}")
    st.metric("Macro ROC-AUC", f"{tm['Macro ROC-AUC']:.3f}")
    st.metric("Macro F1", f"{tm['Macro F1']:.3f}")
    st.divider()
    st.markdown(
        "**Navigate**\n"
        "- \U0001f4ca Overview — dataset & key facts\n"
        "- \U0001f50d Interactive EDA — filter & explore\n"
        "- \U0001f9ee Risk Predictor — try the model\n"
        "- \U0001f916 Model Insights — performance & SHAP"
    )
    st.divider()
    st.caption("Built with scikit-learn · LightGBM · SHAP · Streamlit")

st.markdown(
    f"""
    <div class="hero-banner">
        <h1>\U0001fa7a Diabetes Health Indicators Dashboard</h1>
        <p>BRFSS 2015 Health Indicators · {len(df):,} respondents ·
        Multiclass risk model (No Diabetes / Prediabetes / Diabetes)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_overview, tab_eda, tab_predict, tab_model = st.tabs(
    ["\U0001f4ca Overview", "\U0001f50d Interactive EDA", "\U0001f9ee Risk Predictor", "\U0001f916 Model Insights"]
)

# ---------------------------------------------------------------------------
# Tab 1: Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.header("Dataset Overview")

    counts = df["Diabetes_012"].value_counts().reindex(TARGET_ORDER)
    pct = counts / counts.sum() * 100

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Respondents", f"{len(df):,}")
    c2.metric("No Diabetes", f"{pct[0]:.2f}%", f"{counts[0]:,} respondents")
    c3.metric("Prediabetes", f"{pct[1]:.2f}%", f"{counts[1]:,} respondents")
    c4.metric("Diabetes", f"{pct[2]:.2f}%", f"{counts[2]:,} respondents")

    left, right = st.columns([1, 1.4])

    with left:
        fig = px.pie(
            names=[TARGET_LABELS[c] for c in TARGET_ORDER],
            values=counts.values,
            color=[TARGET_LABELS[c] for c in TARGET_ORDER],
            color_discrete_map=COLOR_MAP,
            hole=0.45,
            title="Target Distribution",
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        with st.container(border=True):
            st.markdown(
                """
**About this dataset**

The Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey,
cleaned and prepared by the CDC, contains **22 health indicators** for
253,680 U.S. adults. The target, `Diabetes_012`, is severely imbalanced:
the vast majority of respondents report no diabetes.

**Key dataset facts**
- 0 missing values across all 22 original columns
- 9.42% duplicate rows — expected given the limited combinations of
  categorical survey responses, not a data quality issue
- 3 engineered features added during feature engineering:
  `BMICategory`, `LifestyleScore`, `ComorbidityScore`
- BMI capped at 52 (Q3 + 3×IQR) to control extreme outliers (~0.67% of rows)

**Strongest univariate risk factors** (Cramér's V / EDA notebook)
1. HighBP (0.272)
2. DiffWalk (0.224)
3. GenHlth (0.219)
4. HighChol (0.211)
5. HeartDiseaseorAttack (0.180)
                """
            )

    with st.expander("\U0001f4d6 Feature Glossary"):
        glossary = pd.DataFrame(
            [{"Feature": k, "Description": v} for k, v in FEATURE_DESCRIPTIONS.items()]
        )
        st.dataframe(glossary, use_container_width=True, hide_index=True)

    st.subheader("Executive Summary")
    with st.container(border=True):
        st.image(str(ROOT / "reports/figures/10_executive_summary.png"), use_column_width=True)

# ---------------------------------------------------------------------------
# Tab 2: Interactive EDA
# ---------------------------------------------------------------------------
with tab_eda:
    st.header("Interactive Exploration")
    st.caption("Filter the population below — every chart on this tab updates live.")

    with st.container(border=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            age_sel = st.multiselect(
                "Age Group", options=list(AGE_MAP.values()), default=list(AGE_MAP.values())
            )
        with f2:
            sex_sel = st.selectbox("Sex", options=["All", "Female", "Male"])
        with f3:
            bmi_sel = st.multiselect(
                "BMI Category", options=list(BMI_CAT_MAP.values()), default=list(BMI_CAT_MAP.values())
            )

    filtered = df[df["AgeGroup"].isin(age_sel) & df["BMICategoryLabel"].isin(bmi_sel)]
    if sex_sel != "All":
        filtered = filtered[filtered["SexLabel"] == sex_sel]

    st.caption(f"Showing **{len(filtered):,}** of {len(df):,} respondents")

    if len(filtered) == 0:
        st.warning("No respondents match the selected filters.")
    else:
        col_a, col_b = st.columns(2)

        with col_a:
            num_choice = st.selectbox("Numeric feature distribution", options=list(NUMERIC_FEATURES.keys()))
            num_col = NUMERIC_FEATURES[num_choice]
            fig_hist = px.histogram(
                filtered, x=num_col, color="DiabetesStatus", barmode="overlay",
                histnorm="percent", opacity=0.6, nbins=30,
                color_discrete_map=COLOR_MAP,
                category_orders={"DiabetesStatus": CLASS_NAMES},
                title=f"{num_choice} — Distribution by Diabetes Status",
            )
            fig_hist.update_layout(yaxis_title="Percent of class")
            st.plotly_chart(fig_hist, use_container_width=True)

        with col_b:
            combined_cat = {**LABELED_CAT_FEATURES, **BINARY_FEATURES}
            cat_choice = st.selectbox("Diabetes prevalence by category", options=list(combined_cat.keys()))
            cat_col = combined_cat[cat_choice]

            cat_data = filtered.copy()
            if cat_choice in BINARY_FEATURES:
                cat_data[cat_col] = cat_data[cat_col].map({0: "No", 1: "Yes"})
                cat_order = ["No", "Yes"]
            elif cat_col == "AgeGroup":
                cat_order = list(AGE_MAP.values())
            elif cat_col == "GenHealthLabel":
                cat_order = list(GENHLTH_MAP.values())
            elif cat_col == "EducationLabel":
                cat_order = list(EDUCATION_MAP.values())
            elif cat_col == "IncomeLabel":
                cat_order = list(INCOME_MAP.values())
            elif cat_col == "BMICategoryLabel":
                cat_order = list(BMI_CAT_MAP.values())
            else:
                cat_order = sorted(cat_data[cat_col].dropna().unique().tolist())

            ct = pd.crosstab(cat_data[cat_col], cat_data["DiabetesStatus"], normalize="index") * 100
            ct = ct.reindex(cat_order)[CLASS_NAMES]

            fig_bar = px.bar(
                ct, barmode="stack",
                color_discrete_map=COLOR_MAP,
                title=f"Diabetes Status by {cat_choice} (% within group)",
                labels={"value": "Percent", "index": cat_choice},
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Correlation with Diabetes Status (filtered subset)")
        corr_cols = feat.FEATURES
        corr = filtered[corr_cols + ["Diabetes_012"]].corr()["Diabetes_012"].drop("Diabetes_012")
        corr = corr.sort_values()
        fig_corr = px.bar(
            corr, orientation="h",
            title="Pearson correlation with Diabetes_012",
            labels={"value": "Correlation", "index": "Feature"},
            color=corr.values, color_continuous_scale="RdBu_r",
        )
        fig_corr.update_layout(coloraxis_showscale=False, height=600)
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------------------------
# Tab 3: Risk Predictor
# ---------------------------------------------------------------------------
with tab_predict:
    st.header("Diabetes Risk Predictor")
    st.warning(
        "⚠️ **This tool is for educational purposes only and is NOT a medical diagnosis.** "
        "It estimates statistical risk based on patterns in BRFSS 2015 survey data using a "
        "machine learning model with modest accuracy (Balanced Accuracy ≈ 0.52, Macro F1 ≈ 0.42). "
        "Always consult a healthcare professional for medical advice."
    )

    st.subheader("Tell us about yourself")

    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Demographics**")
            age_label = st.selectbox("Age Group", options=list(AGE_MAP.values()), index=8)
            sex_label = st.selectbox("Sex", options=["Female", "Male"])
            education_label = st.selectbox("Education", options=list(EDUCATION_MAP.values()), index=4)
            income_label = st.selectbox("Income Level", options=list(INCOME_MAP.values()), index=5)
            bmi = st.slider("BMI", 12.0, 60.0, 27.0, 0.1)

        with c2:
            st.markdown("**Health Conditions**")
            highbp = st.checkbox("High Blood Pressure")
            highchol = st.checkbox("High Cholesterol")
            cholcheck = st.checkbox("Cholesterol check in past 5 years", value=True)
            stroke = st.checkbox("History of Stroke")
            heartdisease = st.checkbox("Heart Disease / Heart Attack")
            diffwalk = st.checkbox("Difficulty Walking / Climbing Stairs")
            genhlth_label = st.selectbox("General Health", options=list(GENHLTH_MAP.values()), index=2)

        with c3:
            st.markdown("**Lifestyle & Access**")
            physactivity = st.checkbox("Physical Activity (past 30 days)", value=True)
            fruits = st.checkbox("Eats Fruit ≥ 1/day", value=True)
            veggies = st.checkbox("Eats Vegetables ≥ 1/day", value=True)
            smoker = st.checkbox("Smoker (100+ cigarettes lifetime)")
            hvyalcohol = st.checkbox("Heavy Alcohol Consumption")
            anyhealthcare = st.checkbox("Has Healthcare Coverage", value=True)
            nodoccost = st.checkbox("Skipped doctor visit due to cost")
            menthlth = st.slider("Poor mental health days (past 30)", 0, 30, 0)
            physhlth = st.slider("Poor physical health days (past 30)", 0, 30, 0)

        submitted = st.form_submit_button("Predict Risk", type="primary")

    if submitted:
        age = [k for k, v in AGE_MAP.items() if v == age_label][0]
        sex = [k for k, v in SEX_MAP.items() if v == sex_label][0]
        education = [k for k, v in EDUCATION_MAP.items() if v == education_label][0]
        income = [k for k, v in INCOME_MAP.items() if v == income_label][0]
        genhlth = [k for k, v in GENHLTH_MAP.items() if v == genhlth_label][0]

        raw = {
            "HighBP": int(highbp), "HighChol": int(highchol), "CholCheck": int(cholcheck), "BMI": bmi,
            "Smoker": int(smoker), "Stroke": int(stroke), "HeartDiseaseorAttack": int(heartdisease),
            "PhysActivity": int(physactivity), "Fruits": int(fruits), "Veggies": int(veggies),
            "HvyAlcoholConsump": int(hvyalcohol), "AnyHealthcare": int(anyhealthcare),
            "NoDocbcCost": int(nodoccost), "GenHlth": genhlth, "MentHlth": menthlth, "PhysHlth": physhlth,
            "DiffWalk": int(diffwalk), "Sex": sex, "Age": age, "Education": education, "Income": income,
        }

        X_input = feat.engineer_single_row(raw)
        proba = model.predict_proba(X_input)[0]
        pred_class = int(np.argmax(proba))

        st.divider()
        st.subheader("Result")

        res_col, eng_col = st.columns([2, 1])

        with res_col:
            risk_msg = {
                0: ("✅", "Lower predicted risk", "#2ecc71"),
                1: ("⚠️", "Elevated predicted risk — Prediabetes pattern", "#f1c40f"),
                2: ("\U0001f6a8", "High predicted risk — Diabetes pattern", "#e74c3c"),
            }
            icon, msg, color = risk_msg[pred_class]
            st.markdown(
                f"""
                <div class="risk-card" style="background-color:{color}26; border-left-color:{color};">
                    <h3 style="color:{color};">{icon} {msg}</h3>
                </div>
                """,
                unsafe_allow_html=True,
            )

            prob_df = pd.DataFrame({
                "Class": CLASS_NAMES,
                "Probability": proba,
            })
            fig_proba = px.bar(
                prob_df, x="Probability", y="Class", orientation="h",
                color="Class", color_discrete_map=COLOR_MAP, range_x=[0, 1],
                title="Predicted Class Probabilities",
                text=[f"{p:.1%}" for p in proba],
            )
            fig_proba.update_layout(showlegend=False)
            st.plotly_chart(fig_proba, use_container_width=True)

        with eng_col:
            st.markdown("**Derived features used by the model**")
            st.metric("BMI Category", BMI_CAT_MAP[int(X_input["BMICategory"].iloc[0])])
            st.metric("Lifestyle Score (0-5, higher = healthier)", int(X_input["LifestyleScore"].iloc[0]))
            st.metric("Comorbidity Score (0-4)", int(X_input["ComorbidityScore"].iloc[0]))

# ---------------------------------------------------------------------------
# Tab 4: Model Insights
# ---------------------------------------------------------------------------
with tab_model:
    st.header("Model Insights")
    st.caption("Final model: tuned LightGBM, retrained on train+validation, evaluated on the held-out test set.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{tm['Accuracy']:.2%}")
    c2.metric("Balanced Accuracy", f"{tm['Balanced Accuracy']:.2%}")
    c3.metric("Macro F1", f"{tm['Macro F1']:.3f}")
    c4.metric("Macro ROC-AUC", f"{tm['Macro ROC-AUC']:.3f}")

    st.subheader("Model Comparison")
    st.caption("All models evaluated on the validation set, except rows explicitly marked TEST SET.")
    st.dataframe(comparison, use_container_width=True)

    col_imp, col_cm = st.columns(2)

    with col_imp:
        st.subheader("SHAP Feature Importance")
        st.caption("Mean |SHAP value| across all 3 classes, computed on a 2,000-row test sample.")
        fig_shap = px.bar(
            shap_importance.iloc[:12][::-1],
            orientation="h",
            labels={"value": "Mean |SHAP value|", "index": "Feature"},
            color=shap_importance.iloc[:12][::-1].values,
            color_continuous_scale="Viridis",
        )
        fig_shap.update_layout(coloraxis_showscale=False, height=450)
        st.plotly_chart(fig_shap, use_container_width=True)

    with col_cm:
        st.subheader("Confusion Matrix (Test Set)")
        cm, report = get_test_evaluation(model, metadata["model_file"])
        fig_cm = px.imshow(
            cm, text_auto=".1%", x=CLASS_NAMES, y=CLASS_NAMES,
            color_continuous_scale="Blues", aspect="auto",
            labels=dict(x="Predicted", y="Actual", color="Proportion"),
        )
        fig_cm.update_layout(height=450, coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.subheader("Classification Report (Test Set)")
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

    with st.expander("Best Hyperparameters (RandomizedSearchCV)"):
        st.json(metadata["best_params"])
