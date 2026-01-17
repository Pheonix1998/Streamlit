# ============================================================
# INSURANCE COST ANALYTICS & PREDICTION — STREAMLIT APP
# ============================================================

# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from sqlalchemy import create_engine

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Insurance Cost Analytics & Prediction",
    layout="wide"
)

# ===============================
# DATA LOAD (SQL + CLOUD CSV FALLBACK)
# ===============================
@st.cache_data(show_spinner=False)
def load_data():
    try:
        # -------- LOCAL SQL SERVER (WORKS ONLY ON YOUR MACHINE) --------
        server = r"TATHAGATA\SQLEXPRESS"
        database = "MyDatabase"

        connection_string = (
            "mssql+pyodbc://@{server}/{database}"
            "?driver=ODBC+Driver+17+for+SQL+Server"
            "&trusted_connection=yes"
        ).format(server=server, database=database)

        engine = create_engine(connection_string)

        query = """
            SELECT
                policy_no,
                age,
                sex,
                bmi,
                children,
                smoker,
                region,
                charges_in_inr
            FROM dbo.[POLICY DETAILS]
        """

        df = pd.read_sql(query, engine)

    except Exception:
        # -------- STREAMLIT CLOUD FALLBACK --------
        df = pd.read_csv("policy_details.csv")

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return df


df_raw = load_data()

# ===============================
# FEATURE ENGINEERING
# ===============================
df_raw["bmi_category"] = pd.cut(
    df_raw["bmi"], [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)

df_raw["age_group"] = pd.cut(
    df_raw["age"], [0, 25, 40, 55, 100],
    labels=["<25", "25–40", "40–55", "55+"]
)

# ===============================
# SIDEBAR FILTERS
# ===============================
st.sidebar.header("Policy Filters")

region_filter = st.sidebar.multiselect(
    "Region",
    sorted(df_raw["region"].dropna().unique())
)

smoker_filter = st.sidebar.multiselect(
    "Smoker Status",
    sorted(df_raw["smoker"].dropna().unique())
)

df = df_raw.copy()

if region_filter:
    df = df[df["region"].isin(region_filter)]

if smoker_filter:
    df = df[df["smoker"].isin(smoker_filter)]

# ===============================
# EXECUTIVE KPI LAYER
# ===============================
st.title("Insurance Cost Intelligence Dashboard")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Policies", len(df))
k2.metric("Average Policy Cost", f"₹ {df['charges_in_inr'].mean():,.0f}")
k3.metric("Median Policy Cost", f"₹ {df['charges_in_inr'].median():,.0f}")

smoker_ratio = (
    df[df["smoker"] == "yes"]["charges_in_inr"].mean() /
    df[df["smoker"] == "no"]["charges_in_inr"].mean()
)
k4.metric("Smoker Premium Multiplier", f"{smoker_ratio:.2f}x")

st.markdown("""
**Executive Interpretation:**  
These KPIs summarize portfolio scale, pricing behavior, and the financial
impact of high-risk policyholder segments.
""")

# ===============================
# COST DRIVERS ANALYSIS
# ===============================
st.subheader("Primary Cost Drivers")

col1, col2 = st.columns(2)

fig_age = px.line(
    df.groupby("age_group")["charges_in_inr"].mean().reset_index(),
    x="age_group",
    y="charges_in_inr",
    markers=True,
    title="Insurance Cost Trend by Age Group"
)
col1.plotly_chart(fig_age, use_container_width=True)

fig_bmi = px.bar(
    df.groupby("bmi_category")["charges_in_inr"].mean().reset_index(),
    x="bmi_category",
    y="charges_in_inr",
    text_auto=".2s",
    title="Average Insurance Cost by BMI Category"
)
col2.plotly_chart(fig_bmi, use_container_width=True)

st.markdown("""
**Business Interpretation:**  
Age and BMI act as cumulative risk multipliers and should be core dimensions
in underwriting and pricing strategy.
""")

# ===============================
# GEOGRAPHIC PRICING IMPACT
# ===============================
st.subheader("Geographic Pricing Impact")

fig_region = px.bar(
    df.groupby("region")["charges_in_inr"]
      .mean()
      .reset_index()
      .sort_values("charges_in_inr", ascending=False),
    x="region",
    y="charges_in_inr",
    text_auto=".2s",
    title="Average Insurance Cost by Region"
)

st.plotly_chart(fig_region, use_container_width=True)

# ===============================
# LOAD TRAINED MODEL
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("model/insurance_model.pkl")

model = load_model()

# ===============================
# PREDICTION ENGINE
# ===============================
st.header("Predict Insurance Premium")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 65, 30)
    sex = st.selectbox("Gender", ["male", "female"])

with c2:
    bmi = st.number_input("BMI", 15.0, 45.0, 25.0)
    smoker = st.selectbox("Smoker", ["yes", "no"])

with c3:
    children = st.selectbox("Number of Dependents", [0, 1, 2, 3, 4])
    region = st.selectbox(
        "Region",
        ["northeast", "northwest", "southeast", "southwest"]
    )

bmi_category = pd.cut(
    [bmi], [0, 18.5, 25, 30, 100],
    labels=["Underweight", "Normal", "Overweight", "Obese"]
)[0]

age_group = pd.cut(
    [age], [0, 25, 40, 55, 100],
    labels=["<25", "25–40", "40–55", "55+"]
)[0]

if st.button("Predict Premium"):
    input_df = pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex": sex,
        "smoker": smoker,
        "region": region,
        "bmi_category": bmi_category,
        "age_group": age_group
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Estimated Insurance Premium: ₹ {prediction:,.0f}")

st.caption(
    "SQL Server–backed analytics with Random Forest–based pricing intelligence"
)