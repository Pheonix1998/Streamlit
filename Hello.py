import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Business Performance Dashboard",
    layout="wide"
)

st.title("📊 Business Performance Dashboard")
st.markdown("High-level operational insights with interactive drill-downs")

# -------------------------------
# Load / Create Sample Data
# -------------------------------
data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "Revenue": [120000, 135000, 128000, 150000, 165000, 170000],
    "Orders": [320, 350, 340, 390, 420, 450],
    "Customer_Rating": [4.1, 4.2, 4.0, 4.3, 4.4, 4.5]
}

df = pd.DataFrame(data)

# -------------------------------
# KPI Section
# -------------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Revenue", f"₹ {df['Revenue'].sum():,}")
col2.metric("Total Orders", df["Orders"].sum())
col3.metric("Avg Customer Rating", round(df["Customer_Rating"].mean(), 2))

st.divider()

# -------------------------------
# Filters
# -------------------------------
selected_months = st.multiselect(
    "Select Month(s)",
    options=df["Month"].unique(),
    default=df["Month"].unique()
)

filtered_df = df[df["Month"].isin(selected_months)]

# -------------------------------
# Visualizations
# -------------------------------
col4, col5 = st.columns(2)

with col4:
    st.subheader("Revenue Trend")
    fig, ax = plt.subplots()
    ax.plot(filtered_df["Month"], filtered_df["Revenue"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue")
    st.pyplot(fig)

with col5:
    st.subheader("Orders Trend")
    fig, ax = plt.subplots()
    ax.bar(filtered_df["Month"], filtered_df["Orders"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Orders")
    st.pyplot(fig)

st.divider()

# -------------------------------
# Data Table
# -------------------------------
st.subheader("Underlying Data")
st.dataframe(filtered_df, use_container_width=True)
