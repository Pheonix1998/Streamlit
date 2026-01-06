import streamlit as st
import pandas as pd
import plotly.express as px
import pyodbc

# =====================================
# CSS FOR SIDEBAR & KPI CARDS
# =====================================
st.markdown("""
<style>
section[data-testid="stSidebar"] {background: linear-gradient(180deg, #0E1117 0%, #0B0F14 100%); border-right: 1px solid #1f2937;}
section[data-testid="stSidebar"] .stMultiSelect, section[data-testid="stSidebar"] .stTextInput {background-color: #111827; border-radius: 12px; padding: 6px; transition: all 0.25s ease;}
section[data-testid="stSidebar"] .stMultiSelect:hover, section[data-testid="stSidebar"] .stTextInput:hover {box-shadow: 0 0 14px rgba(0, 230, 118, 0.45); transform: translateY(-1px);}
span[data-baseweb="tag"] {background-color: #00E676 !important; color: #0E1117 !important; font-weight: 600; border-radius: 6px;}
[data-testid="stMetric"] {background: linear-gradient(145deg, #111827, #0B0F14); padding: 18px; border-radius: 14px; border: 1px solid #1f2937; box-shadow: 0 4px 18px rgba(0,0,0,0.35);}
[data-testid="stDataFrame"] {border-radius: 12px; border: 1px solid #1f2937;}
</style>
""", unsafe_allow_html=True)

# =====================================
# PAGE CONFIG
# =====================================
st.set_page_config(page_title="OLA Ride Analytics", layout="wide")
st.title("🚖 OLA Ride Analytics Dashboard")

# =====================================
# DATABASE CONNECTION
# =====================================
def get_connection():
    return pyodbc.connect(
        "DRIVER={ODBC Driver 17 for SQL Server};"
        "SERVER=TATHAGATA\\SQLEXPRESS;"
        "DATABASE=MyDatabase;"
        "Trusted_Connection=yes;"
    )

# =====================================
# LOAD DATA
# =====================================
@st.cache_data(show_spinner="Fetching data from SQL Server...")
def load_data():
    conn = get_connection()
    query = "SELECT * FROM [dbo].[OLA Cleaned]"
    df = pd.read_sql(query, conn)
    conn.close()
    df.columns = df.columns.str.strip().str.replace("\xa0", " ")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df

df = load_data()

# =====================================
# SIDEBAR FILTERS
# =====================================
st.sidebar.header("Filters")
status_filter = st.sidebar.multiselect("Booking Status", df["Booking_Status"].unique(), df["Booking_Status"].unique())
vehicle_filter = st.sidebar.multiselect("Vehicle Type", df["Vehicle_Type"].unique(), df["Vehicle_Type"].unique())
payment_filter = st.sidebar.multiselect("Payment Method", df["Payment_Method"].unique(), df["Payment_Method"].unique())

filtered_df = df[
    df["Booking_Status"].isin(status_filter) &
    df["Vehicle_Type"].isin(vehicle_filter) &
    df["Payment_Method"].isin(payment_filter)
]

# =====================================
# DASHBOARD TABS
# =====================================
tabs = st.tabs(["Overall", "Vehicle Type", "Revenue", "Cancellation", "Ratings"])

# =====================================
# TAB 1: OVERALL DASHBOARD (KPI + Ride Volume + Booking Status)
# =====================================
with tabs[0]:
    st.subheader("📈 Key Metrics")
    total_rides = filtered_df.shape[0]
    completed_rides = ((filtered_df["Booking_Status"]=="Success") & (filtered_df["Incomplete_Rides"]=="No")).sum()
    incomplete_rides = (filtered_df["Incomplete_Rides"]=="Yes").sum()
    cancelled_rides = filtered_df["Booking_Status"].str.contains("Cancelled", case=False, na=False).sum()
    cancellation_rate = round(cancelled_rides/total_rides*100,2) if total_rides>0 else 0
    avg_customer_rating = round(filtered_df.loc[filtered_df["Booking_Status"]=="Success","Customer_Rating"].mean(),2)
    total_revenue = filtered_df.loc[filtered_df["Booking_Status"]=="Success","Booking_Value"].sum()

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Rides", total_rides)
    c2.metric("Completed Rides", completed_rides)
    c3.metric("Incomplete Rides", incomplete_rides)
    c4.metric("Cancelled Rides", cancelled_rides)
    c5.metric("Cancellation %", cancellation_rate)

    st.divider()
    st.subheader("Monthly Ride Volume")
    ride_volume = filtered_df.groupby(pd.Grouper(key="Date", freq="M")).size().reset_index(name="Total_Rides")
    fig1 = px.line(ride_volume, x="Date", y="Total_Rides", markers=True)
    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Booking Status Breakdown")
    status_count = filtered_df["Booking_Status"].value_counts().reset_index()
    status_count.columns = ["Booking_Status","Count"]
    fig2 = px.bar(status_count, x="Booking_Status", y="Count", color="Booking_Status")
    st.plotly_chart(fig2, use_container_width=True)

# =====================================
# TAB 2: VEHICLE TYPE
# =====================================
with tabs[1]:
    st.subheader("Top 5 Vehicle Types by Ride Distance")
    vehicle_dist = filtered_df.groupby("Vehicle_Type")["Ride_Distance"].sum().nlargest(5).reset_index()
    fig3 = px.bar(vehicle_dist, x="Vehicle_Type", y="Ride_Distance", color="Ride_Distance")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Average Customer Rating by Vehicle Type")
    vehicle_rating = filtered_df.groupby("Vehicle_Type")["Customer_Rating"].mean().reset_index()
    fig4 = px.bar(vehicle_rating, x="Vehicle_Type", y="Customer_Rating", color="Customer_Rating")
    st.plotly_chart(fig4, use_container_width=True)

# =====================================
# TAB 3: REVENUE
# =====================================
with tabs[2]:
    st.subheader("Revenue by Payment Method")
    revenue = filtered_df[filtered_df["Booking_Status"]=="Success"].groupby("Payment_Method")["Booking_Value"].sum().reset_index()
    fig5 = px.pie(revenue, names="Payment_Method", values="Booking_Value")
    st.plotly_chart(fig5, use_container_width=True)

    st.subheader("Top 5 Customers by Booking Value")
    top_customers = filtered_df.groupby("Customer_ID")["Booking_Value"].sum().nlargest(5).reset_index()
    fig6 = px.bar(top_customers, x="Customer_ID", y="Booking_Value", color="Booking_Value")
    st.plotly_chart(fig6, use_container_width=True)

# =====================================
# TAB 4: CANCELLATION
# =====================================
with tabs[3]:
    st.subheader("Cancellation Reasons")
    cancel_reason = filtered_df[filtered_df["Incomplete_Rides"]=="Yes"]["Incomplete_Rides_Reason"].value_counts().reset_index()
    cancel_reason.columns = ["Reason","Count"]
    fig7 = px.bar(cancel_reason, x="Reason", y="Count", color="Count")
    st.plotly_chart(fig7, use_container_width=True)

# =====================================
# TAB 5: RATINGS
# =====================================
with tabs[4]:
    st.subheader("Driver Ratings Distribution")
    fig8 = px.histogram(filtered_df, x="Driver_Ratings", nbins=10)
    st.plotly_chart(fig8, use_container_width=True)

    st.subheader("Customer vs Driver Rating")
    fig9 = px.scatter(filtered_df, x="Customer_Rating", y="Driver_Ratings", color="Booking_Status")
    st.plotly_chart(fig9, use_container_width=True)

# =====================================
# EXPORT DATA
# =====================================
st.sidebar.divider()
st.sidebar.subheader("Export Filtered Data")
st.sidebar.download_button(
    "Download CSV",
    filtered_df.to_csv(index=False),
    "OLA_Filtered_Data.csv",
    "text/csv"
)