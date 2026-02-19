import streamlit as st
import pandas as pd
import plotly.express as px

# Set page configuration for the dashboard
st.set_page_config(page_title="Digital Music Store Dashboard", page_icon="🎵", layout="wide")

# Function to load and clean data
@st.cache_data
def load_data():
    df = pd.read_excel("MUSIC JOINED DATA.xlsx")
    
    # Convert dates to datetime objects
    df['formatted_invoice_date'] = pd.to_datetime(df['formatted_invoice_date'])
    df['YearMonth'] = df['formatted_invoice_date'].dt.to_period('M').astype(str)
    df['Year'] = df['formatted_invoice_date'].dt.year
    
    # Add Day of Week for behavioral analysis
    df['DayOfWeek'] = df['formatted_invoice_date'].dt.day_name()

    # Drop duplicate invoice lines to calculate actual revenue and track sales accurately
    # (Since a track might appear in multiple playlists, it duplicates invoice lines)
    sales_df = df.drop_duplicates(subset=['invoice_line_id'])
    
    return df, sales_df

df, sales_df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("Dashboard Filters ⚙️")

# Year Filter
years = sorted(sales_df['Year'].unique().tolist())
selected_years = st.sidebar.multiselect("Select Year(s)", years, default=years)

# Country Filter
countries = sorted(sales_df['billing_country'].unique().tolist())
selected_countries = st.sidebar.multiselect("Select Billing Country", countries, default=countries)

# Apply filters
filtered_sales = sales_df[
    (sales_df['Year'].isin(selected_years)) & 
    (sales_df['billing_country'].isin(selected_countries))
]

# --- MAIN DASHBOARD ---
st.title("🎵 Digital Music Store Executive Dashboard")
st.markdown("Analyze sales trends, customer distribution, and top-performing tracks & artists.")

# --- KPIs ---
st.subheader("Key Performance Indicators (KPIs)")

col1, col2, col3, col4 = st.columns(4)

total_revenue = filtered_sales['unit_price'].sum()
total_tracks_sold = filtered_sales['quantity'].sum()
unique_customers = filtered_sales['customer_id'].nunique()
top_artist = filtered_sales.groupby('artist_name')['unit_price'].sum().idxmax() if not filtered_sales.empty else "N/A"

col1.metric("Total Revenue", f"${total_revenue:,.2f}")
col2.metric("Tracks Sold", f"{total_tracks_sold:,}")
col3.metric("Unique Customers", f"{unique_customers:,}")
col4.metric("Top Grossing Artist", top_artist)

st.divider()

# --- ROW 1 CHARTS ---
col_left, col_right = st.columns(2)

# 1. Monthly Revenue Trend
with col_left:
    st.subheader("📈 Monthly Revenue Trend")
    monthly_sales = filtered_sales.groupby('YearMonth')['unit_price'].sum().reset_index()
    fig_trend = px.line(
        monthly_sales, 
        x='YearMonth', 
        y='unit_price', 
        markers=True,
        labels={'unit_price': 'Revenue ($)', 'YearMonth': 'Month'},
        template="plotly_white"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# 2. Revenue by Country
with col_right:
    st.subheader("🌎 Revenue by Country (Top 10)")
    country_sales = filtered_sales.groupby('billing_country')['unit_price'].sum().nlargest(10).reset_index()
    fig_country = px.bar(
        country_sales, 
        x='unit_price', 
        y='billing_country', 
        orientation='h',
        labels={'unit_price': 'Revenue ($)', 'billing_country': 'Country'},
        template="plotly_white",
        color='unit_price',
        color_continuous_scale="Blues"
    )
    fig_country.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_country, use_container_width=True)

# --- ROW 2 CHARTS ---
col_bottom_left, col_bottom_right = st.columns(2)

# 3. Top Selling Genres
with col_bottom_left:
    st.subheader("🎸 Top Selling Genres")
    genre_sales = filtered_sales.groupby('genre_name')['unit_price'].sum().nlargest(10).reset_index()
    fig_genre = px.pie(
        genre_sales, 
        names='genre_name', 
        values='unit_price',
        hole=0.4,
        template="plotly_white"
    )
    fig_genre.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_genre, use_container_width=True)

# 4. Sales by Support Rep
with col_bottom_right:
    st.subheader("🧑‍💼 Revenue by Support Representative")
    rep_sales = filtered_sales.groupby('employee_name')['unit_price'].sum().reset_index()
    fig_rep = px.bar(
        rep_sales, 
        x='employee_name', 
        y='unit_price',
        text_auto='.2s',
        labels={'unit_price': 'Revenue ($)', 'employee_name': 'Support Rep'},
        template="plotly_white",
        color='employee_name'
    )
    st.plotly_chart(fig_rep, use_container_width=True)


# --- ROW 3 CHARTS (NEW) ---
st.divider()
col_new1, col_new2 = st.columns(2)

# 5. Top 10 Customers
with col_new1:
    st.subheader("🏆 Top 10 Customers by Revenue")
    customer_sales = filtered_sales.groupby('customer_name')['unit_price'].sum().nlargest(10).reset_index()
    fig_cust = px.bar(
        customer_sales, 
        x='unit_price', 
        y='customer_name', 
        orientation='h',
        labels={'unit_price': 'Total Spent ($)', 'customer_name': 'Customer'},
        template="plotly_white",
        color='unit_price',
        color_continuous_scale="Teal"
    )
    fig_cust.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_cust, use_container_width=True)

# 6. Sales by Media Type
with col_new2:
    st.subheader("💿 Revenue by Media Format")
    media_sales = filtered_sales.groupby('media_type_name')['unit_price'].sum().reset_index()
    fig_media = px.pie(
        media_sales, 
        names='media_type_name', 
        values='unit_price',
        hole=0.4,
        template="plotly_white",
        color_discrete_sequence=px.colors.sequential.Aggrnyl
    )
    fig_media.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig_media, use_container_width=True)


# --- ROW 4 CHARTS (NEW) ---
col_new3, col_new4 = st.columns(2)

# 7. Top 15 Artists Performance
with col_new3:
    st.subheader("🎤 Top 15 Artists Performance")
    
    # Get top 15 artists by revenue
    artist_sales = filtered_sales.groupby('artist_name')['unit_price'].sum().nlargest(15).reset_index()
    
    # Create a horizontal bar chart
    fig_artist = px.bar(
        artist_sales, 
        x='unit_price', 
        y='artist_name', 
        orientation='h',
        labels={'unit_price': 'Revenue ($)', 'artist_name': 'Artist'},
        template="plotly_white",
        color='unit_price',
        color_continuous_scale="Purp" # Kept the purple theme from the treemap
    )
    
    # Sort so the highest revenue is at the top
    fig_artist.update_layout(yaxis={'categoryorder':'total ascending'})
    
    st.plotly_chart(fig_artist, use_container_width=True)

# 8. Purchases by Day of Week
with col_new4:
    st.subheader("📅 Purchasing Patterns (Day of Week)")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_sales = filtered_sales.groupby('DayOfWeek')['unit_price'].sum().reindex(days_order).reset_index()
    
    fig_dow = px.bar(
        dow_sales, 
        x='DayOfWeek', 
        y='unit_price',
        text_auto='.2s',
        labels={'unit_price': 'Revenue ($)', 'DayOfWeek': 'Day'},
        template="plotly_white",
        color='unit_price',
        color_continuous_scale="sunset"
    )
    st.plotly_chart(fig_dow, use_container_width=True)

# --- DATA TABLE ---
st.divider()
st.subheader("📄 Raw Sales Data (Filtered)")

st.dataframe(filtered_sales[['formatted_invoice_date', 'DayOfWeek', 'customer_name', 'billing_country', 'track_name', 'artist_name', 'genre_name', 'media_type_name', 'unit_price']].sort_values(by='formatted_invoice_date', ascending=False).head(100), use_container_width=True)
