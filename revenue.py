import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Netflix Revenue Forecast", layout="wide")

# --- TITLE ---
st.title("📈 Netflix Revenue Forecasting App")
st.markdown("Forecast Netflix's future revenue using historical data with **Prophet**.")

# --- DATA MODE SELECTION ---
st.sidebar.header("📂 Data Source")
use_uploaded = st.sidebar.toggle("Upload custom Excel file", value=False)

if use_uploaded:
    uploaded_file = st.file_uploader("📁 Upload Excel file", type=["xlsx"])
else:
    st.sidebar.info("Using default file: **Netflix Revenue and Usage Statistics.xlsx**")
    uploaded_file = "Netflix Revenue and Usage Statistics.xlsx"

# --- LOAD FILE ---
if uploaded_file:
    try:
        # Handle both uploaded and local file
        xls = pd.ExcelFile(uploaded_file) if use_uploaded else pd.ExcelFile(open(uploaded_file, "rb"))

        # Load sheets
        revenue_df = xls.parse('Netflix annual revenue 2011 to ')
        subs_df = xls.parse('Netflix annual subscribers 2011')
        content_df = xls.parse('Netflix annual content spend ($')
        income_df = xls.parse('Netflix annual net incomeloss (')

        # Rename columns
        revenue_df.columns = ['Year', 'Revenue ($)']
        subs_df.columns = ['Year', 'Subscribers']
        content_df.columns = ['Year', 'Content Spend ($)']
        income_df.columns = ['Year', 'Net Income ($)']

        # Clean currency columns
        def clean_currency(col):
            return col.replace('[\$,]', '', regex=True).astype(float)

        revenue_df['Revenue ($)'] = clean_currency(revenue_df['Revenue ($)'])
        subs_df['Subscribers'] = clean_currency(subs_df['Subscribers'])
        content_df['Content Spend ($)'] = clean_currency(content_df['Content Spend ($)'])
        income_df['Net Income ($)'] = clean_currency(income_df['Net Income ($)'])

        # Merge all
        df = revenue_df.merge(subs_df, on='Year') \
                       .merge(content_df, on='Year') \
                       .merge(income_df, on='Year')

        # --- Show data ---
        st.subheader("📊 Cleaned Netflix Financial Data")
        st.dataframe(df)

        # --- Revenue line chart ---
        st.subheader("📈 Revenue Over Time")
        st.line_chart(df.set_index('Year')['Revenue ($)'])

        # --- Prophet Preparation ---
        prophet_df = df[['Year', 'Revenue ($)']].copy()
        prophet_df['ds'] = pd.to_datetime(prophet_df['Year'], format='%Y')
        prophet_df['y'] = prophet_df['Revenue ($)']
        prophet_df = prophet_df[['ds', 'y']]

        # --- Forecasting ---
        forecast_years = st.slider("🔮 Forecast years ahead", 1, 10, 5)
        periods = forecast_years * 365

        model = Prophet(yearly_seasonality=True)
        model.fit(prophet_df)

        future = model.make_future_dataframe(periods=periods, freq='D')
        forecast = model.predict(future)

        # --- Forecast Plot ---
        st.subheader("📉 Revenue Forecast")
        fig1 = model.plot(forecast)
        st.pyplot(fig1)

        # --- Prediction Summary Under Chart ---
        st.subheader("📌 Forecast Insights")
        forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Convert ds to year only for easier reading
        forecast_out['Year'] = forecast_out['ds'].dt.year

        # Filter to only future years (after max year in original data)
        latest_year = df['Year'].max()
        future_forecast = forecast_out[forecast_out['Year'] > latest_year]

        # Group by year to get annual predicted totals
        annual_pred = future_forecast.groupby('Year')['yhat'].mean().round()

        # Display next year prediction
        next_year = latest_year + 1
        if next_year in annual_pred:
            st.markdown(f"**📅 Predicted Revenue for {next_year}:** ${int(annual_pred[next_year]):,}")

        # Highest & lowest forecasted years
        max_year = annual_pred.idxmax()
        min_year = annual_pred.idxmin()

        st.markdown(f"**🔺 Highest Forecasted Revenue:** ${int(annual_pred[max_year]):,} in {max_year}")
        st.markdown(f"**🔻 Lowest Forecasted Revenue:** ${int(annual_pred[min_year]):,} in {min_year}")


        # --- Components ---
        with st.expander("📊 Show Forecast Components"):
            fig2 = model.plot_components(forecast)
            st.pyplot(fig2)

        # --- CSV Download ---
        forecast_out = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        csv = forecast_out.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Forecast CSV", csv, "netflix_forecast.csv", "text/csv")

    except Exception as e:
        st.error(f"Something went wrong. Check your sheet names and file format.\n\nError: {e}")

else:
    st.info("📤 Please upload a file or use the default to begin.")
