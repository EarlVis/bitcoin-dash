# Import libraries
import yfinance as yf
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from urllib.request import urlopen
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from sklearn.linear_model import LinearRegression
import os
import math
import datetime as dt
import seaborn as sns
sns.set()
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import scipy.stats
from scipy import stats
import pylab
import warnings
warnings.filterwarnings('ignore')

# Titles and Subtitles
st.title("Bitcoin Masters")
st.sidebar.title("Navigation")

# Load and clean data
# Define the ticker symbol for Bitcoin
ticker_symbol = 'BTC-USD'

# Fetch Bitcoin data
BTC_data = yf.Ticker(ticker_symbol).history(period='10y').reset_index()
#BTC_data = pd.read_csv("BTC-USD.csv")
BTC_data['Date'] = pd.to_datetime(BTC_data['Date'])
BTC_data['Date'] = BTC_data['Date'].dt.strftime('%Y-%m-%d')
start_date = '2015-01-01'
end_date = '2026-01-01'
BTC_data = BTC_data[(BTC_data['Date'] >= start_date) & (BTC_data['Date'] < end_date)]
BTC_data = BTC_data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
BTC_data.set_index('Date', inplace=True)

BTCHis = BTC_data

# Create a duplicate DataFrame btc
btc = BTC_data.copy()
btc.index = pd.to_datetime(BTC_data.index)
btc.sort_index(ascending=True, inplace=True)

# Create a sidebar navigation to switch between pages
page = st.sidebar.selectbox("Select a Page", ["Home", "Exploratory Data Analysis", "Time Series Analysis",   "Models"])

if page == "Home":
    st.header("Home Page")
    st.subheader("Welcome to the home of all things crypto.")
    
    # Bitcoin
    st.write("Bitcoin ($)")
    imageBTC = Image.open(urlopen('https://s2.coinmarketcap.com/static/img/coins/64x64/1.png'))
    # Display Image
    st.image(imageBTC)
    
    # Create a checkbox to toggle the visibility of additional rows
    show_more_rows = st.checkbox("Show More Rows")
    # Create an expander to display a limited number of rows by default
    with st.expander("Recent Bitcoin Data"):
        # Display a limited number of rows using st.dataframe
        num_rows_to_display = 5  # You can adjust this number as needed
        if not show_more_rows:
            BTC_display = BTC_data.tail(num_rows_to_display)
        else:
            BTC_display = BTC_data

        st.dataframe(BTC_display)

    # Display Line Chart
    st.header("Bitcoin Price Trend 2015 - 2025")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=BTCHis.index,
        y=BTCHis['Close'],
        mode='lines',
        name='Bitcoin Close Prices',
        line=dict(color='blue', width=2),  # Customize line color and width
        marker=dict(color='blue', size=5),  # Customize marker color and size
    ))

    # Customize the appearance of the chart
    fig.update_layout(
        title_text="Bitcoin Price Trend",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        yaxis_gridcolor='gray',
        plot_bgcolor='black',  # Set background color
    )
    st.plotly_chart(fig)
    
    st.markdown("### Bitcoin Price Trend Over the Years (2015 - 2025)")
    st.markdown("""
    ### Overview

    This interactive chart provides an overview of the Bitcoin price trend from 2015 to 2023. It visualizes the closing prices of Bitcoin over time.

    ### Insights

    - You can observe the fluctuations in Bitcoin's price over the years.
    - Analyze how market events and trends have impacted Bitcoin's price.

    #### Explore the charts to gain insights into Bitcoin's historical performance.

    """)

    
    
#Exploratory Data Analysis Page
elif page == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    # Create a dropdown menu for different analysis options
    analysis_option = st.selectbox("Select Analysis", ["Volume Analysis", "Yearly Trends", "Quarterly Trends", "Regression Analysis",
                                                       "Volatility Analysis"])

    if analysis_option == "Volume Analysis":
        st.subheader("Volume Analysis per Year")

        selected_year_volume = st.selectbox("Select Year", range(2015, 2024))

        # Filter data for the selected year
        year_data_volume = BTC_data[f'{selected_year_volume}-01-01':f'{selected_year_volume}-12-31']

        # Create a line chart for daily trading volume
        fig_volume = px.line(year_data_volume, y='Volume',
                             labels={'Volume': 'Trading Volume'},
                             title=f'Volume Analysis for the Year {selected_year_volume}')

        # Customize the appearance of the chart
        fig_volume.update_xaxes(showgrid=True, gridcolor='gray')
        fig_volume.update_yaxes(showgrid=True, gridcolor='gray')
        fig_volume.update_layout(plot_bgcolor='black', paper_bgcolor='black', font=dict(color='white'))

        # Show the chart for volume analysis
        st.plotly_chart(fig_volume)

        st.markdown("### Volume Analysis for the Selected Year")
        st.markdown("""
        ### Overview

        This line chart provides a visual analysis of the daily trading volume for Bitcoin in the selected year. It allows you to observe how trading volume has changed throughout the year.

        ### Insights

        - High trading volume periods may indicate increased market activity.
        - Low trading volume periods may suggest reduced market interest.
        """)

    
    elif analysis_option == "Yearly Trends":
        st.subheader("Yearly Trends for Close Prices")
        
        # Add a dropdown to select the year
        selected_year = st.selectbox("Select Year", range(2015, 2025))
        
        # Filter data for the selected year
        year_data = BTC_data[f'{selected_year}-01-01':f'{selected_year}-12-31']

        # Create a line chart for the Close prices
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=year_data.index,
            y=year_data['Close'],
            mode='lines',
            name=f'Bitcoin Close Prices ({selected_year})',
            line=dict(color='blue')
        ))

        # Customize the appearance of the chart
        fig.update_layout(
            title=f'Bitcoin Close Price Trend ({selected_year})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=True,
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            yaxis_gridcolor='lightgray',
        )
        # Show the chart
        st.plotly_chart(fig)   
        st.markdown(f"### Bitcoin Close Price Trend {selected_year}")
        st.markdown(f"This line chart displays the trend of Bitcoin's close prices for the year {selected_year}.")
        st.markdown("The blue line represents the close prices, and it shows how they fluctuated throughout the year.")           
        
        
        st.write("\n\n\n", "") 
        st.write("\n\n\n", "") 
        
        
        
        
        
        # Filter data for the selected year
        year_data = BTC_data[f'{selected_year}-01-01':f'{selected_year}-12-31']
        # Convert the index to a DateTimeIndex
        year_data.index = pd.to_datetime(year_data.index)
        # Calculate monthly average high, low, open, and close prices for the selected year
        monthly_averages = year_data.resample('M').mean()
        # Create a DataFrame for the monthly averages
        monthly_averages['Month'] = monthly_averages.index.strftime('%B %Y')

        
        # Create a bar graph for the monthly average High and Low prices with custom styling
        fig_high_low = go.Figure()
        # Add the High price as a bar
        fig_high_low.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['High'],
            name='High',
            marker_color='rgb(0, 102, 204)'  # Custom color for High price bars
        ))

        # Add the Low price as a bar
        fig_high_low.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Low'],
            name='Low',
            marker_color='rgb(255, 0, 0)'  # Custom color for Low price bars
        ))

        # Customize the appearance of the High and Low chart
        fig_high_low.update_xaxes(type='category')  # Set x-axis type to category
        fig_high_low.update_layout(
            barmode='group',  # Set bar mode to group for side-by-side bars
            title=f'Monthly Average High and Low Prices for {selected_year}',
            xaxis_title='Month',
            yaxis_title='Price ($)',
            legend_title='Price Type',
            legend=dict(x=0.85, y=1.0),  # Position of the legend
        )

        # Show the High and Low chart with a description
        st.plotly_chart(fig_high_low)
        st.markdown("### Monthly Average High and Low Prices")
        st.markdown(f"This bar chart displays the monthly average High and Low prices for the year {selected_year}.")
        st.markdown("The blue bars represent High prices, and the red bars represent Low prices.")

        
        st.write("\n\n\n", "") 
        st.write("\n\n\n", "") 
        # Create a bar graph for the monthly average Open and Close prices with custom styling
        fig_open_close = go.Figure()

        # Add the Open price as a bar
        fig_open_close.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Open'],
            name='Open',
            marker_color='rgb(0, 204, 0)'  # Custom color for Open price bars
        ))

        # Add the Close price as a bar
        fig_open_close.add_trace(go.Bar(
            x=monthly_averages['Month'],
            y=monthly_averages['Close'],
            name='Close',
            marker_color='rgb(255, 153, 0)'  # Custom color for Close price bars
        ))

        # Customize the appearance of the Open and Close chart
        fig_open_close.update_xaxes(type='category')  # Set x-axis type to category
        fig_open_close.update_layout(
            barmode='group',  # Set bar mode to group for side-by-side bars
            title=f'Monthly Average Open and Close Prices for {selected_year}',
            xaxis_title='Month',
            yaxis_title='Price ($)',
            legend_title='Price Type',
            legend=dict(x=0.85, y=1.0),  # Position of the legend
        )

        # Show the Open and Close chart with a description
        st.plotly_chart(fig_open_close)
        st.markdown("### Monthly Average Open and Close Prices")
        st.markdown(f"This bar chart displays the monthly average Open and Close prices for the year {selected_year}.")
        st.markdown("The green bars represent Open prices, and the orange bars represent Close prices.")

        
        
    elif analysis_option == "Quarterly Trends":
        st.subheader("Quarterly Trends for Close Prices")
        st.subheader("Candlestick Chart for Quarters")

        # Candlestick Chart for Quarters
        years = range(2015, 2025)

        selected_year = str(st.selectbox("Select a Year", years))

        quarters = ["First Quarter (Jan-Mar)", "Second Quarter (Apr-Jun)", "Third Quarter (Jul-Sep)", "Fourth Quarter (Oct-Dec)"]

        selected_quarter = st.selectbox("Select a Quarter", quarters)

        # Define the start and end months for each quarter
        quarter_start_months = [1, 4, 7, 10]
        quarter_end_months = [3, 6, 9, 12]

        quarter_index = quarters.index(selected_quarter)
        start_month = quarter_start_months[quarter_index]
        end_month = quarter_end_months[quarter_index]

        # Calculate the last day of the end month
        last_day = 31 if end_month == 12 else (30 if end_month in [4, 6, 9, 11] else 28)

        # Filter the data for the selected quarter
        start_date_period = f"{selected_year}-{start_month:02}-01"
        end_date_period = f"{selected_year}-{end_month:02}-{last_day:02}"
        filtered_data = BTC_data[(BTC_data.index >= start_date_period) & (BTC_data.index <= end_date_period)]

        # Create a candlestick chart with custom colors
        fig = go.Figure(data=[go.Candlestick(
            x=filtered_data.index,
            open=filtered_data['Open'],
            high=filtered_data['High'],
            low=filtered_data['Low'],
            close=filtered_data['Close'],
            increasing_fillcolor='green',
            decreasing_fillcolor='red'
        )])

        # Customize the appearance of the chart
        fig.update_layout(
            title=f'Bitcoin Price Candlestick Chart ({selected_quarter} {selected_year})',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            xaxis_rangeslider_visible=True,
            xaxis_showgrid=True,
            yaxis_showgrid=True,
            yaxis_gridcolor='lightgray',
            plot_bgcolor='black',  # Set background color
        )
        fig.update_xaxes(showgrid=True, gridcolor='gray')
        fig.update_yaxes(showgrid=True, gridcolor='gray')
        fig.update_traces(line=dict(width=0.5))  # Customize candlestick line width

        st.plotly_chart(fig)


        st.markdown("### Bitcoin Price Candlestick Chart for the Selected Quarter")
        st.markdown(f"""
        ### Overview

        This candlestick chart provides a visual representation of Bitcoin's price movement during the selected quarter of {selected_year}. It displays the open, high, low, and close prices for each day within the quarter.

        ### Insights

        - Candlesticks are color-coded for easy interpretation: green for price increases and red for price decreases.
        - Observe patterns such as doji, hammer, and shooting star to analyze price trends.

        Explore the chart to gain insights into Bitcoin's quarterly price fluctuations.

        """)

    elif analysis_option == "Regression Analysis":
        st.subheader("Regression Analysis")

        # Create a range of years from 2015 to 2023
        years = range(2015, 2024)

        # Select a year from the dropdown
        selected_year = str(st.selectbox("Select a Year", years))  # Convert to string

        # Convert the index values to strings
        btc['Date'] = btc.index.strftime('%Y-%m-%d')

        # Filter the data for the selected year
        filtered_data = btc[btc['Date'].str.startswith(selected_year)]

        # Create a DataFrame to store the results
        results = pd.DataFrame(columns=['Year', 'Coefficient', 'Intercept', 'R-squared'])

        # Perform linear regression analysis for the selected year if data is available
        if len(filtered_data) > 1:
            X = np.arange(len(filtered_data)).reshape(-1, 1)
            y = filtered_data['Close'].values.reshape(-1, 1)

            # Create and fit the linear regression model
            model = LinearRegression()
            model.fit(X, y)

            # Add the results to the DataFrame
            results = pd.concat([results, pd.DataFrame({
                'Year': [selected_year],
                'Coefficient': [model.coef_[0][0]],
                'Intercept': [model.intercept_[0]],
                'R-squared': [model.score(X, y)]
            })], ignore_index=True)

            # Display the regression results
            st.subheader(f"Regression Results for {selected_year}")
            st.table(results)

            # Plot the regression line
            st.markdown(f"### Regression Line Plot for {selected_year}")
            plt.figure(figsize=(10, 6))
            plt.scatter(X, y, label='Actual Prices', color='blue')
            plt.plot(X, model.predict(X), label='Regression Line', color='red')
            plt.xlabel('Days')
            plt.ylabel('Bitcoin Price ($)')
            plt.title(f'Regression Analysis for {selected_year}')
            plt.legend()
            plt.grid(True)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.tight_layout()

            # Customize the appearance of the plot
            st.pyplot(plt)


            st.markdown(f"""
            ### Overview

            This regression analysis provides insights into the linear relationship between time (days) and Bitcoin's closing prices for the selected year, {selected_year}.

            ### Insights

            - The regression results table displays the coefficient, intercept, and R-squared value, indicating how well the regression line fits the data.
            - The scatter plot shows the actual Bitcoin prices (in blue), and the red line represents the regression line.
            - Analyze how well the linear regression model captures the price trend.

            Explore the results and plot to gain insights into Bitcoin's price behavior in {selected_year}.

            """)

            
    elif analysis_option == "Volatility Analysis":
        st.subheader("Volatility Analysis")

        # Create a range of years from 2015 to 2025
        years = range(2015, 2025)
        # Select a year from the dropdown
        selected_year = st.selectbox("Select a Year", years)

        # Filter the data for the selected year
        year_data = btc[btc.index.year == selected_year]

        # Calculate daily returns
        year_data['Daily Returns'] = year_data['Close'].pct_change()

        # Calculate rolling standard deviation (volatility) with a window of your choice (e.g., 30 days)
        volatility_window = 30
        year_data['Volatility'] = year_data['Daily Returns'].rolling(window=volatility_window).std()

        # Create a line chart to visualize volatility for the selected year
        fig_volatility = px.line(year_data, x=year_data.index, y='Volatility',
                                 title=f'Volatility Analysis for {selected_year}')
        fig_volatility.update_xaxes(title='Date')
        fig_volatility.update_yaxes(title='Volatility')
        fig_volatility.update_traces(line=dict(color='blue'))
        fig_volatility.update_layout(
            title_font=dict(size=24, color='white', family='Arial'),
            xaxis_title_font=dict(size=16, color='white', family='Arial'),
            yaxis_title_font=dict(size=16, color='white', family='Arial'),
            paper_bgcolor='black',
            plot_bgcolor='black',
        )
        st.plotly_chart(fig_volatility)

        st.markdown("---")

        st.markdown(f"""
        ### Overview

        Volatility in financial markets refers to the degree of variation or dispersion in the returns of an asset over a specific period of time. It quantifies how much the price of an asset fluctuates.

        ### Key Concepts

        - **Standard Deviation**: Volatility is often calculated using the standard deviation of an asset's returns. It represents how much the returns of an asset deviate from their average or mean.

        - **Daily Returns**: To calculate volatility, daily returns are used. Daily returns are the percentage change in the price of an asset from one day to the next.

        ### Practical Use

        - **Risk Assessment**: Volatility is a key indicator of risk. Higher volatility implies greater uncertainty and risk.

        - **Investment Strategy**: Understanding an asset's volatility helps investors choose investments that align with their risk tolerance.

        - **Options Pricing**: In options trading, volatility affects the pricing of options contracts.

        In financial analysis, volatility is crucial for assessing and managing risk, constructing diversified portfolios, and making informed investment decisions.

        """)

    
        
    elif analysis_option == "Other Analysis":
          st.subheader("Other Analysis")
            # Add content for other analysis options here    

# You can add more elif blocks for other pages or analysis options    
    
#Time Series Analysis Page
elif page == "Time Series Analysis":
    st.title("Time Series Analysis")  
    
    # Create a dropdown menu for different analysis options
    TS_analysis_option = st.selectbox("Select TS Analysis", ["QQ Plot", "Lag Plot", "ACF Plot", "Seasonality", "Stationarity"])

    ##QQ Plot Tab
    if TS_analysis_option == "QQ Plot":
        st.subheader("QQ Plot Results")
        
        # Add a date selection widget to specify the start date
        start_date = st.date_input("Select Start Date", pd.to_datetime("2015-01-01"))

        # Convert the selected start date to a string
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Filter the DataFrame based on the selected start date
        filtered_data = BTC_data.loc[start_date_str:].copy()

        # Perform the QQ plot
        res = stats.probplot(filtered_data['Close'], plot=plt)
        plt.title("QQ Plot", size=24)

        # Display the QQ plot within the Streamlit app
        st.pyplot(plt)
        st.write("### Lag Plot Summary")
        st.markdown("""
        A **lag plot** is a valuable tool in time series analysis, used to visualize the relationship between a time series and its lagged version.
        In this specific plot, we have compared the original 'Close' values with their values lagged by a specified number of time steps.

        Here are some key observations:

        - If the points in the plot form a random scatter without a clear pattern, it suggests that there is **no significant autocorrelation** at the selected lag. This implies that the current observation is not dependent on its past values at that lag.

        - If there is a recognizable **pattern** in the plot, it may indicate the presence of **autocorrelation**. Autocorrelation means that there is a statistical relationship between a data point and its lagged versions at the selected lag. This can provide insights into the temporal dependency structure of the time series data.

        Lag plots are particularly useful in understanding the temporal behavior of time series data and can help identify patterns that may be exploited for forecasting or analysis.
        """)


    elif TS_analysis_option == "Lag Plot":
        st.subheader("Lag Plot Results")
        # Create a date selection widget to specify the start date
        start_date = st.date_input("Select Start Date", pd.to_datetime("2015-01-01"))

        # Convert the selected start date to a string
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Filter the DataFrame based on the selected start date
        filtered_data = BTC_data.loc[start_date_str:].copy()

        # Ensure that the index is a DatetimeIndex with a valid frequency
        filtered_data.index = pd.to_datetime(filtered_data.index)  # Convert to DatetimeIndex
        filtered_data.index.freq = 'D'  # Set frequency to 'D' for daily data (adjust as needed)

        # Create a slider to adjust the lag value
        lag = st.slider("Select Lag Value", min_value=1, max_value=365, value=1)

        # Create a lag column in the DataFrame
        filtered_data[f'Lag{lag}'] = filtered_data['Close'].shift(lag)

        # Plot the original data against the lagged data
        st.subheader(f"Lag Plot (Lag {lag})")
        fig, ax = plt.subplots()
        ax.scatter(filtered_data['Close'], filtered_data[f'Lag{lag}'], edgecolors='k')
        ax.set_xlabel('Close')
        ax.set_ylabel(f'Lagged Value (Lag {lag} time steps)')
        ax.set_title(f'Lag Plot (Lag {lag})')
        st.pyplot(fig)
        st.write("### Lag Plot:")
        st.write("A lag plot is used to visualize the relationship between a time series and its lagged version.")
        st.write(f"In this plot, we've compared the original 'Close' values with their values lagged by {lag} time steps.")
        st.write("Here are some key observations:")
        st.write("- If the points in the plot form a random scatter without a clear pattern, it suggests that there is no significant autocorrelation at the selected lag.")
        st.write("- If there is a pattern, it may indicate the presence of autocorrelation.")
        st.write("- Autocorrelation can provide insights into the temporal dependency structure of the time series data.")

           
     
    elif TS_analysis_option == "ACF Plot":
        st.subheader("ACF Plot Results")   
        
        # Create a date selection widget to specify the start date
        start_date = st.date_input("Select Start Date", pd.to_datetime("2015-01-01"))

        # Convert the selected start date to a string
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Filter the DataFrame based on the selected start date
        filtered_data = BTC_data.loc[start_date_str:].copy()

        # Ensure that the index is a DatetimeIndex with a valid frequency
        filtered_data.index = pd.to_datetime(filtered_data.index)  # Convert to DatetimeIndex
        filtered_data.index.freq = 'D'  # Set frequency to 'D' for daily data (adjust as needed)

        # Add a slider to adjust the number of lags
        max_lags = 365  # Set the maximum number of lags
        num_lags = st.slider("Select Number of Lags", 1, max_lags, 365)  # Adjust the range as needed

        # Plot the ACF with adjustable lags
        fig, ax = plt.subplots(figsize=(12, 6))
        plot_acf(filtered_data['Close'], lags=num_lags, zero=False, ax=ax)
        plt.title(f"ACF Bitcoin Close Price (Lags = {num_lags})", size=24)
        st.pyplot(fig)
        st.write("### ACF Plot: ")
        st.markdown("""
        An **Autocorrelation Function (ACF) plot** is a fundamental tool in time series analysis used to understand the correlation between a time series and its lagged versions.
        In this specific plot, we have visualized the ACF for Bitcoin's daily close prices.

        Here are some key takeaways:

        - The ACF plot shows **correlation coefficients** for different lags on the x-axis, indicating how closely each lagged value relates to the current observation.

        - The first bar (lag 0) always has a correlation coefficient of 1 because it represents the correlation of the time series with itself.

        - Significant spikes or bars extending beyond the shaded region indicate **significant autocorrelation** at those lags. This suggests that past values at those lags have an influence on the current value.

        - The shaded region represents the **95% confidence interval**. Correlation bars outside this region are considered statistically significant.

        ACF plots are essential in identifying patterns and dependencies in time series data, aiding in the selection of appropriate models for forecasting and understanding the temporal behavior of the data.
        """)
    

    #Obverse Seasonality
    elif TS_analysis_option == "Seasonality":
        st.subheader("Seasonal Decomposition (Multiplicative)")

        # Create a date selection widget to specify the start date
        start_date = st.date_input("Select Start Date", pd.to_datetime("2015-01-01"))

        # Convert the selected start date to a string
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Filter the DataFrame based on the selected start date
        filtered_data = BTC_data.loc[start_date_str:].copy()

        # Ensure that the index is a DatetimeIndex with a valid frequency
        filtered_data.index = pd.to_datetime(filtered_data.index)  # Convert to DatetimeIndex
        filtered_data.index.freq = 'D'  # Set frequency to 'D' for daily data (adjust as needed)

        # Perform seasonal decomposition with a multiplicative model for yearly seasonality
        s_dec_multiplicative_yearly = seasonal_decompose(filtered_data['Close'], model='multiplicative', period=365)

        # Perform seasonal decomposition with a multiplicative model for weekly seasonality
        s_dec_multiplicative_weekly = seasonal_decompose(filtered_data['Close'], model='multiplicative', period=7)

        # Create buttons to switch between yearly and weekly seasonality
        seasonality_option = st.radio("Select Seasonality Option:", ["Yearly Seasonality", "Weekly Seasonality"])

        # Plot decomposition components based on the selected option
        if seasonality_option == "Yearly Seasonality":
            st.subheader("Seasonal Decomposition Plot (Multiplicative Model) - Yearly Seasonality")
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
            ax1.plot(s_dec_multiplicative_yearly.trend)
            ax1.set_ylabel('Trend')
            ax2.plot(s_dec_multiplicative_yearly.seasonal)
            ax2.set_ylabel('Yearly Seasonality')
            ax3.plot(s_dec_multiplicative_yearly.resid)
            ax3.set_ylabel('Residuals')
            ax4.plot(s_dec_multiplicative_yearly.observed)
            ax4.set_ylabel('Observed')
            plt.xlabel('Date')
            st.pyplot(fig)
            
            st.write("### Yearly Seasonality Decomposition Summary")
            st.markdown("""
            Seasonal decomposition is a technique used to understand the underlying seasonal patterns in time series data.
            In this specific analysis, we applied a multiplicative model to decompose the Bitcoin's daily close prices into three components:

            - **Trend**: This component represents the long-term trend in the data, capturing any overall upward or downward movement.

            - **Yearly Seasonality**: This component represents the recurring seasonal pattern that occurs on a yearly basis. It captures any regular seasonality, such as holidays or annual events.

            - **Residuals**: These are the leftover variations in the data that are not explained by the trend or seasonal patterns.

            By decomposing the data, we gain insights into the underlying structure of the time series, which can be valuable for forecasting and understanding the impact of seasonality on the data.
            """)
            
        else:
            st.subheader("Seasonal Decomposition Plot (Multiplicative Model) - Weekly Seasonality")
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 8))
            ax1.plot(s_dec_multiplicative_weekly.trend)
            ax1.set_ylabel('Trend')
            ax2.plot(s_dec_multiplicative_weekly.seasonal)
            ax2.set_ylabel('Weekly Seasonality')
            ax3.plot(s_dec_multiplicative_weekly.resid)
            ax3.set_ylabel('Residuals')
            ax4.plot(s_dec_multiplicative_weekly.observed)
            ax4.set_ylabel('Observed')
            plt.xlabel('Date')
            st.pyplot(fig)
            st.write("### Weekly Seasonality Decomposition Summary")
            st.markdown("""
            In this analysis, we applied a multiplicative model to decompose Bitcoin's daily close prices into three key components:

            - **Trend**: This component captures the long-term trends in the data, which can be influenced by factors such as market conditions.

            - **Weekly Seasonality**: The weekly seasonality component captures recurring patterns that occur on a weekly basis. This could include variations in the data that repeat each week.

            - **Residuals**: The residuals represent the unexplained variations in the data that are not accounted for by the trend or weekly seasonality.

            Seasonal decomposition helps us better understand how different time components contribute to the overall behavior of the time series data. This knowledge can be crucial for making informed decisions and predictions.
            """)

        
    elif TS_analysis_option == "Stationarity":
        st.subheader("Stationarity Results - Augmented Dickey-Fuller Test ")    

        # Add a date selection widget to specify the start date
        start_date = st.date_input("Select Start Date", pd.to_datetime("2015-01-01"))

        # Convert the selected start date to a string
        start_date_str = start_date.strftime('%Y-%m-%d')

        # Filter the DataFrame based on the selected start date
        filtered_data = BTC_data.loc[start_date_str:].copy()

        # Perform the Augmented Dickey-Fuller test
        result = adfuller(filtered_data['Close'])

        # Display the results of the Augmented Dickey-Fuller test
        st.write(f"Test Statistic: {result[0]}")
        st.write(f"P-Value: {result[1]}")
        st.write(f"Lags Used: {result[2]}")
        st.write(f"Number of Observations Used: {result[3]}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"{key}: {value}")
            
        # Check stationarity based on the p-value
        alpha = 0.05  # Significance level
        p_value = result[1]

        st.subheader("Stationarity Assessment")
        
        
        
        if p_value <= alpha:
            st.write(f"P-Value ({p_value}) is less than or equal to Significance Level (alpha = {alpha}):")
            st.write("The data is stationary (Reject the null hypothesis)")
        else:
            st.write(f"P-Value ({p_value}) is greater than Significance Level (alpha = {alpha}):")
            st.write("The data is not stationary (Fail to reject the null hypothesis)")
       
        critical_values = result[4]
        for key, value in critical_values.items():
            st.write(f"{key} Critical Value ({alpha}): {value}")
    

        # Compare the test statistic to critical values
        if result[0] < critical_values["1%"]:
            st.write("The test statistic is less than the 1% critical value. Data is stationary at 1% significance level.")
        elif result[0] < critical_values["5%"]:
            st.write("The test statistic is less than the 5% critical value. Data is stationary at 5% significance level.")
        elif result[0] < critical_values["10%"]:
            st.write("The test statistic is less than the 10% critical value. Data is stationary at 10% significance level.")
        else:
            st.write("The test statistic is greater than all critical values. Data is not stationary.")


