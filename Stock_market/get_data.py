import pandas as pd
import requests
import yfinance as yf
from alpha_vantage.alpha_vantage.techindicators import TechIndicators
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta

# Define the start date for 6 years ago
start_date = datetime.now() - timedelta(days=6*365)
end_date = datetime.now()

# Create a common date range that covers the past 6 years
date_range = pd.date_range(start=start_date, end=end_date, freq='D')

def fill_with_rolling_mean(df, window=50):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = df[numeric_cols].apply(
        lambda x: x.fillna(x.rolling(window=window, min_periods=1, center=True).mean()), axis=0
    )
    return df

# Function to get historical data from Yahoo Finance
def get_historical_data(symbol):
    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    hist.index = hist.index.tz_localize(None)  # Ensure timezone-naive
    hist = hist.reindex(date_range, method='ffill')  # Align dates to the common date range, forward fill missing data
    hist['Date'] = hist.index  # Add Date column
    hist_filled = fill_with_rolling_mean(hist)
    return hist_filled[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Function to get technical indicators from Alpha Vantage
def get_technical_indicators(symbol):
    ti = TechIndicators(key='###########', output_format='pandas')
    data, _ = ti.get_rsi(symbol=symbol, interval='daily')
    data.index = data.index.tz_localize(None)  # Ensure timezone-naive
    data = data.reindex(date_range, method='ffill')  # Align dates to the common date range, forward fill missing data
    data['Date'] = data.index  # Add Date column
    data_filled = fill_with_rolling_mean(data)
    return data_filled[['Date', 'RSI']]

# Function to get market sentiment
def get_market_sentiment():
    url = f'https://api.alternative.me/fng/?limit=10000'
    response = requests.get(url)
    sentiment = response.json()
    sentiment_df = pd.DataFrame(sentiment['data'])
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['timestamp'], unit='s')
    sentiment_df.set_index('Date', inplace=True)
    sentiment_df = sentiment_df.reindex(date_range, method='ffill')  # Align dates to the common date range, forward fill missing data
    sentiment_df.drop('timestamp', axis=1, inplace=True)
    sentiment_df['Date'] = sentiment_df.index  # Add Date column
    sentiment_df_filled = fill_with_rolling_mean(sentiment_df)
    return sentiment_df_filled[['Date', 'value']]


# Define industry ETFs
industry_etfs = {
    'Technology': 'XLK',
    'Healthcare': 'XLV',
    'Financials': 'XLF',
    'Consumer Discretionary': 'XLY',
    'Industrials': 'XLI',
    'Energy': 'XLE',
    'Consumer Staples': 'XLP',
    'Utilities': 'XLU',
    'Materials': 'XLB',
    'Real Estate': 'XLRE',
    'Communication Services': 'XLC'
}

# Function to get daily industry performance for the past 6 years
def get_daily_industry_performance():
    all_performance_data = {}

    for sector, etf in industry_etfs.items():
        data = yf.Ticker(etf).history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        data.index = data.index.tz_localize(None)  # Ensure timezone-naive
        
        # Calculate daily performance as percentage change
        data['Performance'] = data['Close'].pct_change() * 100
        
        # Reindex to ensure the dates align and fill missing dates with NaN
        performance = data['Performance'].reindex(date_range, method='ffill')
        
        # Add the performance data to the DataFrame
        all_performance_data[sector] = performance

    # Combine all performance data into a single DataFrame
    combined_df = pd.DataFrame(all_performance_data)
    combined_df.reset_index(inplace=True)
    combined_df.rename(columns={'index': 'Date'}, inplace=True)
    combined_df_filled = fill_with_rolling_mean(combined_df)
    return combined_df_filled



# Collecting the data for Apple Inc. (AAPL)
symbol = 'AAPL'

market_sentiment_df = get_market_sentiment()
market_sentiment_df.to_csv('apple_sentiment.csv', index=False)

historical_data_df = get_historical_data(symbol)
historical_data_df.to_csv('apple_historicals.csv', index=False)

technical_indicators_df = get_technical_indicators(symbol)
technical_indicators_df.to_csv('apple_techIndicators.csv', index=False)

industry_performance_df = get_daily_industry_performance()
industry_performance_df.to_csv('apple_industryPerform.csv', index=False)

# Load the RSI data, explicitly parsing the 'Date' column
# rsi = pd.read_csv('apple_rsi_filtered.csv', index_col=0, sep=',', decimal='.')

# # Check the first few rows to ensure 'Date' is parsed correctly
# print(rsi.head())

# # Filter the data from the date 2018-06-16 onwards
# filtered_rsi = rsi.loc['2018-06-16':, ['RSI']]
# filtered_rsi.index.name = 'Date'

# # Define the date range
# start_date = '2018-06-16'
# end_date = rsi.index.max()

# # Create a date range with all days between start_date and end_date
# all_days = pd.date_range(start=start_date, end=end_date, freq='D')

# # Reindex the DataFrame to include all days in the range
# filtered_rsi = filtered_rsi.reindex(all_days)

# # Fill missing values if necessary and drop rows with NaN values
# filtered_rsi = filtered_rsi.fillna(method='ffill').dropna()

# # Save the filtered RSI data to a new CSV file
# filtered_rsi.to_csv('apple_rsi_filtered.csv', index=True)

# File paths for the CSV files
file1 = 'apple_historicals.csv'
file2 = 'apple_techIndicators.csv'
file3 = 'apple_sentiment.csv'
file4 = 'apple_industryPerform.csv'

# Read each CSV file into a DataFrame
df1 = pd.read_csv(file1, parse_dates=['Date'])
df2 = pd.read_csv(file2, parse_dates=['Date'])
df3 = pd.read_csv(file3, parse_dates=['Date'])
df4 = pd.read_csv(file4, parse_dates=['Date'])

# Ensure the Date column is set as index for each DataFrame
df1.set_index('Date', inplace=True)
df2.set_index('Date', inplace=True)
df3.set_index('Date', inplace=True)
df4.set_index('Date', inplace=True)

# Concatenate the DataFrames along columns
combined_df = pd.concat([df1, df2,df3, df4], axis=1)
combined_df.fillna(combined_df.mean())

# Save combined dataframe to CSV
combined_df.to_csv('apple_stock_data.csv', index=True)
