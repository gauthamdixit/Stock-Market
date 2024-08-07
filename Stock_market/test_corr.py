import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting import Baseline, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import ta.trend
import torch
import matplotlib.pyplot as plt    
import ta

def test():    
    data = pd.read_csv('apple_stock_data.csv', index_col=0, sep=',', decimal='.')

    data.index = pd.to_datetime(data.index)
    data.sort_index(inplace=True)
    start_date = min(data.fillna(method='ffill').dropna().index)
    end_date = max(data.fillna(method='bfill').dropna().index)
    print(start_date)
    print(end_date)
    ts = data.fillna(0.)

    tmp = ts.copy()

    date = tmp.index
    earliest_time = data.index.min()

    tmp['days_from_start'] = (date - earliest_time).days
    tmp['date'] = date
    tmp['id'] = 1
    tmp['day'] = date.day
    tmp['day_of_week'] = date.dayofweek
    tmp['month'] = date.month
    tmp['MACD'] = ta.trend.MACD(tmp['Close']).macd()
    tmp['MACD_Signal'] = ta.trend.MACD(tmp['Close']).macd_signal()
    tmp['MACD_Hist'] = ta.trend.MACD(tmp['Close']).macd_diff()
    tmp['EMA_20'] = ta.trend.EMAIndicator(tmp['Close'], window=20).ema_indicator()
    tmp['SMA_50'] = ta.trend.SMAIndicator(tmp['Close'], window=50).sma_indicator()
    tmp['ADX'] = ta.trend.ADXIndicator(tmp['High'], tmp['Low'], tmp['Close'], window=14).adx()
    tmp['CCI'] = ta.trend.CCIIndicator(tmp['High'], tmp['Low'], tmp['Close'], window=20).cci()
    tmp.drop(columns=['Healthcare','Financials','Consumer Discretionary','Consumer Staples','Utilities','Materials','Real Estate','Communication Services','Industrials','Technology', 'Energy'], inplace=True)
    
        
    total_days = (data.index.max() - data.index.min()).days
    time_df = tmp
    
    # Calculate the correlation matrix
    correlation_matrix = time_df.corr()
    
    # Extract the correlation values for the 'Close' column
    correlations = correlation_matrix['Close']
    
    print(correlations)

test()
