import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import StockPrice, NewsArticle, Base
from datetime import datetime

# Load configuration from config.json
with open('Rl_Portfolio_Optimization/config.json', 'r') as file:
    config = json.load(file)

db_config = config['database']

# Create database engine and session
engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}")
Session = sessionmaker(bind=engine)
session = Session()

def load_data(tickers):
    data = {}
    for ticker in tickers:
        query = session.query(StockPrice).filter(StockPrice.ticker == ticker).all()
        df = pd.DataFrame([(row.date, row.open, row.high, row.low, row.close, row.volume) for row in query],
                          columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df.set_index('Date', inplace=True)
        data[ticker] = df
    return data

def preprocess_data(data):
    combined_df = pd.concat(data.values(), axis=1, keys=data.keys())
    combined_df.fillna(method='ffill', inplace=True)
    combined_df.fillna(method='bfill', inplace=True)
    
    # Feature Engineering
    for ticker in data.keys():
        df = combined_df[ticker]
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['RSI'] = compute_RSI(df['Close'])
        df['MACD'] = compute_MACD(df['Close'])
    
    combined_df.fillna(0, inplace=True)
    return combined_df

def compute_RSI(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_MACD(series, short_period=12, long_period=26, signal_period=9):
    short_ema = series.ewm(span=short_period, adjust=False).mean()
    long_ema = series.ewm(span=long_period, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd - signal

def split_data(df, train_size=0.7, validation_size=0.15):
    n = len(df)
    train_end = int(train_size * n)
    validation_end = train_end + int(validation_size * n)
    
    train_data = df.iloc[:train_end]
    validation_data = df.iloc[train_end:validation_end]
    test_data = df.iloc[validation_end:]
    
    return train_data, validation_data, test_data

def main():
    tickers = config['tickers']
    data = load_data(tickers)
    processed_data = preprocess_data(data)
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(processed_data.values.reshape(-1, processed_data.shape[-1])).reshape(processed_data.shape)
    processed_data = pd.DataFrame(scaled_data, index=processed_data.index, columns=processed_data.columns)
    
    train_data, validation_data, test_data = split_data(processed_data)
    
    # Print shapes of the splits for verification
    print(f"Train Data Shape: {train_data.shape}")
    print(f"Validation Data Shape: {validation_data.shape}")
    print(f"Test Data Shape: {test_data.shape}")

if __name__ == "__main__":
    main()
