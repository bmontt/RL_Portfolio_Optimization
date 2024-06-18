import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # Correct import path
import gym
from gym import spaces
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import StockPrice, NewsArticle, Base
from datetime import datetime
from sklearn.metrics import mean_squared_error
import time

# Load configuration from config.json
with open('RL_Portfolio_Optimization/config.json', 'r') as file:
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
    # Ensure unique index values
    for ticker, df in data.items():
        df.reset_index(inplace=True)
        df.drop_duplicates(subset=['Date'], keep='last', inplace=True)
        df.set_index('Date', inplace=True)
    
    combined_df = pd.concat(data.values(), axis=1, keys=data.keys())
    combined_df.ffill(inplace=True)
    combined_df.bfill(inplace=True)
    combined_df = combined_df.infer_objects(copy=False)
    
    # Feature Engineering
    for ticker in data.keys():
        df = combined_df[ticker]
        df.loc[:, 'SMA_20'] = df['Close'].rolling(window=20).mean()
        df.loc[:, 'SMA_50'] = df['Close'].rolling(window=50).mean()
        df.loc[:, 'RSI'] = compute_RSI(df['Close'])
        df.loc[:, 'MACD'] = compute_MACD(df['Close'])
    
    combined_df.fillna(0, inplace=True)
    combined_df.replace([np.inf, -np.inf], 0, inplace=True)  # Replace infinite values with 0
    combined_df = combined_df.applymap(lambda x: 0 if pd.isnull(x) else x)  # Replace NaNs with 0
    
    # Debug: Check for NaN values
    if combined_df.isnull().values.any():
        print("NaN values found in combined_df after preprocessing.")
        print(combined_df[combined_df.isnull().any(axis=1)])
        time.sleep(10)  # Pause for inspection
    
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

class StockPortfolioEnv(gym.Env):
    def __init__(self, df, initial_amount=10000, transaction_cost=0.001, lookback=50):
        try:
            self.df = df
            self.initial_amount = initial_amount
            self.transaction_cost = transaction_cost
            self.lookback = lookback
            self.n_stock = len(df.columns.levels[0])  # number of stocks
            self.n_features = len(df.columns.levels[1])  # number of features
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.n_stock,))
            self.observation_space = spaces.Box(low=0, high=1, shape=(lookback, self.n_stock * self.n_features + 1))
            print(f"Observation space shape: {self.observation_space.shape}")
        except Exception as e:
            print(f"Error in initializing StockPortfolioEnv: {e}")

    def reset(self):
        try:
            self.current_step = self.lookback
            self.balance = self.initial_amount
            self.stock_owned = np.zeros(self.n_stock)
            self.stock_price = self._get_stock_prices()
            self.portfolio_value = self.initial_amount
            obs = self._get_observation()
            print(f"Reset observation shape: {obs.shape}")
            return obs
        except Exception as e:
            print(f"Error in reset: {e}")

    def _get_observation(self):
        try:
            obs = self.stock_price
            balance_array = np.array([self.balance] * self.lookback).reshape(-1, 1)
            obs = np.hstack((obs, balance_array))
            obs = np.nan_to_num(obs)  # Replace NaNs with 0
            obs = np.where(np.isfinite(obs), obs, 0)  # Ensure no infinite values
            
            # Debug: Check for NaN values
            if np.isnan(obs).any():
                print("NaN values found in observation.")
            
            print(f"Observation shape: {obs.shape}")
            return obs
        except Exception as e:
            print(f"Error in _get_observation: {e}")

    def step(self, action):
        try:
            prev_portfolio_value = self.portfolio_value
            self._trade(action)
            self.current_step += 1
            self.stock_price = self._get_stock_prices()
            self.portfolio_value = self._get_portfolio_value()
            reward = self.portfolio_value - prev_portfolio_value
            done = self.current_step == len(self.df) - 1
            obs = self._get_observation()
            return obs, reward, done, {}
        except Exception as e:
            print(f"Error in step: {e}")

    def _trade(self, action):
        try:
            action = (action + 1) / 2  # map actions to [0, 1]
            available_cash = self.balance * (1 - self.transaction_cost)
            for i in range(self.n_stock):
                self.stock_owned[i] += available_cash * action[i] / self.stock_price[-1][i * self.n_features + 3]  # Index for 'Close' prices
            self.balance -= available_cash
        except Exception as e:
            print(f"Error in _trade: {e}")

    def _get_portfolio_value(self):
        try:
            close_prices = self.stock_price[-1][3::self.n_features]  # Extract closing prices from 1-d array
            return self.balance + np.sum(self.stock_owned * close_prices)
        except Exception as e:
            print(f"Error in _get_portfolio_value: {e}")

    def _get_stock_prices(self):
        try:
            prices = self.df.iloc[self.current_step - self.lookback:self.current_step].values
            prices = np.nan_to_num(prices)  # Replace NaNs with 0
            prices = np.where(np.isfinite(prices), prices, 0)  # Ensure no infinite values
            # Debug: Check for NaN values
            if np.isnan(prices).any():
                print("NaN values found in stock prices.")
            
            print(f"Stock prices shape: {prices.shape}")
            return prices
        except Exception as e:
            print(f"Error in _get_stock_prices: {e}")

def train_model(df):
    env = DummyVecEnv([lambda: StockPortfolioEnv(df)])
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=20000)
    return model

def evaluate_model(model, df):
    env = DummyVecEnv([lambda: StockPortfolioEnv(df)])
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward

def main():
    tickers = config['tickers']
    data = load_data(tickers)
    processed_data = preprocess_data(data)
    
    # Normalize the data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(processed_data.values.reshape(-1, processed_data.shape[-1])).reshape(processed_data.shape)
    processed_data = pd.DataFrame(scaled_data, index=processed_data.index, columns=processed_data.columns)
    
    # Debug: Check for NaN values
    if processed_data.isnull().values.any():
        print("NaN values found in processed_data after normalization.")
        print(processed_data[processed_data.isnull().any(axis=1)])
    
    train_data, validation_data, test_data = split_data(processed_data)
    
    model = train_model(train_data)
    model.save("ppo_stock_portfolio")

    # Evaluate the model
    validation_reward = evaluate_model(model, validation_data)
    test_reward = evaluate_model(model, test_data)
    
    print(f"Validation Reward: {validation_reward}")
    print(f"Test Reward: {test_reward}")

if __name__ == "__main__":
    main()
