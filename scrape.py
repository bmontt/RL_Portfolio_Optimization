import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import StockPrice, NewsArticle, Base
from datetime import datetime

# Load configuration from config.json
with open('Rl_Portfolio_Optimization/config.json', 'r') as file:
    config = json.load(file)

alpha_vantage_api_key = config['alpha_vantage_api_key']
tickers = config['tickers']
yahoo_finance_url = config['news_sources']['yahoo_finance']['url']
cnbc_finance_url = config['news_sources']['cnbc_finance']['url']
db_config = config['database']

# Create database engine and session
engine = create_engine(f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['name']}")
Session = sessionmaker(bind=engine)
session = Session()

def scrape_yahoo_finance(ticker):
    """
    Scrapes historical stock price data from Yahoo Finance for a given ticker symbol.

    Args:
        ticker (str): The ticker symbol of the stock.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data with columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    try:
        url = yahoo_finance_url.format(ticker=ticker)
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        table = soup.find('table', {'data-test': 'historical-prices'})
        rows = table.find_all('tr')

        data = []
        for row in rows[1:]:  # Skip header row
            cols = row.find_all('td')
            if len(cols) == 7:  # Ensure it has the correct number of columns
                data.append([col.text for col in cols])

        df = pd.DataFrame(data, columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'])
        return df
    except AttributeError:
        return pd.DataFrame()

def scrape_alpha_vantage(api_key, ticker):
    """
    Scrapes daily stock price data from Alpha Vantage for a given ticker symbol.

    Args:
        api_key (str): Your Alpha Vantage API key.
        ticker (str): The ticker symbol of the stock.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data with columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    try:
        url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&apikey={api_key}"
        response = requests.get(url)
        data = response.json()['Time Series (Daily)']

        df = pd.DataFrame.from_dict(data, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        df.index.name = 'Date'
        df.reset_index(inplace=True)
        return df
    except (AttributeError, KeyError):
        return pd.DataFrame()

def scrape_financial_news():
    """
    Scrapes financial news articles from CNBC.

    Returns:
        pandas.DataFrame: A DataFrame containing the scraped data with columns: 'Title', 'Link'.
    """
    try:
        url = cnbc_finance_url
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('div', class_='Card-titleContainer')

        data = []
        for article in articles:
            title = article.find('a').text.strip()
            link = article.find('a')['href']
            data.append({'Title': title, 'Link': link, 'Source': 'CNBC'})

        df = pd.DataFrame(data)
        return df
    except AttributeError:
        return pd.DataFrame()

# Function to save stock prices to the database
def save_stock_prices(df, ticker):
    for _, row in df.iterrows():
        stock_price = StockPrice(
            ticker=ticker,
            date=datetime.strptime(row['Date'], '%Y-%m-%d'),
            open=float(row['Open'].replace(',', '')),
            high=float(row['High'].replace(',', '')),
            low=float(row['Low'].replace(',', '')),
            close=float(row['Close'].replace(',', '')),
            volume=int(row['Volume'].replace(',', ''))
        )
        session.add(stock_price)
    session.commit()

# Function to save news articles to the database
def save_news_articles(df):
    for _, row in df.iterrows():
        news_article = NewsArticle(
            title=row['Title'],
            link=row['Link'],
            source=row['Source']
        )
        session.add(news_article)
    session.commit()

# Main function to execute scraping simultaneously
def main():
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(scrape_yahoo_finance, ticker) for ticker in tickers
        ]
        futures.append(executor.submit(scrape_alpha_vantage, alpha_vantage_api_key, tickers[0]))
        futures.append(executor.submit(scrape_financial_news))

        results = [future.result() for future in futures]

    yahoo_finance_dfs = results[:len(tickers)]
    alpha_vantage_df = results[len(tickers)]
    financial_news_df = results[-1]

    # Save data to the database
    for i, ticker in enumerate(tickers):
        save_stock_prices(yahoo_finance_dfs[i], ticker)
    save_stock_prices(alpha_vantage_df, tickers[0])
    save_news_articles(financial_news_df)

    print("Data has been successfully saved to the database.")

if __name__ == "__main__":
    main()