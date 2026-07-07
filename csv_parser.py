import pandas as pd
import os

class CSVParser:
    @staticmethod
    def parse_stocks_csv(file_path):
        """
        Parses the stock configuration CSV.
        Expected columns: Ticker, Market, Banks, Real Estate, Others
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Configuration file not found at: {file_path}")
            
        df = pd.read_csv(file_path)
        required_cols = ['Ticker', 'Market', 'Banks', 'Real Estate', 'Others']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column in CSV: {col}")
        
        # Clean and suffix tickers for yfinance
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        df['yf_ticker'] = df['Ticker'].apply(lambda x: x if x.endswith('.SI') else f"{x}.SI")
        
        return df

    @staticmethod
    def parse_portfolio_csv(file_path):
        """
        Parses current portfolio positions.
        Expected columns: Ticker, Shares
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Portfolio file not found at: {file_path}")
            
        df = pd.read_csv(file_path)
        for col in ['Ticker', 'Shares']:
            if col not in df.columns:
                raise ValueError(f"Missing portfolio column: {col}")
                
        df['Ticker'] = df['Ticker'].astype(str).str.strip().str.upper()
        df['yf_ticker'] = df['Ticker'].apply(lambda x: x if x.endswith('.SI') else f"{x}.SI")
        return df