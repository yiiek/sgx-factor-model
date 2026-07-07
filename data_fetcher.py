import yfinance as yf
import pandas as pd
import numpy as np

class DataFetcher:
    def __init__(self, tickers, benchmark='^STI'):
        """
        Initializes the data acquisition layer.
        tickers: List of user-selected SGX stock tickers (e.g., ['D05.SI', 'U11.SI'])
        benchmark: The baseline tracking index (Straits Times Index '^STI')
        """
        self.tickers = list(tickers)
        self.benchmark = benchmark
        self.all_tickers = self.tickers + [self.benchmark]

    def fetch_historical_prices(self, period="2y"):
        """
        Fetches historical closing prices for all assets and the benchmark index.
        A 2-year lookback period (~504 trading days) provides enough data runway
        for a 1-year rolling beta lookback plus a 60-day factor return window.
        """
        data = yf.download(self.all_tickers, period=period, progress=False)
        if 'Adj Close' in data:
            prices = data['Adj Close']
        else:
            prices = data['Close']
            
        # Clean structural data gaps (halts/weekends) common in SGX microcaps
        return prices.ffill().bfill()

    def fetch_metadata(self):
        """
        Extracts structural company fundamentals required for rolling custom calculations
        and descriptive labels for the user interface.
        """
        metadata = {}
        for ticker in self.tickers:
            try:
                t_obj = yf.Ticker(ticker)
                info = t_obj.info
                metadata[ticker] = {
                    'sharesOutstanding': info.get('sharesOutstanding', np.nan),
                    'trailingEps': info.get('trailingEps', np.nan),
                    'longName': info.get('longName', ticker)  # <--- ADDED: Fallback to ticker if missing
                }
            except Exception:
                metadata[ticker] = {'sharesOutstanding': np.nan, 'trailingEps': np.nan, 'longName': ticker}
        return pd.DataFrame.from_dict(metadata, orient='index')