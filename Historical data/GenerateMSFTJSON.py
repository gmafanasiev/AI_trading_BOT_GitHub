import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class MSFTDataProcessor:
    def __init__(self, MSFT_file, spy_file, output_json='historical_analysis_MSFT.json'):
        """Initialize with MSFT and SPY Excel files and JSON output file."""
        self.MSFT_file = MSFT_file
        self.spy_file = spy_file
        self.output_json = output_json
        self.data = None
        self.market_data = None
        self.analysis_results = {}

    def load_data(self):
        """Load and clean MSFT and SPY data from Excel files, assuming headers in first row."""
        # Load MSFT data
        try:
            df = pd.read_excel(self.MSFT_file, sheet_name='MSFT', skiprows=0)
        except Exception as e:
            raise ValueError(f"Error reading MSFT Excel file {self.MSFT_file}, sheet MSFT: {str(e)}")

        expected_columns = ['Date', 'Symbol', 'Open', 'High', 'Low', 'Close', 'Volume']
        print(f"Processing MSFT sheet")
        print(f"Columns: {df.columns.tolist()}")
        print(f"First 5 rows:\n{df.head().to_string()}\n")
        print(f"Date column dtype: {df['Date'].dtype}")
        print(f"First 5 Date values: {df['Date'].head().tolist()}\n")

        # Check for required columns
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"MSFT sheet is missing columns: {missing_cols}")

        # Handle MSFT Date column (MM/DD/YYYY format)
        try:
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce')
            if df['Date'].isna().any():
                invalid_dates = df[df['Date'].isna()][['Date']].head()
                print(f"MSFT - Dropping {df['Date'].isna().sum()} rows with invalid dates:\n{invalid_dates.to_string()}\n")
                df = df.dropna(subset=['Date'])
        except Exception as e:
            raise ValueError(f"Error converting Date column in MSFT sheet to datetime (format MM/DD/YYYY): {str(e)}\nFirst 5 Date values: {df['Date'].head().tolist()}")

        # Clean MSFT numeric columns
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isna().any():
                print(f"MSFT - Dropping {df[col].isna().sum()} rows with invalid values in {col}")
                df = df.dropna(subset=[col])

        # Ensure Symbol column is string
        df['Symbol'] = df['Symbol'].astype(str)

        # Drop any remaining rows with NaN values
        df = df.dropna()

        # Set Date as index
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)

        print(f"MSFT - First 5 rows after cleaning:\n{df.head().to_string()}\n")
        print(f"MSFT - Volume for 2025-07-17: {df.loc['2025-07-17', 'Volume'] if '2025-07-17' in df.index else 'Not found'}")
        self.data = df

        # Load SPY data
        try:
            spy = pd.read_excel(self.spy_file, sheet_name='SPY', skiprows=0)
        except Exception as e:
            raise ValueError(f"Error reading SPY Excel file {self.spy_file}, sheet SPY: {str(e)}")

        print(f"Processing SPY sheet")
        print(f"Columns: {spy.columns.tolist()}")
        print(f"First 5 rows:\n{spy.head().to_string()}\n")
        print(f"Date column dtype: {spy['Date'].dtype}")
        print(f"First 5 Date values: {spy['Date'].head().tolist()}\n")

        # Check for required columns
        missing_cols = [col for col in expected_columns if col not in spy.columns]
        if missing_cols:
            raise ValueError(f"SPY sheet is missing columns: {missing_cols}")

        # Handle SPY Date column (MM/DD/YYYY format)
        try:
            spy['Date'] = pd.to_datetime(spy['Date'], format='%m/%d/%Y', errors='coerce')
            if spy['Date'].isna().any():
                invalid_dates = spy[spy['Date'].isna()][['Date']].head()
                print(f"SPY - Dropping {spy['Date'].isna().sum()} rows with invalid dates:\n{invalid_dates.to_string()}\n")
                spy = spy.dropna(subset=['Date'])
        except Exception as e:
            raise ValueError(f"Error converting Date column in SPY sheet to datetime (format MM/DD/YYYY): {str(e)}\nFirst 5 Date values: {spy['Date'].head().tolist()}")

        # Clean SPY numeric columns
        for col in ['Close']:
            spy[col] = pd.to_numeric(spy[col], errors='coerce')
            if spy[col].isna().any():
                print(f"SPY - Dropping {spy[col].isna().sum()} rows with invalid values in {col}")
                spy = spy.dropna(subset=[col])

        # Set Date as index and align with MSFT dates
        spy.set_index('Date', inplace=True)
        spy = spy[['Close']].rename(columns={'Close': 'Market_Close'})
        spy = spy.reindex(df.index, method='ffill')  # Align with MSFT dates
        if spy['Market_Close'].isna().any():
            print("SPY - Filling missing market data with forward-fill")
            spy = spy.fillna(method='ffill')

        print(f"SPY - First 5 rows after cleaning:\n{spy.head().to_string()}\n")
        self.market_data = spy

    def calculate_moving_averages(self, df, short_window=50, long_window=200):
        """Calculate 50-day and 200-day SMAs."""
        df['SMA_50'] = df['Close'].rolling(window=short_window).mean()
        df['SMA_200'] = df['Close'].rolling(window=long_window).mean()
        return df

    def calculate_volatility(self, df, window=21):
        """Calculate 21-day rolling annualized volatility."""
        daily_returns = np.log(df['Close'] / df['Close'].shift(1))
        df['Volatility'] = daily_returns.rolling(window=window).std() * np.sqrt(252)
        return df

    def calculate_rsi(self, df, window=14):
        """Calculate 14-day RSI."""
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def calculate_roc(self, df, window=10):
        """Calculate 10-day Rate of Change."""
        df['ROC'] = (df['Close'] - df['Close'].shift(window)) / df['Close'].shift(window) * 100
        return df

    def calculate_adx(self, df, window=14):
        """Calculate 14-day ADX."""
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        dm_plus = high_low.where((high_low > low_close) & (high_low > 0), 0)
        dm_minus = low_close.where((low_close > high_low) & (low_close > 0), 0)
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()
        di_plus = (dm_plus.rolling(window=window).mean() / atr) * 100
        di_minus = (dm_minus.rolling(window=window).mean() / atr) * 100
        dx = ((di_plus - di_minus).abs() / (di_plus + di_minus)) * 100
        df['ADX'] = dx.rolling(window=window).mean()
        return df

    def calculate_price_volume_correlation(self, df, window=20):
        """Calculate rolling price-volume correlation."""
        price_changes = df['Close'].diff()
        df['Price_Volume_Corr'] = price_changes.rolling(window=window).corr(df['Volume'])
        return df

    def calculate_market_correlation(self, df, market_df, window=20):
        """Calculate rolling correlation with market (SPY)."""
        df_returns = np.log(df['Close'] / df['Close'].shift(1))
        market_returns = np.log(market_df['Market_Close'] / market_df['Market_Close'].shift(1))
        df['Market_Corr'] = df_returns.rolling(window=window).corr(market_returns)
        return df

    def identify_support_resistance(self, df, window=50, threshold=0.02):
        """Identify support and resistance levels."""
        support_levels = []
        resistance_levels = []
        rolling_low = df['Low'].rolling(window=window, center=True).min()
        rolling_high = df['High'].rolling(window=window, center=True).max()
        for date, row in df.iterrows():
            if not pd.isna(rolling_low[date]) and abs(row['Low'] - rolling_low[date]) / row['Low'] < threshold:
                support_levels.append((str(date), float(row['Low'])))
            if not pd.isna(rolling_high[date]) and abs(row['High'] - rolling_high[date]) / row['High'] < threshold:
                resistance_levels.append((str(date), float(row['High'])))
        return {'support': support_levels, 'resistance': resistance_levels}

    def run_analysis(self):
        """Run comprehensive analysis and store results for MSFT."""
        self.load_data()
        df = self.data.copy()
        df = self.calculate_moving_averages(df)
        df = self.calculate_volatility(df)
        df = self.calculate_rsi(df)
        df = self.calculate_roc(df)
        df = self.calculate_adx(df)
        df = self.calculate_price_volume_correlation(df)
        df = self.calculate_market_correlation(df, self.market_data)
        sr_levels = self.identify_support_resistance(df)
        latest_metrics = {
            'Symbol': 'MSFT',
            'Date': str(df.index[-1]),
            'Close': float(df['Close'].iloc[-1]),
            'SMA_50': float(df['SMA_50'].iloc[-1]) if not pd.isna(df['SMA_50'].iloc[-1]) else None,
            'SMA_200': float(df['SMA_200'].iloc[-1]) if not pd.isna(df['SMA_200'].iloc[-1]) else None,
            'Volatility': float(df['Volatility'].iloc[-1]) if not pd.isna(df['Volatility'].iloc[-1]) else None,
            'RSI': float(df['RSI'].iloc[-1]) if not pd.isna(df['RSI'].iloc[-1]) else None,
            'ROC': float(df['ROC'].iloc[-1]) if not pd.isna(df['ROC'].iloc[-1]) else None,
            'ADX': float(df['ADX'].iloc[-1]) if not pd.isna(df['ADX'].iloc[-1]) else None,
            'Price_Volume_Corr': float(df['Price_Volume_Corr'].iloc[-1]) if not pd.isna(df['Price_Volume_Corr'].iloc[-1]) else None,
            'Market_Corr': float(df['Market_Corr'].iloc[-1]) if not pd.isna(df['Market_Corr'].iloc[-1]) else None
        }
        data_dict = df[['Close', 'Volume', 'SMA_50', 'SMA_200', 'Volatility', 'RSI', 'ROC', 'ADX', 'Price_Volume_Corr', 'Market_Corr']].to_dict(orient='index')
        data_dict = {str(k): {col: float(v[col]) if not pd.isna(v[col]) else None for col in v} for k, v in data_dict.items()}
        self.analysis_results['MSFT'] = {
            'data': data_dict,
            'support_resistance': sr_levels,
            'latest_metrics': latest_metrics
        }

    def save_analysis(self):
        """Save analysis results to JSON file."""
        with open(self.output_json, 'w') as f:
            json.dump(self.analysis_results, f, indent=4)
        print(f"JSON file saved to: {self.output_json}")

if __name__ == "__main__":
    MSFT_file = "/home/gmafanasiev/MSFT since 2017-01-01.xlsx"
    spy_file = "/home/gmafanasiev/SPY since 2017-01-01.xlsx"
    output_json = "/home/gmafanasiev/historical_analysis_MSFT.json"
    processor = MSFTDataProcessor(MSFT_file, spy_file, output_json)
    processor.run_analysis()
    processor.save_analysis()