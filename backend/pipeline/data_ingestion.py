"""
Data Ingestion Module
Fetches historical OHLCV data for 20 Nifty 50 stocks, the Nifty 50 index,
and India VIX from Yahoo Finance.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import requests
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# 20 representative Nifty 50 stocks
NIFTY_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "MARUTI.NS", "HCLTECH.NS",
    "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ADANIENT.NS",
]

NIFTY_INDEX = "^NSEI"
INDIA_VIX = "^INDIAVIX"


def fetch_stock_data(years: int = 11) -> dict:
    """
    Fetch OHLCV data for all stocks, Nifty index, and India VIX.

    Args:
        years: Number of years of historical data to fetch (5-10 recommended).

    Returns:
        Dictionary with keys:
            'stocks': {ticker: DataFrame} for each stock
            'nifty': DataFrame for Nifty 50 index
            'vix': DataFrame for India VIX (or synthetic proxy)
            'metadata': dict with fetch info
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    })

    logger.info(f"Fetching data from {start_str} to {end_str} ({years} years)")

    result = {"stocks": {}, "nifty": None, "vix": None, "metadata": {}}

    # --- Fetch Nifty 50 Index ---
    logger.info("Fetching Nifty 50 index data...")
    nifty_data = yf.download(NIFTY_INDEX, start=start_str, end=end_str, session=session, progress=False)
    if nifty_data.empty:
        raise ValueError("Failed to fetch Nifty 50 index data")
    # Flatten multi-level columns if present
    if isinstance(nifty_data.columns, pd.MultiIndex):
        nifty_data.columns = nifty_data.columns.get_level_values(0)
    result["nifty"] = nifty_data
    logger.info(f"  Nifty 50: {len(nifty_data)} trading days fetched")

    # --- Fetch India VIX ---
    logger.info("Fetching India VIX data...")
    try:
        vix_data = yf.download(INDIA_VIX, start=start_str, end=end_str, session=session, progress=False)
        if isinstance(vix_data.columns, pd.MultiIndex):
            vix_data.columns = vix_data.columns.get_level_values(0)
        if vix_data.empty or len(vix_data) < 100:
            raise ValueError("Insufficient VIX data")
        result["vix"] = vix_data
        logger.info(f"  India VIX: {len(vix_data)} trading days fetched")
    except Exception as e:
        logger.warning(f"  India VIX fetch failed ({e}), computing synthetic proxy")
        nifty_returns = np.log(nifty_data["Close"] / nifty_data["Close"].shift(1))
        synthetic_vix = nifty_returns.rolling(20).std() * np.sqrt(252) * 100
        result["vix"] = pd.DataFrame({"Close": synthetic_vix}, index=nifty_data.index)
        logger.info("  Synthetic VIX proxy computed from Nifty realized volatility")

    # --- Fetch Individual Stocks ---
    failed_tickers = []
    for i, ticker in enumerate(NIFTY_STOCKS):
        logger.info(f"Fetching {ticker} ({i+1}/{len(NIFTY_STOCKS)})...")
        try:
            stock_data = yf.download(ticker, start=start_str, end=end_str, session=session, progress=False)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.get_level_values(0)
            if stock_data.empty or len(stock_data) < 252:
                logger.warning(f"  {ticker}: Insufficient data ({len(stock_data)} days), skipping")
                failed_tickers.append(ticker)
                continue
            result["stocks"][ticker] = stock_data
            logger.info(f"  {ticker}: {len(stock_data)} trading days fetched")
        except Exception as e:
            logger.error(f"  {ticker}: Fetch failed — {e}")
            failed_tickers.append(ticker)

    # --- Metadata ---
    result["metadata"] = {
        "start_date": start_str,
        "end_date": end_str,
        "years": years,
        "stocks_fetched": len(result["stocks"]),
        "stocks_failed": failed_tickers,
        "total_requested": len(NIFTY_STOCKS),
        "nifty_days": len(result["nifty"]),
    }

    logger.info(
        f"Data ingestion complete: {len(result['stocks'])}/{len(NIFTY_STOCKS)} stocks fetched"
    )
    if failed_tickers:
        logger.warning(f"Failed tickers: {failed_tickers}")

    return result


def preprocess_data(raw_data: dict) -> dict:
    """
    Preprocess fetched data: align dates, forward-fill, remove nulls.

    Returns:
        Dictionary with:
            'close': DataFrame of aligned close prices (stocks as columns)
            'volume': DataFrame of aligned volumes
            'nifty_close': Series of Nifty close prices
            'vix_close': Series of VIX close prices
            'log_returns': DataFrame of log returns for all stocks
            'nifty_returns': Series of Nifty log returns
    """
    logger.info("Preprocessing and aligning data...")

    # Build close price and volume DataFrames
    close_dict = {}
    volume_dict = {}
    for ticker, df in raw_data["stocks"].items():
        close_dict[ticker] = df["Close"]
        volume_dict[ticker] = df["Volume"]

    close_df = pd.DataFrame(close_dict)
    volume_df = pd.DataFrame(volume_dict)
    nifty_close = raw_data["nifty"]["Close"].copy()
    vix_close = raw_data["vix"]["Close"].copy()

    # Align all data to common trading dates
    common_index = close_df.dropna(how="all").index
    common_index = common_index.intersection(nifty_close.dropna().index)

    close_df = close_df.loc[common_index]
    volume_df = volume_df.loc[common_index]
    nifty_close = nifty_close.loc[common_index]
    vix_close = vix_close.reindex(common_index)

    # Forward-fill then backward-fill small gaps
    close_df = close_df.ffill().bfill()
    volume_df = volume_df.ffill().bfill()
    nifty_close = nifty_close.ffill().bfill()
    vix_close = vix_close.ffill().bfill()

    # Drop any remaining columns with nulls
    null_cols = close_df.columns[close_df.isnull().any()].tolist()
    if null_cols:
        logger.warning(f"Dropping stocks with persistent nulls: {null_cols}")
        close_df = close_df.drop(columns=null_cols)
        volume_df = volume_df.drop(columns=null_cols)

    # Compute log returns
    log_returns = np.log(close_df / close_df.shift(1))
    nifty_returns = np.log(nifty_close / nifty_close.shift(1))

    # Drop first row (NaN from returns)
    log_returns = log_returns.iloc[1:]
    nifty_returns = nifty_returns.iloc[1:]
    close_df = close_df.iloc[1:]
    volume_df = volume_df.iloc[1:]
    nifty_close = nifty_close.iloc[1:]
    vix_close = vix_close.iloc[1:]

    logger.info(
        f"Preprocessing complete: {len(close_df)} days, {len(close_df.columns)} stocks"
    )

    return {
        "close": close_df,
        "volume": volume_df,
        "nifty_close": nifty_close,
        "vix_close": vix_close,
        "log_returns": log_returns,
        "nifty_returns": nifty_returns,
    }
