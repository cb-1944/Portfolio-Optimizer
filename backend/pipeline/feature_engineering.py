"""
Feature Engineering Module
Computes 9 carefully normalized features for each stock:
  1. Log Returns (raw — LSTM handles noise via 20-day sequences)
  2. RSI(14) — percentile rank normalized [-1, +1]
  3. MACD — price-normalized, z-scored, clipped [-1, +1]
  4. Momentum_20 — percentage return over 20 days
  5. Volatility_20 — inverted percentile rank [-1, +1]
  6. Rolling Correlation with Nifty — 20-day window [-1, +1]
  7. Volume Momentum — log ratio, z-scored, clipped [-1, +1]
  8. Event Decay Sentiment — shared across all stocks [-1, +1]
  9. Risk-Adjusted Nifty Trend — market regime signal
"""

import pandas as pd
import numpy as np
import logging
from .sentiment import compute_event_decay_sentiment

logger = logging.getLogger(__name__)


# ─── Helper Functions ────────────────────────────────────────────────────────

def _rolling_zscore(series: pd.Series, window: int = 252) -> pd.Series:
    """Z-score normalized over a rolling window, clipped to [-1, +1]."""
    mean = series.rolling(window, min_periods=60).mean()
    std = series.rolling(window, min_periods=60).std()
    z = (series - mean) / (std + 1e-10)
    return z.clip(-1.0, 1.0)


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI calculation."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi


# ─── Feature Computation Functions ───────────────────────────────────────────

def compute_feature_1_log_returns(log_returns: pd.Series) -> pd.Series:
    """
    Feature 1: Log Returns (raw).
    Daily log return is noisy but LSTM extracts patterns across the 20-day
    lookback window. Do NOT smooth — the noise IS information at sequence level.
    """
    return log_returns


def compute_feature_2_rsi(close: pd.Series) -> pd.Series:
    """
    Feature 2: RSI(14) — Percentile Rank Normalized.
    Option B: Percentile rank over rolling 252 days → [-1, +1]
    More robust to different market regimes than simple (RSI-50)/50.
    """
    rsi = _compute_rsi(close, period=14)
    # Percentile rank over rolling 252 days
    rsi_pct = rsi.rolling(252, min_periods=60).rank(pct=True)
    # Map [0, 1] → [-1, +1]
    rsi_norm = rsi_pct * 2 - 1
    return rsi_norm


def compute_feature_3_macd(close: pd.Series) -> pd.Series:
    """
    Feature 3: MACD — Price-Normalized, Z-Scored.
    Raw MACD is price-level dependent. Normalizing by price makes it a percentage,
    then z-scoring over 252 days makes it comparable across stocks and time.
    """
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    # Normalize by price level → percentage
    macd_norm = (ema12 - ema26) / (close + 1e-10)
    # Z-score over rolling 252 days, clipped [-1, +1]
    macd_final = _rolling_zscore(macd_norm, window=252)
    return macd_final


def compute_feature_4_momentum(close: pd.Series) -> pd.Series:
    """
    Feature 4: Momentum_20.
    (Close_t / Close_t-20) - 1 — already a percentage return, comparable across stocks.
    """
    momentum = (close / close.shift(20)) - 1
    return momentum


def compute_feature_5_volatility(log_returns: pd.Series) -> pd.Series:
    """
    Feature 5: Volatility_20 — Inverted Percentile Rank.
    Step 1: 20-day rolling std → raw volatility
    Step 2: Percentile rank over 252 days → [0, 1]
    Step 3: Invert (high vol = low score) → [0, 1]
    Step 4: Center around zero → [-1, +1]
    High vol = negative = risk signal, Low vol = positive = stability signal.
    """
    vol_20 = log_returns.rolling(20, min_periods=10).std()
    vol_pct = vol_20.rolling(252, min_periods=60).rank(pct=True)
    vol_score = 1 - vol_pct  # Invert
    vol_final = vol_score * 2 - 1  # Center
    return vol_final


def compute_feature_6_nifty_correlation(
    stock_returns: pd.Series, nifty_returns: pd.Series
) -> pd.Series:
    """
    Feature 6: 20-day Rolling Correlation with Nifty Index.
    Already in [-1, +1] range — no further normalization needed.
    """
    # Align indices
    aligned_nifty = nifty_returns.reindex(stock_returns.index)
    corr = stock_returns.rolling(20, min_periods=10).corr(aligned_nifty)
    return corr


def compute_feature_7_volume_momentum(volume: pd.Series) -> pd.Series:
    """
    Feature 7: Volume Momentum — Log Ratio, Z-Scored.
    Step 1: Log transform to compress spikes
    Step 2: Z-score over rolling 252 days, clipped [-1, +1]
    +1 = volume dramatically above history
    -1 = volume dramatically below history
     0 = volume at historical average
    """
    avg_vol = volume.rolling(20, min_periods=10).mean()
    vol_ratio = np.log(volume / (avg_vol + 1e-10) + 1e-10)
    vol_mom_final = _rolling_zscore(vol_ratio, window=252)
    return vol_mom_final


def compute_feature_9_risk_adjusted_trend(
    nifty_close: pd.Series, vix_close: pd.Series
) -> pd.Series:
    """
    Feature 9: Risk-Adjusted Nifty Index Trend.
    TrendStrength = (EMA20 - EMA50) / EMA50
    Momentum_20 = (Close_t / Close_t-20) - 1
    RiskAdjusted = TrendStrength * (Momentum_20 / VIX)

    This is the market regime signal — shared across all stocks.
    """
    ema20 = nifty_close.ewm(span=20, adjust=False).mean()
    ema50 = nifty_close.ewm(span=50, adjust=False).mean()
    trend_strength = (ema20 - ema50) / (ema50 + 1e-10)

    momentum_20 = (nifty_close / nifty_close.shift(20)) - 1

    # VIX is in percentage points — normalize to decimal
    vix_decimal = vix_close / 100.0
    vix_decimal = vix_decimal.clip(lower=0.05)  # Floor at 5% to avoid division explosion

    risk_adj = trend_strength * (momentum_20 / vix_decimal)

    # Z-score and clip for consistency
    risk_adj_final = _rolling_zscore(risk_adj, window=252)
    return risk_adj_final


# ─── Main Feature Pipeline ──────────────────────────────────────────────────

def compute_all_features(preprocessed_data: dict, progress_callback=None) -> dict:
    """
    Compute all 9 features for every stock.

    Args:
        preprocessed_data: Output from data_ingestion.preprocess_data()
        progress_callback: Optional callable(step_name, progress_pct) for UI updates

    Returns:
        Dictionary with:
            'features': {ticker: DataFrame with 9 feature columns}
            'target': {ticker: Series of next-day log returns (prediction target)}
            'dates': DatetimeIndex of valid dates
            'tickers': List of tickers with valid features
    """
    close = preprocessed_data["close"]
    volume = preprocessed_data["volume"]
    log_returns = preprocessed_data["log_returns"]
    nifty_close = preprocessed_data["nifty_close"]
    nifty_returns = preprocessed_data["nifty_returns"]
    vix_close = preprocessed_data["vix_close"]

    tickers = close.columns.tolist()
    n_stocks = len(tickers)

    logger.info(f"Computing 9 features for {n_stocks} stocks...")

    # --- Market-wide features (computed once) ---
    logger.info("  Computing market-wide features...")
    event_sentiment = compute_event_decay_sentiment(close.index)
    risk_adj_trend = compute_feature_9_risk_adjusted_trend(nifty_close, vix_close)

    features_dict = {}
    target_dict = {}

    for i, ticker in enumerate(tickers):
        logger.info(f"  [{i+1}/{n_stocks}] Computing features for {ticker}...")
        if progress_callback:
            progress_callback("feature_engineering", 20 + int(40 * (i / n_stocks)))

        stock_close = close[ticker]
        stock_volume = volume[ticker]
        stock_returns = log_returns[ticker]

        # Build feature DataFrame
        feat_df = pd.DataFrame(index=close.index)

        feat_df["log_return"] = compute_feature_1_log_returns(stock_returns)
        feat_df["rsi_norm"] = compute_feature_2_rsi(stock_close)
        feat_df["macd_norm"] = compute_feature_3_macd(stock_close)
        feat_df["momentum_20"] = compute_feature_4_momentum(stock_close)
        feat_df["volatility_20"] = compute_feature_5_volatility(stock_returns)
        feat_df["nifty_corr"] = compute_feature_6_nifty_correlation(
            stock_returns, nifty_returns
        )
        feat_df["volume_momentum"] = compute_feature_7_volume_momentum(stock_volume)
        feat_df["event_sentiment"] = event_sentiment
        feat_df["risk_adj_trend"] = risk_adj_trend

        # Target: 20-day forward log return — aligned with rebalancing frequency
        # Daily returns (shift(-1)) have SNR~0.05 → essentially random
        # 20-day returns have SNR~0.15-0.25 → actually learnable
        target = np.log(stock_close.shift(-20) / stock_close)

        # Drop rows with any NaN in features or target
        valid_mask = feat_df.notna().all(axis=1) & target.notna()
        feat_df = feat_df[valid_mask]
        target = target[valid_mask]

        if len(feat_df) < 300:
            logger.warning(f"  {ticker}: Only {len(feat_df)} valid rows, skipping")
            continue

        features_dict[ticker] = feat_df
        target_dict[ticker] = target
        logger.info(f"  {ticker}: {len(feat_df)} valid feature rows")

    # Find common date range across all stocks
    if not features_dict:
        raise ValueError("No stocks have sufficient feature data")

    common_dates = None
    for ticker, feat_df in features_dict.items():
        if common_dates is None:
            common_dates = set(feat_df.index)
        else:
            common_dates = common_dates.intersection(set(feat_df.index))

    common_dates = sorted(common_dates)
    logger.info(f"  Common date range: {common_dates[0]} to {common_dates[-1]}")
    logger.info(f"  Total common trading days: {len(common_dates)}")

    # Align all features to common dates
    for ticker in list(features_dict.keys()):
        features_dict[ticker] = features_dict[ticker].loc[common_dates]
        target_dict[ticker] = target_dict[ticker].loc[common_dates]

    logger.info(
        f"Feature engineering complete: {len(features_dict)} stocks, "
        f"{len(common_dates)} days, 9 features each"
    )

    return {
        "features": features_dict,
        "target": target_dict,
        "dates": pd.DatetimeIndex(common_dates),
        "tickers": list(features_dict.keys()),
    }
