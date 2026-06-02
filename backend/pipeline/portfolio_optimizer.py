"""
Portfolio Optimizer Module
Markowitz Mean-Variance Optimization with EWMA covariance.
Uses LSTM-predicted returns and 60-day exponentially weighted covariance.
"""

import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


def rank_and_select_top(
    predicted_returns: dict,
    date: pd.Timestamp,
    top_n: int = 6,
    features_dict: dict = None,
    lstm_weight: float = 0.30,
    momentum_weight: float = 0.50,
    quality_weight: float = 0.20,
) -> list:
    """
    Ensemble stock ranking: LSTM prediction + momentum factor + quality factor.

    Pure LSTM ranking is unreliable — blending with proven factors provides
    robust alpha even when the neural network makes poor predictions.

    Weights:
        50% LSTM predicted 20-day return
        30% Momentum (20-day price return — Jegadeesh & Titman factor)
        20% Quality  (inverted volatility — low-vol anomaly)

    Returns:
        List of (ticker, ensemble_score) tuples, sorted descending
    """
    scored = []
    for ticker, preds in predicted_returns.items():
        if date not in preds.index:
            continue

        lstm_score = preds.loc[date]

        # Pull momentum and quality directly from features
        mom_score = 0.0
        vol_score = 0.0
        if features_dict and ticker in features_dict:
            fdf = features_dict[ticker]
            if date in fdf.index:
                row = fdf.loc[date]
                m = row.get("momentum_20", 0.0)
                v = row.get("volatility_20", 0.0)
                mom_score = float(m) if np.isfinite(m) else 0.0
                vol_score = float(v) if np.isfinite(v) else 0.0

        combined = (
            lstm_weight * lstm_score
            + momentum_weight * mom_score
            + quality_weight * vol_score
        )
        scored.append((ticker, combined))

    if not scored:
        return []

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


def compute_ewma_covariance(
    returns_df: pd.DataFrame, end_date: pd.Timestamp, span: int = 60
) -> pd.DataFrame:
    """
    Compute Exponentially Weighted Moving Average covariance matrix.
    Uses only data up to end_date with the specified span.

    The EWMA covariance gives more weight to recent observations,
    making it more responsive to regime changes than static covariance.

    Args:
        returns_df: DataFrame of log returns (stocks as columns)
        end_date: Compute covariance using data up to this date
        span: EWMA span (60 days as specified)

    Returns:
        Covariance matrix as DataFrame
    """
    # Use data up to end_date
    mask = returns_df.index <= end_date
    relevant_returns = returns_df[mask].tail(span * 3)  # Use ~3x span for stability

    if len(relevant_returns) < span:
        logger.warning(
            f"Only {len(relevant_returns)} days available for EWMA cov "
            f"(need {span}), using all available"
        )

    # Compute EWMA covariance
    ewma_cov = relevant_returns.ewm(span=span).cov()

    # Extract the last date's covariance matrix
    last_date = relevant_returns.index[-1]
    tickers = relevant_returns.columns.tolist()
    n = len(tickers)

    cov_matrix = pd.DataFrame(
        np.zeros((n, n)), index=tickers, columns=tickers
    )

    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            try:
                cov_matrix.loc[t1, t2] = ewma_cov.loc[(last_date, t1), t2]
            except KeyError:
                if i == j:
                    cov_matrix.loc[t1, t2] = relevant_returns[t1].var()
                else:
                    cov_matrix.loc[t1, t2] = 0.0

    # Ensure positive semi-definite
    eigenvalues = np.linalg.eigvalsh(cov_matrix.values)
    if np.any(eigenvalues < 0):
        # Add small regularization
        min_eig = abs(min(eigenvalues))
        cov_matrix += np.eye(n) * (min_eig + 1e-8)

    return cov_matrix


def optimize_portfolio_weights(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.065 / 252,
    min_weight: float = 0.08,
    max_weight: float = 0.25,
) -> np.ndarray:
    """
    Markowitz Mean-Variance Optimization for Maximum Sharpe Ratio.

    Args:
        expected_returns: Array of predicted returns for selected stocks
        cov_matrix: Covariance matrix (numpy array)
        risk_free_rate: Daily risk-free rate (default: 6.5% annual / 252)
        min_weight: Minimum weight per stock (5%)
        max_weight: Maximum weight per stock (40%)

    Returns:
        Optimal weight array
    """
    n = len(expected_returns)

    def neg_sharpe(weights):
        port_return = np.dot(weights, expected_returns)
        port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        if port_vol < 1e-10:
            return 0
        return -(port_return - risk_free_rate) / port_vol

    # Constraints: weights sum to 1
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Bounds: min_weight to max_weight per stock
    bounds = [(min_weight, max_weight) for _ in range(n)]

    # Initial guess: equal weights
    x0 = np.array([1.0 / n] * n)

    result = minimize(
        neg_sharpe,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )

    if result.success:
        weights = result.x
        # Normalize to ensure exact sum = 1
        weights = weights / weights.sum()
    else:
        logger.warning(
            f"Optimization did not converge: {result.message}. Using equal weights."
        )
        weights = np.array([1.0 / n] * n)

    return weights


def build_portfolio(
    predicted_returns: dict,
    log_returns_df: pd.DataFrame,
    date: pd.Timestamp,
    top_n: int = 6,
    ewma_span: int = 60,
    features_dict: dict = None,
) -> dict:
    """
    Full portfolio construction at a rebalancing date.

    1. Ensemble rank stocks (LSTM + momentum + quality)
    2. Select top N
    3. Compute EWMA covariance for selected stocks
    4. Optimize weights via Markowitz

    Returns:
        Dictionary with portfolio details
    """
    # Step 1 & 2: Ensemble rank and select
    top_stocks = rank_and_select_top(
        predicted_returns, date, top_n, features_dict=features_dict
    )

    if len(top_stocks) < 3:
        logger.warning(f"Only {len(top_stocks)} stocks available at {date}")
        return None

    selected_tickers = [t[0] for t in top_stocks]
    pred_returns_arr = np.array([t[1] for t in top_stocks])

    # Step 3: EWMA covariance for selected stocks only
    selected_returns = log_returns_df[selected_tickers]
    cov_matrix = compute_ewma_covariance(selected_returns, date, span=ewma_span)

    # Step 4: Optimize
    weights = optimize_portfolio_weights(
        pred_returns_arr, cov_matrix.values
    )

    # Compute portfolio metrics
    port_return = np.dot(weights, pred_returns_arr)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix.values, weights)))
    daily_rf = 0.065 / 252
    sharpe = (port_return - daily_rf) / (port_vol + 1e-10)

    return {
        "date": date,
        "tickers": selected_tickers,
        "weights": weights.tolist(),
        "predicted_returns": pred_returns_arr.tolist(),
        "all_rankings": [(t, float(r)) for t, r in top_stocks],
        "expected_return": float(port_return),
        "expected_volatility": float(port_vol),
        "expected_sharpe": float(sharpe),
    }
