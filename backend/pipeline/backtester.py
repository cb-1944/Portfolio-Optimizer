"""
Backtesting Engine
Walk-forward backtest with 20-day rebalancing.
Produces detailed logs, performance metrics, and comparison with Nifty index.
"""

import numpy as np
import pandas as pd
import logging
from .portfolio_optimizer import build_portfolio

logger = logging.getLogger(__name__)


def run_backtest(
    predicted_returns: dict,
    log_returns_df: pd.DataFrame,
    nifty_returns: pd.Series,
    test_dates: pd.DatetimeIndex,
    rebalance_freq: int = 20,
    top_n: int = 6,
    ewma_span: int = 60,
    features_dict: dict = None,
    progress_callback=None,
) -> dict:
    """
    Execute walk-forward backtest with periodic rebalancing.

    At every rebalance_freq trading days:
    1. Use LSTM predictions to rank stocks
    2. Select top N stocks
    3. Optimize portfolio weights (Markowitz + EWMA covariance)
    4. Hold for rebalance_freq days
    5. Record actual returns

    Args:
        predicted_returns: {ticker: Series} of LSTM predictions
        log_returns_df: DataFrame of actual log returns
        nifty_returns: Series of Nifty log returns
        test_dates: DatetimeIndex of the test/backtest period
        rebalance_freq: Days between rebalances (20)
        top_n: Number of stocks per portfolio (6)
        ewma_span: EWMA covariance span (60)
        progress_callback: Optional progress updater

    Returns:
        Dictionary with full backtest results
    """
    logger.info(
        f"Starting backtest: {test_dates[0].date()} to {test_dates[-1].date()}"
    )
    logger.info(
        f"  Rebalance every {rebalance_freq} days, top {top_n} stocks, "
        f"EWMA span {ewma_span}"
    )

    # Results tracking
    portfolio_returns = []
    nifty_daily_returns = []
    rebalance_logs = []
    all_daily_logs = []

    current_portfolio = None
    days_since_rebalance = rebalance_freq  # Force rebalance on first day

    n_dates = len(test_dates)

    for i, date in enumerate(test_dates):
        if progress_callback and i % 10 == 0:
            progress_callback("backtesting", 75 + int(20 * (i / n_dates)))

        # --- Check if rebalancing is needed ---
        if days_since_rebalance >= rebalance_freq:
            new_portfolio = build_portfolio(
                predicted_returns, log_returns_df, date,
                top_n=top_n, ewma_span=ewma_span,
                features_dict=features_dict,
            )

            if new_portfolio is not None:
                # Log the rebalancing event
                rebalance_log = {
                    "rebalance_number": len(rebalance_logs) + 1,
                    "date": date.strftime("%Y-%m-%d"),
                    "stocks_selected": new_portfolio["tickers"],
                    "weights": [round(w, 4) for w in new_portfolio["weights"]],
                    "predicted_returns": [
                        round(r, 6) for r in new_portfolio["predicted_returns"]
                    ],
                    "expected_sharpe": round(new_portfolio["expected_sharpe"], 4),
                    "reason": "Scheduled 20-day rebalance",
                }

                if current_portfolio is not None:
                    # Log what changed
                    old_set = set(current_portfolio["tickers"])
                    new_set = set(new_portfolio["tickers"])
                    added = new_set - old_set
                    removed = old_set - new_set
                    rebalance_log["stocks_added"] = list(added)
                    rebalance_log["stocks_removed"] = list(removed)
                    rebalance_log["turnover"] = len(added) + len(removed)
                else:
                    rebalance_log["stocks_added"] = new_portfolio["tickers"]
                    rebalance_log["stocks_removed"] = []
                    rebalance_log["turnover"] = len(new_portfolio["tickers"])

                rebalance_logs.append(rebalance_log)
                current_portfolio = new_portfolio
                days_since_rebalance = 0

                logger.info(
                    f"  Rebalance #{len(rebalance_logs)} on {date.date()}: "
                    f"{new_portfolio['tickers']} | "
                    f"Weights: {[round(w,2) for w in new_portfolio['weights']]}"
                )

        # --- Compute daily portfolio return ---
        if current_portfolio is not None:
            tickers = current_portfolio["tickers"]
            weights = np.array(current_portfolio["weights"])

            daily_returns = []
            for ticker in tickers:
                if date in log_returns_df.index and ticker in log_returns_df.columns:
                    daily_returns.append(log_returns_df.loc[date, ticker])
                else:
                    daily_returns.append(0.0)

            daily_returns = np.array(daily_returns)
            port_daily_return = np.dot(weights, daily_returns)
        else:
            port_daily_return = 0.0

        # Nifty return for comparison
        if date in nifty_returns.index:
            nifty_ret = nifty_returns.loc[date]
        else:
            nifty_ret = 0.0

        portfolio_returns.append(port_daily_return)
        nifty_daily_returns.append(nifty_ret)

        # Daily log
        daily_log = {
            "date": date.strftime("%Y-%m-%d"),
            "portfolio_return": round(float(port_daily_return), 6),
            "nifty_return": round(float(nifty_ret), 6),
            "holdings": (
                current_portfolio["tickers"] if current_portfolio else []
            ),
            "is_rebalance_day": days_since_rebalance == 0,
        }
        all_daily_logs.append(daily_log)
        days_since_rebalance += 1

    # --- Compute Performance Metrics ---
    port_returns = np.array(portfolio_returns)
    nifty_rets = np.array(nifty_daily_returns)

    metrics = compute_performance_metrics(port_returns, nifty_rets)

    # --- Build Equity Curves ---
    port_cumulative = np.cumprod(1 + port_returns)
    nifty_cumulative = np.cumprod(1 + nifty_rets)

    equity_curve = {
        "dates": [d.strftime("%Y-%m-%d") for d in test_dates],
        "portfolio": port_cumulative.tolist(),
        "nifty": nifty_cumulative.tolist(),
    }

    logger.info("=" * 60)
    logger.info("BACKTEST RESULTS")
    logger.info("=" * 60)
    for key, val in metrics.items():
        logger.info(f"  {key}: {val}")
    logger.info("=" * 60)

    return {
        "metrics": metrics,
        "equity_curve": equity_curve,
        "rebalance_logs": rebalance_logs,
        "daily_logs": all_daily_logs,
        "n_rebalances": len(rebalance_logs),
        "test_period": {
            "start": test_dates[0].strftime("%Y-%m-%d"),
            "end": test_dates[-1].strftime("%Y-%m-%d"),
            "trading_days": len(test_dates),
        },
    }


def compute_performance_metrics(
    portfolio_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    risk_free_annual: float = 0.065,
) -> dict:
    """
    Compute comprehensive performance metrics and comparison.
    """
    trading_days = 252
    daily_rf = risk_free_annual / trading_days

    # --- Portfolio Metrics ---
    port_total = float(np.prod(1 + portfolio_returns) - 1)
    n_years = len(portfolio_returns) / trading_days
    port_annual = float((1 + port_total) ** (1 / max(n_years, 0.01)) - 1)
    port_vol = float(np.std(portfolio_returns) * np.sqrt(trading_days))
    port_sharpe = float(
        (port_annual - risk_free_annual) / (port_vol + 1e-10)
    )

    # Sortino (downside deviation)
    downside = portfolio_returns[portfolio_returns < 0]
    downside_std = float(np.std(downside) * np.sqrt(trading_days)) if len(downside) > 0 else 1e-10
    port_sortino = float((port_annual - risk_free_annual) / downside_std)

    # Max Drawdown
    cum_returns = np.cumprod(1 + portfolio_returns)
    rolling_max = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - rolling_max) / rolling_max
    max_drawdown = float(np.min(drawdowns))

    # Calmar Ratio
    calmar = float(port_annual / abs(max_drawdown)) if max_drawdown != 0 else 0

    # Win Rate
    win_rate = float(np.mean(portfolio_returns > 0) * 100)

    # --- Benchmark (Nifty) Metrics ---
    bench_total = float(np.prod(1 + benchmark_returns) - 1)
    bench_annual = float((1 + bench_total) ** (1 / max(n_years, 0.01)) - 1)
    bench_vol = float(np.std(benchmark_returns) * np.sqrt(trading_days))
    bench_sharpe = float(
        (bench_annual - risk_free_annual) / (bench_vol + 1e-10)
    )
    bench_cum = np.cumprod(1 + benchmark_returns)
    bench_rolling_max = np.maximum.accumulate(bench_cum)
    bench_dd = (bench_cum - bench_rolling_max) / bench_rolling_max
    bench_max_dd = float(np.min(bench_dd))

    # --- Alpha & Beta ---
    if np.std(benchmark_returns) > 0:
        beta = float(
            np.cov(portfolio_returns, benchmark_returns)[0, 1]
            / np.var(benchmark_returns)
        )
    else:
        beta = 1.0
    alpha_annual = float(port_annual - (risk_free_annual + beta * (bench_annual - risk_free_annual)))

    # Information Ratio
    excess = portfolio_returns - benchmark_returns
    tracking_error = float(np.std(excess) * np.sqrt(trading_days))
    info_ratio = float(np.mean(excess) * trading_days / (tracking_error + 1e-10))

    return {
        # Portfolio
        "portfolio_total_return": round(port_total * 100, 2),
        "portfolio_annual_return": round(port_annual * 100, 2),
        "portfolio_annual_volatility": round(port_vol * 100, 2),
        "portfolio_sharpe_ratio": round(port_sharpe, 3),
        "portfolio_sortino_ratio": round(port_sortino, 3),
        "portfolio_max_drawdown": round(max_drawdown * 100, 2),
        "portfolio_calmar_ratio": round(calmar, 3),
        "portfolio_win_rate": round(win_rate, 1),
        # Benchmark
        "nifty_total_return": round(bench_total * 100, 2),
        "nifty_annual_return": round(bench_annual * 100, 2),
        "nifty_annual_volatility": round(bench_vol * 100, 2),
        "nifty_sharpe_ratio": round(bench_sharpe, 3),
        "nifty_max_drawdown": round(bench_max_dd * 100, 2),
        # Comparison
        "alpha": round(alpha_annual * 100, 2),
        "beta": round(beta, 3),
        "information_ratio": round(info_ratio, 3),
        "excess_return": round((port_total - bench_total) * 100, 2),
        "backtest_years": round(n_years, 2),
    }
