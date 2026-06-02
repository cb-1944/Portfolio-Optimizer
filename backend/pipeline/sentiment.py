"""
Sentiment Module — Event Decay Sentiment (Feature 8)
Computes exponentially decaying sentiment signals from historical market events.
Uses dummy events with REAL dates and contextually accurate headlines.
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Dummy events with REAL dates and contextually accurate information
# S0 = initial sentiment score (like FinBERT output), lambda = decay rate
DUMMY_EVENTS = [
    # --- Demonetization ---
    {"date": "2016-11-08", "headline": "India announces demonetization of 500 and 1000 rupee notes",
     "sentiment": -0.80, "decay_rate": 0.020, "category": "policy"},

    # --- GST Implementation ---
    {"date": "2017-07-01", "headline": "GST rolled out across India replacing complex indirect tax structure",
     "sentiment": -0.30, "decay_rate": 0.030, "category": "policy"},

    # --- IL&FS Crisis ---
    {"date": "2018-09-21", "headline": "IL&FS defaults on debt obligations triggering NBFC liquidity crisis",
     "sentiment": -0.70, "decay_rate": 0.025, "category": "crisis"},

    # --- 2019 General Elections ---
    {"date": "2019-05-23", "headline": "BJP wins landslide victory in 2019 general elections with 303 seats",
     "sentiment": 0.85, "decay_rate": 0.040, "category": "election"},

    # --- Corporate Tax Cut ---
    {"date": "2019-09-20", "headline": "Government slashes corporate tax rate to 22% to boost economy",
     "sentiment": 0.75, "decay_rate": 0.035, "category": "policy"},

    # --- COVID-19 Pandemic ---
    {"date": "2020-01-30", "headline": "India confirms first COVID-19 case as WHO declares global emergency",
     "sentiment": -0.60, "decay_rate": 0.030, "category": "pandemic"},
    {"date": "2020-03-11", "headline": "WHO declares COVID-19 a global pandemic markets crash worldwide",
     "sentiment": -0.95, "decay_rate": 0.015, "category": "pandemic"},
    {"date": "2020-03-24", "headline": "India announces 21-day nationwide lockdown to combat COVID spread",
     "sentiment": -0.90, "decay_rate": 0.020, "category": "pandemic"},

    # --- RBI Emergency Rate Cut ---
    {"date": "2020-03-27", "headline": "RBI cuts repo rate by 75 bps to 4.4% announces moratorium on loans",
     "sentiment": 0.60, "decay_rate": 0.050, "category": "monetary"},

    # --- Stimulus Package ---
    {"date": "2020-05-12", "headline": "PM announces 20 lakh crore Atmanirbhar Bharat stimulus package",
     "sentiment": 0.65, "decay_rate": 0.040, "category": "policy"},

    # --- Vaccine Rally ---
    {"date": "2020-11-09", "headline": "Pfizer announces 90% vaccine efficacy global markets surge",
     "sentiment": 0.80, "decay_rate": 0.035, "category": "pandemic"},

    # --- Budget 2021 ---
    {"date": "2021-02-01", "headline": "Union Budget 2021 increases capex by 34% boosts infra spending",
     "sentiment": 0.70, "decay_rate": 0.045, "category": "budget"},

    # --- Second Wave ---
    {"date": "2021-04-15", "headline": "India reports record COVID cases as devastating second wave hits",
     "sentiment": -0.70, "decay_rate": 0.025, "category": "pandemic"},

    # --- Russia-Ukraine Conflict ---
    {"date": "2022-02-24", "headline": "Russia invades Ukraine crude oil spikes global risk-off sentiment",
     "sentiment": -0.75, "decay_rate": 0.020, "category": "geopolitical"},

    # --- RBI Surprise Rate Hike ---
    {"date": "2022-05-04", "headline": "RBI surprises market with 40 bps repo rate hike to tame inflation",
     "sentiment": -0.50, "decay_rate": 0.055, "category": "monetary"},

    # --- Adani Short-Seller Report ---
    {"date": "2023-01-25", "headline": "Hindenburg Research publishes short-seller report on Adani Group",
     "sentiment": -0.65, "decay_rate": 0.035, "category": "corporate"},

    # --- SVB Collapse ---
    {"date": "2023-03-10", "headline": "Silicon Valley Bank collapses sparking global banking sector fears",
     "sentiment": -0.55, "decay_rate": 0.060, "category": "crisis"},

    # --- Budget 2023 ---
    {"date": "2023-02-01", "headline": "Union Budget 2023 focuses on green growth and capital expenditure",
     "sentiment": 0.55, "decay_rate": 0.050, "category": "budget"},

    # --- 2024 General Elections ---
    {"date": "2024-06-04", "headline": "NDA wins 2024 elections with reduced majority coalition uncertainty",
     "sentiment": 0.25, "decay_rate": 0.050, "category": "election"},

    # --- Budget 2024 ---
    {"date": "2024-07-23", "headline": "Union Budget 2024 raises LTCG tax removes indexation benefit",
     "sentiment": -0.45, "decay_rate": 0.055, "category": "budget"},
]


def compute_event_decay_sentiment(dates_index: pd.DatetimeIndex) -> pd.Series:
    """
    Compute the aggregate event decay sentiment for each trading day.

    Formula: S(t) = sum over all events of S0_i * exp(-lambda_i * t_i)
    where t_i = max(0, days since event i occurred)

    For each day, we sum the decaying sentiment from ALL past events.
    Future events contribute 0.

    Args:
        dates_index: DatetimeIndex of trading days

    Returns:
        pd.Series with event decay sentiment values, clipped to [-1, +1]
    """
    logger.info(f"Computing event decay sentiment for {len(DUMMY_EVENTS)} events...")

    sentiment_series = pd.Series(0.0, index=dates_index, dtype=float)

    for event in DUMMY_EVENTS:
        event_date = pd.Timestamp(event["date"])
        s0 = event["sentiment"]
        decay = event["decay_rate"]

        for i, date in enumerate(dates_index):
            days_since = (date - event_date).days
            if days_since >= 0:
                # Event has occurred — apply exponential decay
                contribution = s0 * np.exp(-decay * days_since)
                sentiment_series.iloc[i] += contribution

    # Clip to [-1, +1]
    sentiment_series = sentiment_series.clip(-1.0, 1.0)

    logger.info("  Event decay sentiment computed successfully")
    return sentiment_series
