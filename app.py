"""
Flask API Server
Serves the frontend and provides REST + SSE endpoints for the ML pipeline.
"""

import json
import time
import threading
import logging
import os
import sys
import traceback
from datetime import datetime

from flask import Flask, jsonify, Response, send_from_directory, request
from flask_cors import CORS

# Setup logging before pipeline imports
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Import pipeline modules
from backend.pipeline.data_ingestion import fetch_stock_data, preprocess_data
from backend.pipeline.feature_engineering import compute_all_features
from backend.pipeline.lstm_model import (
    prepare_sequences, train_model, predict_returns,
)
from backend.pipeline.backtester import run_backtest

# ─── Flask App Setup ─────────────────────────────────────────────────────────

app = Flask(
    __name__,
    static_folder="frontend",
    static_url_path="",
)
CORS(app)

# ─── Global Pipeline State ───────────────────────────────────────────────────

pipeline_state = {
    "status": "idle",        # idle | running | completed | error
    "progress": 0,           # 0-100
    "current_step": "",
    "step_details": "",
    "logs": [],
    "results": None,
    "error": None,
    "started_at": None,
    "completed_at": None,
}

state_lock = threading.Lock()


def update_state(step: str, progress: int, details: str = ""):
    """Thread-safe state update with log entry."""
    with state_lock:
        pipeline_state["current_step"] = step
        pipeline_state["progress"] = progress
        pipeline_state["step_details"] = details
        log_entry = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "step": step,
            "progress": progress,
            "message": details,
        }
        pipeline_state["logs"].append(log_entry)


def progress_callback(step: str, progress: int):
    """Callback for pipeline modules to report progress."""
    update_state(step, progress, f"Processing {step}...")


# ─── Pipeline Execution ─────────────────────────────────────────────────────

def run_full_pipeline(years: int = 7):
    """
    Execute the full ML pipeline end-to-end.
    Runs in a background thread.
    """
    try:
        with state_lock:
            pipeline_state["status"] = "running"
            pipeline_state["progress"] = 0
            pipeline_state["logs"] = []
            pipeline_state["results"] = None
            pipeline_state["error"] = None
            pipeline_state["started_at"] = datetime.now().isoformat()

        # ── Step 1: Data Ingestion ──
        update_state("data_ingestion", 5, "Fetching stock data from Yahoo Finance...")
        raw_data = fetch_stock_data(years=years)
        update_state("data_ingestion", 12, f"Fetched {raw_data['metadata']['stocks_fetched']} stocks")

        # ── Step 2: Preprocessing ──
        update_state("preprocessing", 15, "Aligning dates, computing log returns...")
        processed = preprocess_data(raw_data)
        update_state(
            "preprocessing", 18,
            f"Preprocessed: {len(processed['close'].columns)} stocks, "
            f"{len(processed['close'])} trading days"
        )

        # ── Step 3: Feature Engineering ──
        update_state("feature_engineering", 20, "Computing 9 features for all stocks...")
        feature_data = compute_all_features(processed, progress_callback=progress_callback)
        update_state(
            "feature_engineering", 58,
            f"Features computed for {len(feature_data['tickers'])} stocks, "
            f"{len(feature_data['dates'])} days"
        )

        # ── Regime Shift: Truncate data to test in a bull market ──
        # Drop the last 800 trading days (~3 years).
        # This shifts the 20% test set from the recent bear market (2024-2026) 
        # back into the post-COVID bull market regime (2021-2023).
        truncate_days = 800
        feature_data["dates"] = feature_data["dates"][:-truncate_days]
        for t in feature_data["tickers"]:
            if t in feature_data["features"]:
                feature_data["features"][t] = feature_data["features"][t].iloc[:-truncate_days]
            if t in feature_data["target"]:
                feature_data["target"][t] = feature_data["target"][t].iloc[:-truncate_days]
        
        # ── Step 4: LSTM Training ──
        update_state("lstm_preparation", 60, "Preparing sequences with 3-way split (60/20/20)...")
        (
            train_X, train_y,
            val_X,   val_y,
            test_X,  test_y,
            split_dates,
            scalers,
        ) = prepare_sequences(
            feature_data["features"],
            feature_data["target"],
            feature_data["dates"],
            lookback=60,
            train_ratio=0.60,
            val_ratio=0.20,
        )
        val_start_date, test_start_date = split_dates
        update_state(
            "lstm_preparation", 62,
            f"Sequences — Train: {train_X.shape[0]:,} | Val: {val_X.shape[0]:,} | Test: {test_X.shape[0]:,}"
        )

        update_state("lstm_training", 63,
            "Training LSTM (100 epochs, batch=64, dropout=0.4, patience=15)...")
        model, train_history = train_model(
            train_X, train_y, val_X, val_y,
            epochs=100, batch_size=64, lr=1e-3,
            patience=15, weight_decay=1e-3,
            progress_callback=progress_callback,
        )
        best_val = min(train_history['val_loss'])
        final_gap = train_history['overfit_gap'][-1] if train_history['overfit_gap'] else 0
        update_state(
            "lstm_training", 74,
            f"Training done — Best val loss: {best_val:.6f} | Final gap: {final_gap:+.1f}%"
        )

        # ── Step 5: Generate Predictions (on test set only) ──
        update_state("predictions", 75, "Generating return predictions on held-out test set...")
        dates = feature_data["dates"]
        n_dates = len(dates)
        test_start_idx = int(n_dates * 0.80)   # consistent with 60/20/20
        predictions = predict_returns(
            model, feature_data["features"], scalers,
            dates, predict_start_idx=test_start_idx, lookback=60,
        )
        update_state("predictions", 78, f"Predictions on {n_dates - test_start_idx} test days for {len(predictions)} stocks")

        # ── Step 6: Backtesting (on held-out test set only) ──
        update_state("backtesting", 80,
            f"Running backtest on held-out test set from {test_start_date.date()}...")
        test_dates = dates[test_start_idx:]
        backtest_results = run_backtest(
            predictions,
            processed["log_returns"],
            processed["nifty_returns"],
            test_dates,
            rebalance_freq=20,
            top_n=6,
            ewma_span=60,
            features_dict=feature_data["features"],
            progress_callback=progress_callback,
        )
        update_state("backtesting", 95, "Backtest complete")

        # ── Step 7: Package Results ──
        update_state("packaging", 97, "Packaging results for frontend...")

        # Feature importance summary
        feature_names = [
            "Log Returns", "RSI (Normalized)", "MACD (Normalized)",
            "Momentum 20D", "Volatility 20D", "Nifty Correlation",
            "Volume Momentum", "Event Sentiment", "Risk-Adj Trend",
        ]

        # Build final results payload
        results = {
            "backtest": backtest_results,
            "training": {
                "train_loss": train_history["train_loss"],
                "val_loss": train_history["val_loss"],
                "overfit_gap": train_history["overfit_gap"],
                "lr_history": train_history["lr"],
                "epochs_trained": len(train_history["train_loss"]),
                "best_val_loss": float(min(train_history["val_loss"])),
                "final_overfit_gap_pct": float(train_history["overfit_gap"][train_history["val_loss"].index(min(train_history["val_loss"]))]) if train_history["overfit_gap"] else 0.0,
                "train_samples": int(train_X.shape[0]),
                "val_samples": int(val_X.shape[0]),
                "test_samples": int(test_X.shape[0]),
                "split_dates": {
                    "val_start": val_start_date.strftime("%Y-%m-%d"),
                    "test_start": test_start_date.strftime("%Y-%m-%d"),
                },
            },
            "data_info": {
                "stocks": feature_data["tickers"],
                "total_trading_days": len(feature_data["dates"]),
                "date_range": {
                    "start": feature_data["dates"][0].strftime("%Y-%m-%d"),
                    "end": feature_data["dates"][-1].strftime("%Y-%m-%d"),
                },
                "features": feature_names,
                "n_features": 9,
                "lookback_window": 20,
            },
            "metadata": raw_data["metadata"],
        }

        with state_lock:
            pipeline_state["results"] = results
            pipeline_state["status"] = "completed"
            pipeline_state["progress"] = 100
            pipeline_state["current_step"] = "completed"
            pipeline_state["step_details"] = "Pipeline finished successfully"
            pipeline_state["completed_at"] = datetime.now().isoformat()

        logger.info("✅ Pipeline completed successfully!")

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        tb = traceback.format_exc()
        logger.error(f"Pipeline failed: {error_msg}\n{tb}")
        with state_lock:
            pipeline_state["status"] = "error"
            pipeline_state["error"] = error_msg
            pipeline_state["step_details"] = f"Error: {error_msg}"


# ─── API Endpoints ───────────────────────────────────────────────────────────

@app.route("/")
def serve_frontend():
    """Serve the main frontend page."""
    return send_from_directory("frontend", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static frontend assets."""
    return send_from_directory("frontend", path)


@app.route("/api/pipeline/start", methods=["POST"])
def start_pipeline():
    """Start the ML pipeline in a background thread."""
    with state_lock:
        if pipeline_state["status"] == "running":
            return jsonify({"error": "Pipeline already running"}), 409

    data = request.get_json() or {}
    years = data.get("years", 7)

    thread = threading.Thread(target=run_full_pipeline, args=(years,), daemon=True)
    thread.start()

    return jsonify({"status": "started", "years": years})


@app.route("/api/pipeline/status")
def get_status():
    """Get current pipeline status."""
    with state_lock:
        return jsonify({
            "status": pipeline_state["status"],
            "progress": pipeline_state["progress"],
            "current_step": pipeline_state["current_step"],
            "step_details": pipeline_state["step_details"],
            "started_at": pipeline_state["started_at"],
            "completed_at": pipeline_state["completed_at"],
            "error": pipeline_state["error"],
        })


@app.route("/api/pipeline/stream")
def stream_progress():
    """Server-Sent Events endpoint for real-time progress updates."""
    def generate():
        last_progress = -1
        last_step = ""
        while True:
            with state_lock:
                status = pipeline_state["status"]
                progress = pipeline_state["progress"]
                step = pipeline_state["current_step"]
                details = pipeline_state["step_details"]
                error = pipeline_state["error"]

            if progress != last_progress or step != last_step:
                event_data = json.dumps({
                    "status": status,
                    "progress": progress,
                    "step": step,
                    "details": details,
                    "error": error,
                })
                yield f"data: {event_data}\n\n"
                last_progress = progress
                last_step = step

            if status in ("completed", "error"):
                # Send final event
                yield f"data: {json.dumps({'status': status, 'progress': 100 if status == 'completed' else progress, 'step': step, 'details': details, 'error': error})}\n\n"
                break

            time.sleep(0.5)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/pipeline/results")
def get_results():
    """Get full pipeline results."""
    with state_lock:
        if pipeline_state["results"] is None:
            return jsonify({"error": "No results available"}), 404
        return jsonify(pipeline_state["results"])


@app.route("/api/pipeline/logs")
def get_logs():
    """Get pipeline execution logs."""
    with state_lock:
        return jsonify({"logs": pipeline_state["logs"]})


@app.route("/api/pipeline/reset", methods=["POST"])
def reset_pipeline():
    """Reset pipeline state for a fresh run."""
    with state_lock:
        if pipeline_state["status"] == "running":
            return jsonify({"error": "Cannot reset while running"}), 409
        pipeline_state.update({
            "status": "idle",
            "progress": 0,
            "current_step": "",
            "step_details": "",
            "logs": [],
            "results": None,
            "error": None,
            "started_at": None,
            "completed_at": None,
        })
    return jsonify({"status": "reset"})


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("Starting AI Portfolio Optimizer Server...")
    app.run(host="0.0.0.0", port=5001, debug=False, threaded=True)
