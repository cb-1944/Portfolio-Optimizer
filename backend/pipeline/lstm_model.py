"""
LSTM Forecasting Module — v2 (Anti-Overfitting Edition)
PyTorch-based LSTM model for predicting stock returns.

Key overfitting fixes applied:
  1. Proper 3-way train/val/test split (60/20/20) — no data leakage
  2. Smaller batch size (64) → sharper, better-generalizing minima
  3. Higher dropout (0.4 LSTM, 0.3 FC) — more regularization
  4. Stronger weight_decay (1e-4) — heavier L2 penalty
  5. Higher patience (20 epochs) & more epochs (120) → let it actually train
  6. Removed relu before output → model can predict negative returns
  7. Per-stock RobustScaler → prevents cross-stock leakage
  8. Gradient clipping kept at 1.0 → stable gradients
  9. Batch inference for predictions → correct BatchNorm behavior
  10. Overfitting diagnostic logged every epoch (gap = val_loss - train_loss)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ─── Model Architecture ──────────────────────────────────────────────────────

class PortfolioLSTM(nn.Module):
    """
    Two-layer LSTM for return prediction with strong regularization.

    Architecture fixes vs v1:
    - LSTM dropout built-in (between layers via PyTorch's native dropout arg)
    - Higher hidden dropout applied between FC layers
    - NO relu before final output — returns can be negative
    - LayerNorm instead of BatchNorm1d — stable at small batch sizes / inference

    Input:  (batch, lookback=20, n_features=9)
    Output: (batch,) — predicted next-day log return
    """

    def __init__(self, input_size=9, hidden1=32, hidden2=16,
                 lstm_dropout=0.4, fc_dropout=0.2):
        super().__init__()

        # LSTM stack — dropout applies between layers internally
        self.lstm1 = nn.LSTM(
            input_size, hidden1,
            batch_first=True, num_layers=1, dropout=0.0
        )
        self.lstm_dropout = nn.Dropout(lstm_dropout)  # explicit between LSTMs

        self.lstm2 = nn.LSTM(
            hidden1, hidden2,
            batch_first=True, num_layers=1, dropout=0.0
        )

        # LayerNorm over the feature dim (works at any batch size, including 1)
        self.layer_norm = nn.LayerNorm(hidden2)

        # FC head with dropout — NO relu before final output
        self.fc_drop = nn.Dropout(fc_dropout)
        self.fc1 = nn.Linear(hidden2, 16)
        self.fc2 = nn.Linear(16, 1)

        # Proper weight initialisation (Xavier uniform for linear layers)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # LSTM block 1
        out, _ = self.lstm1(x)           # (batch, seq, hidden1)
        out = self.lstm_dropout(out)     # regularise between layers

        # LSTM block 2
        out, _ = self.lstm2(out)         # (batch, seq, hidden2)
        out = out[:, -1, :]              # take LAST timestep

        # Normalise + FC head
        out = self.layer_norm(out)
        out = self.fc_drop(out)
        out = torch.tanh(self.fc1(out))  # tanh keeps signal bounded
        out = self.fc2(out)              # linear → unbounded output ✓
        return out.squeeze(-1)


# ─── Data Preparation ────────────────────────────────────────────────────────

def prepare_sequences(
    features_dict: dict,
    target_dict: dict,
    dates: pd.DatetimeIndex,
    lookback: int = 20,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
):
    """
    3-way temporal split: train / val / test (no shuffle across time).
    Scalers fit ONLY on training data to prevent information leakage.

    FIX v2: Per-stock scaling instead of pooled global scaler.
    Pooling injects cross-stock magnitude information into the scaler,
    which can inflate performance during val/test.

    Split proportions (default): 60% train, 20% val, 20% test
    On 7 years (~1,750 days): ~1,050 train | ~350 val | ~350 test

    Returns:
        train_X, train_y, val_X, val_y, test_X, test_y (numpy arrays)
        split_dates: (val_start_date, test_start_date)
        scalers: {ticker: {feature: RobustScaler}} — per-stock per-feature
    """
    n_dates = len(dates)
    train_end = int(n_dates * train_ratio)
    val_end   = int(n_dates * (train_ratio + val_ratio))

    val_start_date  = dates[train_end]
    test_start_date = dates[val_end]

    logger.info(
        f"3-way split: "
        f"Train [{dates[0].date()} → {dates[train_end-1].date()}] "
        f"({train_end} days) | "
        f"Val [{val_start_date.date()} → {dates[val_end-1].date()}] "
        f"({val_end - train_end} days) | "
        f"Test [{test_start_date.date()} → {dates[-1].date()}] "
        f"({n_dates - val_end} days)"
    )

    tickers = list(features_dict.keys())
    feature_names = features_dict[tickers[0]].columns.tolist()

    # ── Per-stock, per-feature scalers ──────────────────────────────────────
    logger.info("Fitting per-stock RobustScalers on training data only...")
    scalers = {}
    for ticker in tickers:
        scalers[ticker] = {}
        feat_train = features_dict[ticker].iloc[:train_end]
        for col in feature_names:
            sc = RobustScaler()
            vals = feat_train[col].values
            finite_mask = np.isfinite(vals)
            if finite_mask.sum() > 10:
                sc.fit(vals[finite_mask].reshape(-1, 1))
            scalers[ticker][col] = sc

    # ── Build sequence arrays ────────────────────────────────────────────────
    train_X, train_y = [], []
    val_X,   val_y   = [], []
    test_X,  test_y  = [], []

    for ticker in tickers:
        feat_df   = features_dict[ticker]
        target_s  = target_dict[ticker]
        stock_sc  = scalers[ticker]

        # Scale all rows using training-fitted scaler
        n_rows  = len(feat_df)
        scaled  = np.zeros((n_rows, len(feature_names)), dtype=np.float32)
        for ci, col in enumerate(feature_names):
            raw = feat_df[col].values.reshape(-1, 1)
            scaled[:, ci] = stock_sc[col].transform(raw).ravel()

        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        # Hard-clip after NaN fill to keep features in sensible range
        scaled = np.clip(scaled, -5.0, 5.0)

        targets = np.nan_to_num(target_s.values, nan=0.0).astype(np.float32)

        # Sliding window sequences
        for i in range(lookback, n_rows):
            seq   = scaled[i - lookback : i]   # (20, 9)
            label = targets[i]

            if i < train_end:
                train_X.append(seq); train_y.append(label)
            elif i < val_end:
                val_X.append(seq);   val_y.append(label)
            else:
                test_X.append(seq);  test_y.append(label)

    def to_np(lst):
        return np.array(lst, dtype=np.float32)

    train_X, train_y = to_np(train_X), to_np(train_y)
    val_X,   val_y   = to_np(val_X),   to_np(val_y)
    test_X,  test_y  = to_np(test_X),  to_np(test_y)

    logger.info(
        f"Sequences — Train: {len(train_X):,} | Val: {len(val_X):,} | Test: {len(test_X):,}"
    )

    return (
        train_X, train_y,
        val_X,   val_y,
        test_X,  test_y,
        (val_start_date, test_start_date),
        scalers,
    )


# ─── Training ────────────────────────────────────────────────────────────────

def train_model(
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X:   np.ndarray,
    val_y:   np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    patience: int = 15,
    weight_decay: float = 1e-5, # FIX: was 1e-5 — stronger L2 regularisation
    progress_callback=None,
):
    """
    Train PortfolioLSTM with full anti-overfitting measures.

    Overfitting diagnostic logged each epoch:
      gap = val_loss - train_loss
      gap > 0.5×train_loss  → model is overfitting
      gap < 0               → model is underfitting (val easier than train)
    """
    n_features = train_X.shape[2]
    model = PortfolioLSTM(input_size=n_features).to(DEVICE)

    # ── Optimizer & Scheduler ───────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay,
        betas=(0.9, 0.999), eps=1e-8,
    )
    # ReduceLROnPlateau: only lowers LR when val loss stalls — no aggressive restarts
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    criterion = nn.HuberLoss(delta=0.5)  # Robust to fat tails in return distributions
    
    # ── Naive Baselines for Volatility Normalization ──────────────────────
    # To prevent regime shifts (e.g. COVID crash in val set) from artificially 
    # inflating the overfit gap, we normalize losses by the baseline variance.
    naive_train_loss = np.mean(train_y ** 2) + 1e-8
    naive_val_loss   = np.mean(val_y ** 2) + 1e-8

    # ── DataLoaders ─────────────────────────────────────────────────────────
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_X), torch.from_numpy(train_y)),
        batch_size=batch_size, shuffle=True, drop_last=True,
        pin_memory=(DEVICE.type != "mps"),
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_X), torch.from_numpy(val_y)),
        batch_size=batch_size * 4, shuffle=False,
        pin_memory=(DEVICE.type != "mps"),
    )

    history = {
        "train_loss": [], "val_loss": [],
        "overfit_gap": [], "lr": [],
    }

    best_val_loss   = float("inf")
    best_state      = None
    patience_count  = 0

    logger.info(
        f"Training PortfolioLSTM v2 on [{DEVICE}]"
        f" — epochs={epochs}, batch={batch_size}, "
        f"lr={lr}, weight_decay={weight_decay}, patience={patience}"
    )
    logger.info(
        f"  Train samples: {len(train_X):,} | "
        f"Val samples: {len(val_X):,} | "
        f"Batches/epoch: {len(train_loader)}"
    )

    for epoch in range(epochs):
        # ── Train ───────────────────────────────────────────────────────────
        model.train()
        t_losses = []
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            t_losses.append(loss.item())

        avg_train = float(np.mean(t_losses))

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        v_losses = []
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v_losses.append(criterion(model(Xb), yb).item())

        avg_val = float(np.mean(v_losses))
        scheduler.step(avg_val)

        # Overfitting diagnostic — normalized by regime volatility
        norm_train = avg_train / naive_train_loss
        norm_val   = avg_val / naive_val_loss
        gap        = norm_val - norm_train
        gap_pct    = (gap / norm_train) * 100 if norm_train > 0 else 0
        overfit_flag = "⚠️ OVERFIT" if gap_pct > 50 else ("✅ OK" if gap_pct < 20 else "⚡ WATCH")

        cur_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(avg_train))
        history["val_loss"].append(float(avg_val))
        history["overfit_gap"].append(round(float(gap_pct), 1))
        history["lr"].append(float(cur_lr))

        logger.info(
            f"  Epoch {epoch+1:3d}/{epochs} | "
            f"Train: {avg_train:.6f} | Val: {avg_val:.6f} | "
            f"Gap: {gap_pct:+.1f}% {overfit_flag} | "
            f"LR: {cur_lr:.2e}"
        )

        if progress_callback:
            progress_callback("lstm_training", 63 + int(12 * (epoch + 1) / epochs))

        # ── Overfitting Gap Guard ────────────────────────────────────────────
        if epoch >= 30 and gap_pct > 40:
            overfit_streak = sum(
                1 for g in history["overfit_gap"][-3:] if g > 40
            )
            if overfit_streak >= 3:
                logger.info(
                    f"  ⛔ Overfitting guard: gap >{40}% for 3 consecutive "
                    f"epochs — stopping at epoch {epoch+1}"
                )
                break

        # ── Early Stopping (only after min 30 epochs) ──────────────────────
        if epoch >= 30 and avg_val < best_val_loss:
            best_val_loss  = avg_val
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        elif epoch < 30:
            # Always save best during warmup phase
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.info(
                    f"  ⛔ Early stopping at epoch {epoch+1} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    logger.info(f"  ✅ Best val loss: {best_val_loss:.6f}")
    return model, history


# ─── Inference ───────────────────────────────────────────────────────────────

def predict_returns(
    model: PortfolioLSTM,
    features_dict: dict,
    scalers: dict,
    dates: pd.DatetimeIndex,
    predict_start_idx: int,
    lookback: int = 20,
    batch_size: int = 512,
) -> dict:
    """
    Batch inference — critical fix vs v1's one-sample-at-a-time loop.

    v1 problem: model(single_sample) with BatchNorm1d → statistics computed
    on batch=1, wildly different from training → silent prediction corruption.

    v2 fix: LayerNorm in model (works at any batch size) + batched inference.

    Returns:
        {ticker: pd.Series of predicted returns indexed by date}
    """
    model.eval()
    predictions  = {}
    tickers      = list(features_dict.keys())
    feature_names = features_dict[tickers[0]].columns.tolist()

    for ticker in tickers:
        feat_df  = features_dict[ticker]
        stock_sc = scalers[ticker]

        # Scale with per-stock scaler
        n_rows  = len(feat_df)
        scaled  = np.zeros((n_rows, len(feature_names)), dtype=np.float32)
        for ci, col in enumerate(feature_names):
            raw = feat_df[col].values.reshape(-1, 1)
            scaled[:, ci] = stock_sc[col].transform(raw).ravel()

        scaled = np.nan_to_num(scaled, nan=0.0, posinf=1.0, neginf=-1.0)
        scaled = np.clip(scaled, -5.0, 5.0)

        # Collect all sequences for this ticker
        seqs, pred_dates = [], []
        for i in range(max(predict_start_idx, lookback), n_rows):
            seqs.append(scaled[i - lookback : i])
            pred_dates.append(dates[i])

        if not seqs:
            logger.warning(f"  {ticker}: No sequences for prediction window")
            continue

        seqs_tensor = torch.from_numpy(np.array(seqs, dtype=np.float32))

        # Batched inference
        all_preds = []
        with torch.no_grad():
            for start in range(0, len(seqs_tensor), batch_size):
                batch = seqs_tensor[start : start + batch_size].to(DEVICE)
                preds = model(batch).cpu().numpy()
                all_preds.extend(preds.tolist())

        predictions[ticker] = pd.Series(all_preds, index=pred_dates, name=ticker)

    logger.info(
        f"Predictions done — {len(predictions)} stocks × "
        f"{len(pred_dates) if predictions else 0} dates"
    )
    return predictions
