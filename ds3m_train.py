import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from ds3m_model import DS3M

TICKERS = [
    'AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT', 'NVDA',
    'MU', 'INTC', 'NFLX', 'NKE', 'SBUX', 'DIS',
    'TSLA', 'WMT', 'XOM', 'PFE', 'UNH', 'BA', 'CAT', 'GE',
]

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device("cpu")
SEQ_LEN = 20
X_DIM = 1
Y_DIM = 1
H_DIM = 32
Z_DIM = 4
D_DIM = 2
N_LAYERS = 1
TRAIN_RATIO = 0.75
VAL_RATIO = 0.15   # fraction of train split used for validation
MAX_EPOCHS = 300
LR = 1e-3
MIN_LR = 1e-4
L2_RATE = 1e-2
PATIENCE = 30
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_START = pd.Timestamp("2010-01-01")
TRAIN_END   = pd.Timestamp("2019-12-31")
OOS_START   = pd.Timestamp("2020-01-01")
OOS_END     = pd.Timestamp("2024-12-31")

def load_abs_log_returns(ticker, start, end):
    data_path = f"data/{ticker}_stock_prices_2010_2024.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    prices = df["prc"].astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    abs_log_returns = np.abs(log_returns)
    return abs_log_returns

def build_ds3m_sequences(abs_log_returns, seq_len=SEQ_LEN):
    X_list, Y_list = [], []
    for i in range(len(abs_log_returns) - seq_len):
        X_list.append(abs_log_returns.iloc[i:i+seq_len].values)
        Y_list.append(abs_log_returns.iloc[i+1:i+seq_len+1].values)
    X = np.array(X_list)
    Y = np.array(Y_list)
    return X, Y

def train_ds3m_for_ticker(ticker):
    abs_log_returns = load_abs_log_returns(ticker, TRAIN_START, TRAIN_END)
    X, Y = build_ds3m_sequences(abs_log_returns)
    if len(X) == 0:
        print(f"Not enough data for {ticker}")
        return

    # Z-score normalise using training data statistics
    rv_mean = abs_log_returns.mean()
    rv_std  = abs_log_returns.std()
    scaler = {'mean': float(rv_mean), 'std': float(rv_std)}
    X = (X - float(rv_mean)) / float(rv_std)
    Y = (Y - float(rv_mean)) / float(rv_std)

    # (N, seq_len, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    N = X_tensor.size(0)

    ds3m = DS3M(X_DIM, Y_DIM, H_DIM, Z_DIM, D_DIM, N_LAYERS, DEVICE).to(DEVICE)
    optimizer = optim.Adam(ds3m.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=MIN_LR)

    best_loss = float('inf')
    no_improve = 0

    ds3m.train()
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        t_epoch = time.time()
        # Shuffle mini-batches each epoch
        perm = torch.randperm(N, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            # x_b: (batch, seq_len, 1) → (seq_len, batch, 1) for DS3M
            x_b = X_tensor[idx].transpose(0, 1)
            y_b = Y_tensor[idx].transpose(0, 1)

            optimizer.zero_grad()
            out = ds3m.forward(x_b, y_b)
            # KL annealing: ramp KL weight from 0.01 to 1.0 over second half of training
            if epoch < MAX_EPOCHS / 2:
                anneal = 0.01
            else:
                anneal = min(1.0, 0.01 + (epoch - MAX_EPOCHS / 2) / (MAX_EPOCHS / 2))
            loss = out['nll'] + anneal * (out['kld_gaussian'] + out['kld_category'])
            loss.backward()
            nn.utils.clip_grad_norm_(ds3m.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)
        epoch_secs = time.time() - t_epoch

        if epoch % 10 == 0 or epoch == 1:
            print(f"{ticker} Epoch {epoch}/{MAX_EPOCHS} | Loss: {avg_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e} | {epoch_secs:.2f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            no_improve = 0
            torch.save(ds3m.state_dict(), f"{MODEL_DIR}/ds3m_{ticker}_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"{ticker} Early stopping at epoch {epoch}")
                break

    # Use best checkpoint
    best_path = f"{MODEL_DIR}/ds3m_{ticker}_best.pt"
    if os.path.exists(best_path):
        ds3m.load_state_dict(torch.load(best_path, map_location=DEVICE))
        os.replace(best_path, f"{MODEL_DIR}/ds3m_{ticker}.pt")
    else:
        torch.save(ds3m.state_dict(), f"{MODEL_DIR}/ds3m_{ticker}.pt")
    scaler_path = f"{MODEL_DIR}/ds3m_scaler_{ticker}.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    elapsed = time.time() - t_start
    print(f"Saved DS3M model for {ticker} to {MODEL_DIR}/ds3m_{ticker}.pt (best loss: {best_loss:.6f}) | Training time: {elapsed:.1f}s")

def evaluate_ds3m_for_ticker(ticker, n_samples=100):
    model_path = f"{MODEL_DIR}/ds3m_{ticker}.pt"
    scaler_path = f"{MODEL_DIR}/ds3m_scaler_{ticker}.pkl"
    if not os.path.exists(model_path):
        print(f"No trained model found for {ticker}, skipping evaluation.")
        return

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    rv_mean, rv_std = scaler['mean'], scaler['std']

    abs_log_returns = load_abs_log_returns(ticker, OOS_START, OOS_END)
    X, Y = build_ds3m_sequences(abs_log_returns)
    if len(X) == 0:
        print(f"Not enough OOS data for {ticker}")
        return

    X_norm = (X - rv_mean) / rv_std
    Y_norm = (Y - rv_mean) / rv_std

    X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y_tensor = torch.tensor(Y_norm, dtype=torch.float32, device=DEVICE).unsqueeze(-1)

    ds3m = DS3M(X_DIM, Y_DIM, H_DIM, Z_DIM, D_DIM, N_LAYERS, DEVICE).to(DEVICE)
    ds3m.load_state_dict(torch.load(model_path, map_location=DEVICE))
    ds3m.eval()

    all_preds, all_targets = [], []

    with torch.no_grad():
        for i in range(len(X_tensor)):
            x_b = X_tensor[i].unsqueeze(1)   # (seq_len, 1, 1)
            y_b = Y_tensor[i].unsqueeze(1)

            fc = ds3m.forecast(x_b, y_b, steps=1, n_samples=n_samples)
            # forecast is normalized; denormalize
            pred_norm = fc['vol_forecast_mean'][0, 0, 0]   # (steps, batch, y_dim)
            pred = pred_norm * rv_std + rv_mean

            target = Y[i, -1]   # last step target in original scale
            all_preds.append(pred)
            all_targets.append(target)

    preds = np.array(all_preds)
    targets = np.array(all_targets)

    mse = np.mean((preds - targets) ** 2)
    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(mse)
    # QLIKE loss: common in vol forecasting literature
    qlike = np.mean(targets / (preds + 1e-8) - np.log(targets / (preds + 1e-8)) - 1)

    print(f"\n{'='*50}")
    print(f"OOS Evaluation: {ticker} ({OOS_START.date()} to {OOS_END.date()})")
    print(f"  Samples : {len(targets)}")
    print(f"  RMSE    : {rmse:.6f}")
    print(f"  MAE     : {mae:.6f}")
    print(f"  QLIKE   : {qlike:.6f}")
    print(f"{'='*50}\n")

    return {'ticker': ticker, 'rmse': rmse, 'mae': mae, 'qlike': qlike, 'n': len(targets)}


def train_ds3m(ticker: str) -> None:
    """Train a DS3M model for a single ticker and save weights + scaler to disk."""
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    train_ds3m_for_ticker(ticker.upper())


def _train_worker(ticker):
    try:
        train_ds3m_for_ticker(ticker)
    except Exception as e:
        print(f"Failed to train DS3M for {ticker}: {e}")


def _eval_worker(ticker):
    try:
        return evaluate_ds3m_for_ticker(ticker)
    except Exception as e:
        print(f"Failed to evaluate DS3M for {ticker}: {e}")
        return None


if __name__ == "__main__":
    import sys
    from concurrent.futures import ProcessPoolExecutor, as_completed

    args = sys.argv[1:]
    mode = "train"
    workers = 2  # default parallelism

    if args and args[0] in ("--eval", "--evaluate"):
        mode = "eval"
        args = args[1:]

    # --workers N  (must come after --eval if used together)
    if "--workers" in args:
        idx = args.index("--workers")
        workers = int(args[idx + 1])
        args = args[:idx] + args[idx + 2:]

    tickers_to_run = [t.upper() for t in args] if args else TICKERS

    if mode == "train":
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_train_worker, t): t for t in tickers_to_run}
            for fut in as_completed(futures):
                fut.result()  # surface any unexpected exceptions
    else:
        results = []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_eval_worker, t): t for t in tickers_to_run}
            for fut in as_completed(futures):
                r = fut.result()
                if r:
                    results.append(r)
        if results:
            df = pd.DataFrame(results).set_index('ticker')
            print("\nAggregate OOS Results:")
            print(df.to_string())
