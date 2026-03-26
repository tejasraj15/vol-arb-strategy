import os
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
TRAIN_END   = pd.Timestamp("2020-01-01")

def load_abs_log_returns(ticker):
    data_path = f"data/{ticker}_stock_prices_2010_2024.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date")
    df = df[(df["date"] >= TRAIN_START) & (df["date"] <= TRAIN_END)]
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
    abs_log_returns = load_abs_log_returns(ticker)
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

    # Train/validation split (chronological — no shuffling across the split)
    N_total = len(X)
    N_val = max(1, int(N_total * VAL_RATIO))
    N_train = N_total - N_val

    X_train = torch.tensor(X[:N_train], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y_train = torch.tensor(Y[:N_train], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    X_val   = torch.tensor(X[N_train:], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    Y_val   = torch.tensor(Y[N_train:], dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    N = X_train.size(0)

    print(f"{ticker}: {N_train} train sequences, {N_val} val sequences")

    ds3m = DS3M(X_DIM, Y_DIM, H_DIM, Z_DIM, D_DIM, N_LAYERS, DEVICE).to(DEVICE)
    optimizer = optim.Adam(ds3m.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, min_lr=MIN_LR)

    best_val_loss = float('inf')
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        ds3m.train()
        # Shuffle mini-batches each epoch
        perm = torch.randperm(N, device=DEVICE)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, N, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            # x_b: (batch, seq_len, 1) → (seq_len, batch, 1) for DS3M
            x_b = X_train[idx].transpose(0, 1)
            y_b = Y_train[idx].transpose(0, 1)

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

        avg_train_loss = epoch_loss / n_batches

        # Validation pass
        ds3m.eval()
        with torch.no_grad():
            val_loss = 0.0
            n_val_batches = 0
            for start in range(0, N_val, BATCH_SIZE):
                x_b = X_val[start:start + BATCH_SIZE].transpose(0, 1)
                y_b = Y_val[start:start + BATCH_SIZE].transpose(0, 1)
                out_v = ds3m.forward(x_b, y_b)
                v_loss = out_v['nll'] + out_v['kld_gaussian'] + out_v['kld_category']
                val_loss += v_loss.item()
                n_val_batches += 1
            avg_val_loss = val_loss / n_val_batches

        scheduler.step(avg_val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"{ticker} Epoch {epoch}/{MAX_EPOCHS} | Train: {avg_train_loss:.6f} | Val: {avg_val_loss:.6f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            torch.save(ds3m.state_dict(), f"{MODEL_DIR}/ds3m_{ticker}_best.pt")
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"{ticker} Early stopping at epoch {epoch} (best val loss: {best_val_loss:.6f})")
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
    print(f"Saved DS3M model for {ticker} to {MODEL_DIR}/ds3m_{ticker}.pt (best val loss: {best_val_loss:.6f})")

if __name__ == "__main__":
    import sys
    tickers_to_train = sys.argv[1:] if len(sys.argv) > 1 else TICKERS
    for ticker in tickers_to_train:
        try:
            train_ds3m_for_ticker(ticker.upper())
        except Exception as e:
            print(f"Failed to train DS3M for {ticker}: {e}")
