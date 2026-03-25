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
MAX_EPOCHS = 1000
LR = 1e-3
MIN_LR = 1e-4
L2_RATE = 1e-2
PATIENCE = 30
RANDOM_SEED = 42
BATCH_SIZE = 32
EPOCHS = 50
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def load_abs_log_returns(ticker):
    data_path = f"data/{ticker}_stock_prices_2020_2024.csv"
    df = pd.read_csv(data_path, parse_dates=["date"])
    df = df.sort_values("date")
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
    X_tensor = torch.tensor(X, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (N, seq_len, 1)
    Y_tensor = torch.tensor(Y, dtype=torch.float32, device=DEVICE).unsqueeze(-1)  # (N, seq_len, 1)

    ds3m = DS3M(X_DIM, Y_DIM, H_DIM, Z_DIM, D_DIM, N_LAYERS, DEVICE).to(DEVICE)
    optimizer = optim.Adam(ds3m.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    ds3m.train()
    for epoch in range(1, MAX_EPOCHS + 1):
        optimizer.zero_grad()
        # For demonstration, use only the last window (batch size 1)
        out = ds3m.forward(X_tensor.transpose(0, 1), Y_tensor.transpose(0, 1))
        # Dummy loss: mean squared error between predicted and actual (placeholder)
        # Replace with your actual DS3M loss
        loss = loss_fn(X_tensor, Y_tensor)  # Placeholder
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"{ticker} Epoch {epoch}/{MAX_EPOCHS} | Loss: {loss.item():.6f}")

    # Save model
    model_path = f"{MODEL_DIR}/ds3m_{ticker}.pt"
    torch.save(ds3m.state_dict(), model_path)
    print(f"Saved DS3M model for {ticker} to {model_path}")

if __name__ == "__main__":
    for ticker in TICKERS:
        try:
            train_ds3m_for_ticker(ticker)
        except Exception as e:
            print(f"Failed to train DS3M for {ticker}: {e}")
