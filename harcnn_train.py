import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pickle
TICKERS = [
    'AAPL', 'AMD', 'AMZN', 'GOOG', 'MSFT', 'NVDA',
    'MU', 'INTC', 'NFLX', 'NKE', 'SBUX', 'DIS',
    'TSLA', 'WMT', 'XOM', 'PFE', 'UNH', 'BA', 'CAT', 'GE',
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

TRAIN_RATIO = 0.75
VAL_RATIO = 0.15   # fraction of train split used for validation
BATCH_SIZE = 32
MAX_EPOCHS = 1000
LR = 1e-3
MIN_LR = 1e-4
DROPOUT = 0.5
L2_RATE = 1e-2
PATIENCE = 30
RANDOM_SEED = 42

DATA_DIR = "data"
WEIGHTS_PATH = "cnn_har_ks_weights.pth"
SCALER_PATH  = "cnn_image_scaler.pkl"

lags = [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
components_inorder = [
    "RV", "BPV", "ABD_jump", "ABD_CSP", "BNS_jump", "BNS_CSP",
    "Jo_jump", "Jo_CSP", "RS_positive", "RS_negative", "ret",
    "SJ", "SJ_positive", "SJ_negative", "negative_RV", "TQ"
]

def load_ticker(ticker: str) -> pd.DataFrame:
    path = f"{DATA_DIR}/{ticker}_stock_prices_2020_2024.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["ret"] = df["ret"].fillna(df["prc"].pct_change())
    df = df.dropna(subset=["ret"]).reset_index(drop=True)
    return df


def build_HAR_components(df: pd.DataFrame) -> pd.DataFrame:
    ret = df["ret"].values.copy()
    RV = ret ** 2
    abs_ret = np.abs(ret)
    BPV = np.concatenate([[0.0], abs_ret[:-1] * abs_ret[1:]])
    BPV_std = pd.Series(BPV).rolling(21, min_periods=5).std().values
    BPV_std = np.nan_to_num(BPV_std, nan=1e-8)
    BPV_std = np.where(BPV_std == 0, 1e-8, BPV_std)
    ABD_jump = np.maximum(RV - BPV, 0.0)
    ABD_CSP = RV - ABD_jump
    BNS_jump = np.where(RV > 3.0 * BPV_std, ABD_jump, 0.0)
    BNS_CSP = RV - BNS_jump
    Jo_jump = np.where(np.abs(ret) > 2.0 * BPV_std, ABD_jump, 0.0)
    Jo_CSP = RV - Jo_jump
    RS_positive = np.where(ret >= 0, RV, 0.0)
    RS_negative = np.where(ret < 0,  RV, 0.0)
    SJ = RS_positive - RS_negative
    SJ_positive = np.where(SJ > 0, SJ, 0.0)
    SJ_negative = np.where(SJ < 0, SJ, 0.0)
    negative_RV = np.where(ret < 0, RV, 0.0)
    TQ = np.abs(ret) ** (4.0 / 3.0)
    return pd.DataFrame({
        "RV": RV, "BPV": BPV, "ABD_jump": ABD_jump, "ABD_CSP": ABD_CSP,
        "BNS_jump": BNS_jump, "BNS_CSP": BNS_CSP, "Jo_jump": Jo_jump,
        "Jo_CSP": Jo_CSP, "RS_positive": RS_positive, "RS_negative": RS_negative,
        "ret": ret, "SJ": SJ, "SJ_positive": SJ_positive, "SJ_negative": SJ_negative,
        "negative_RV": negative_RV, "TQ": TQ,
    }, index=df.index)


def build_labels(components: pd.DataFrame) -> np.ndarray:
    RV = components["RV"].values
    label = np.where(np.roll(RV, -1) < RV, 1, 0)
    label[-1] = 0
    return label


def compute_rolling_window(series: np.ndarray, lags: list) -> np.ndarray:
    arr = np.zeros((len(series), len(lags)))
    for i, lag in enumerate(lags):
        if lag == 1:
            arr[:, i] = series
        else:
            arr[:, i] = pd.Series(series).rolling(lag, min_periods=1).mean().values
    return arr


def build_images(components: pd.DataFrame) -> np.ndarray:
    n = len(components)
    images = np.zeros((n, 16, 16), dtype=np.float32)
    for i, col in enumerate(components_inorder):
        images[:, i, :] = compute_rolling_window(components[col].values, lags)
    return images[:, np.newaxis, :, :]  # (n, 1, 16, 16)


def prepare_ticker(ticker: str):
    #Returns (images, labels, rv_next) for one ticker with burn in already dropped
    df = load_ticker(ticker)
    components = build_HAR_components(df)
    labels = build_labels(components)
    valid_comp = components.iloc[21:-1].reset_index(drop=True)
    labels_v = labels[21:-1]
    rv_next = components["RV"].values[22:]   # tomorrow's RV
    images = build_images(valid_comp)
    return images, labels_v, rv_next


class RVDataset(Dataset):
    def __init__(self, images, labels):
        self.x = torch.tensor(images, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CNN_HAR_KS(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=dropout),
        )
        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.fc_block(self.conv_block(x))

    def extract_features(self, x):
        x = self.conv_block(x)
        x = nn.Flatten()(x)
        x = self.fc_block[1](x)  # Linear(4096, 64)
        x = self.fc_block[2](x)  # ReLU
        return x    # (batch, 64)


def train_model(model, train_loader, val_loader):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=L2_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=PATIENCE // 2, min_lr=MIN_LR
    )
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y   = X.to(device), y.to(device)
                logits = model(X)
                val_loss += criterion(logits, y).item() * len(y)
                correct  += (logits.argmax(dim=1) == y).sum().item()
        val_loss /= len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"Early stop epoch {epoch} | best_val_loss {best_val_loss:.4f}")
                break

        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f}")

    model.load_state_dict(best_state)
    return model


def fit_image_scaler(images_train: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(images_train.reshape(len(images_train), -1))
    return scaler


def apply_image_scaler(scaler: StandardScaler, images: np.ndarray) -> np.ndarray:
    scaled = scaler.transform(images.reshape(len(images), -1))
    return scaled.reshape(images.shape).astype(np.float32)


def train_harcnn():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    all_images, all_labels = [], []
    for ticker in TICKERS:
        try:
            imgs, lbls, _ = prepare_ticker(ticker)
            all_images.append(imgs)
            all_labels.append(lbls)
            print(f"  {ticker}: {len(lbls)} samples")
        except Exception as e:
            print(f"  {ticker}: SKIPPED ({e})")

    images = np.concatenate(all_images, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"\nTotal pooled samples: {len(labels)}")

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.permutation(len(labels))
    n = len(labels)
    n_tv = int(n * TRAIN_RATIO)    # train+val
    n_t = int(n_tv * (1 - VAL_RATIO)) # train only

    idx_train = idx[:n_t]
    idx_val = idx[n_t:n_tv]

    x_train_raw = images[idx_train]
    x_val_raw = images[idx_val]
    y_train = labels[idx_train]
    y_val = labels[idx_val]

    img_scaler = fit_image_scaler(x_train_raw)
    x_train = apply_image_scaler(img_scaler, x_train_raw)
    x_val = apply_image_scaler(img_scaler, x_val_raw)

    with open(SCALER_PATH, "wb") as f:
        pickle.dump(img_scaler, f)
    print(f"Saved image scaler: {SCALER_PATH}")

    train_loader = DataLoader(RVDataset(x_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(RVDataset(x_val,   y_val),   batch_size=BATCH_SIZE, shuffle=False)

    print(f"Train: {len(y_train)} | Val: {len(y_val)}")
    print("Training CNN")
    model = CNN_HAR_KS(DROPOUT)
    print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    model = train_model(model, train_loader, val_loader)

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"Saved CNN weights: {WEIGHTS_PATH}")


if __name__ == "__main__":
    train_harcnn()
