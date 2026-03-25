"""
Loads the pre-trained CNN (from harcnn_train.py), extracts features
for each ticker, concatenates with classic HAR predictors (RV_d, RV_w, RV_m),
then fits a separate Ridge regressor per ticker to get the RV.
"""

import numpy as np
import pandas as pd
import torch
import pickle
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from harcnn_train import (
    TICKERS, TRAIN_RATIO, DROPOUT, WEIGHTS_PATH, SCALER_PATH,
    CNN_HAR_KS, load_ticker, build_HAR_components, build_images,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)


def build_har_features(components: pd.DataFrame) -> np.ndarray:
    # HAR predictors: RV_d (lag-1), RV_w (5d mean), RV_m (22d mean)
    rv = components["RV"].values
    rv_d = rv
    rv_w = pd.Series(rv).rolling(5,  min_periods=1).mean().values
    rv_m = pd.Series(rv).rolling(22, min_periods=1).mean().values
    return np.column_stack([rv_d, rv_w, rv_m])


def extract_cnn_features(model: CNN_HAR_KS, images: np.ndarray, img_scaler: StandardScaler) -> np.ndarray:
    scaled = img_scaler.transform(images.reshape(len(images), -1)).reshape(images.shape).astype(np.float32)
    model.eval()
    feats = []
    with torch.no_grad():
        for i in range(0, len(scaled), 64):
            batch = torch.tensor(scaled[i:i+64]).to(device)
            feats.append(model.extract_features(batch).cpu().numpy())
    return np.vstack(feats)


def fit_ridge(X_train, y_train):
    feat_scaler = StandardScaler()
    X_train_scaled = feat_scaler.fit_transform(X_train)

    log_y_train = np.log(y_train + 1e-10)
    rv_scaler = StandardScaler()
    y_train_scaled = rv_scaler.fit_transform(log_y_train.reshape(-1, 1)).ravel()

    regressor = Ridge(alpha=1.0)
    regressor.fit(X_train_scaled, y_train_scaled)
    return regressor, rv_scaler, feat_scaler


def predict_rv(regressor, rv_scaler, feat_scaler, X_test) -> np.ndarray:
    X_scaled = feat_scaler.transform(X_test)
    log_rv_scaled = regressor.predict(X_scaled)
    log_rv = rv_scaler.inverse_transform(log_rv_scaled.reshape(-1, 1)).ravel()
    return np.exp(log_rv)


def forecast_next_rv(stock_data: pd.DataFrame, cnn_model: CNN_HAR_KS,
                     img_scaler: StandardScaler, ridge_bundle: dict) -> float:
    # stock_data must have a 'ret' column and at least 22 rows (21 burn-in + 1)
    components = build_HAR_components(stock_data)
    valid_comp = components.iloc[21:].reset_index(drop=True)  # drop burn-in, keep all
    if len(valid_comp) == 0:
        raise ValueError("Not enough data after burn-in (need > 21 rows)")

    image = build_images(valid_comp) # (n, 1, 16, 16)
    har = build_har_features(valid_comp)   # (n, 3)
    cnn_feat = extract_cnn_features(cnn_model, image[-1:], img_scaler)  # last row only
    X = np.hstack([cnn_feat, har[-1:]]) # (1, 67)

    return float(predict_rv(
        ridge_bundle["regressor"],
        ridge_bundle["rv_scaler"],
        ridge_bundle["feat_scaler"],
        X,
    )[0])


def fit_ridge_for_ticker(ticker: str, cnn_model: CNN_HAR_KS, img_scaler) -> dict:
    df = load_ticker(ticker)
    components = build_HAR_components(df)
    valid_comp = components.iloc[21:-1].reset_index(drop=True)
    rv_next = components["RV"].values[22:]

    images = build_images(valid_comp)
    har_feats = build_har_features(valid_comp)

    n_train = int(len(rv_next) * TRAIN_RATIO)

    cnn_train = extract_cnn_features(cnn_model, images[:n_train], img_scaler)
    har_train = har_feats[:n_train]
    rv_train = rv_next[:n_train]

    X_train = np.hstack([cnn_train, har_train])
    regressor, rv_scaler, feat_scaler = fit_ridge(X_train, rv_train)

    bundle = {"regressor": regressor, "rv_scaler": rv_scaler, "feat_scaler": feat_scaler}
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/ridge_{ticker}.pkl", "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def main():
    model = CNN_HAR_KS(DROPOUT).to(device)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model.eval()
    print(f"Loaded CNN weights from {WEIGHTS_PATH}")

    with open(SCALER_PATH, "rb") as f:
        img_scaler = pickle.load(f)
    print(f"Loaded image scaler from {SCALER_PATH}")

    all_forecasts = []

    for ticker in TICKERS:
        try:
            df = load_ticker(ticker)
            components = build_HAR_components(df)
            dates = df["date"].values[22:] # aligned to rv_next
            valid_comp = components.iloc[21:-1].reset_index(drop=True)
            rv_next = components["RV"].values[22:]   # tomorrow's RV target

            images = build_images(valid_comp)  # (n, 1, 16, 16)
            har_feats = build_har_features(valid_comp)    # (n, 3)

            n = len(rv_next)
            n_train = int(n * TRAIN_RATIO)

            images_train = images[:n_train]
            images_test = images[n_train:]
            har_train = har_feats[:n_train]
            har_test = har_feats[n_train:]
            rv_train = rv_next[:n_train]
            rv_test = rv_next[n_train:]
            dates_test = dates[n_train:]

            # Extract CNN features
            cnn_train = extract_cnn_features(model, images_train, img_scaler)
            cnn_test = extract_cnn_features(model, images_test,  img_scaler)

            # CNN + HAR features
            X_train = np.hstack([cnn_train, har_train])
            X_test = np.hstack([cnn_test,  har_test])

            # fit Ridge on train, predict on test
            regressor, rv_scaler, feat_scaler = fit_ridge(X_train, rv_train)
            rv_pred = predict_rv(regressor, rv_scaler, feat_scaler, X_test)

            mae  = mean_absolute_error(rv_test, rv_pred)
            rmse = np.sqrt(mean_squared_error(rv_test, rv_pred))
            corr = np.corrcoef(rv_test, rv_pred)[0, 1]
            print(f"  MAE={mae:.6f}  RMSE={rmse:.6f}  Corr={corr:.4f}")

            bundle = {"regressor": regressor, "rv_scaler": rv_scaler, "feat_scaler": feat_scaler}
            with open(f"{MODELS_DIR}/ridge_{ticker}.pkl", "wb") as f:
                pickle.dump(bundle, f)

            # Collect forecasts
            for d, rv_f, rv_a in zip(dates_test, rv_pred, rv_test):
                all_forecasts.append({
                    "ticker": ticker,
                    "date": pd.Timestamp(d),
                    "rv_forecast": rv_f,
                    "rv_actual": rv_a,
                })

        except Exception as e:
            print(f"  SKIPPED: {e}")

    forecast_df = pd.DataFrame(all_forecasts).sort_values(["date", "ticker"])
    forecast_df.to_csv("rv_forecasts.csv", index=False)
    print(f"\nSaved forecasts: rv_forecasts.csv ({len(forecast_df)} rows)")


if __name__ == "__main__":
    main()
