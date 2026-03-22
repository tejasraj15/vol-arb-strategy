"""
DS3M Training Pipeline
Handles data preparation, training loop, validation, and model saving.
"""

import torch
import torch.optim as optim
import numpy as np
import os


def prepare_sequences(vol_series, features, seq_len=20):
    """
    Slice time series into overlapping subsequences for training.

    Args:
        vol_series: numpy array of realized vol, shape (T,) or (T, y_dim)
        features:   numpy array of input features, shape (T, x_dim)
        seq_len:    length of each subsequence

    Returns:
        X: (seq_len, n_sequences, x_dim)
        Y: (seq_len, n_sequences, y_dim)
    """
    if vol_series.ndim == 1:
        vol_series = vol_series.reshape(-1, 1)

    T = len(vol_series)
    n_seq = T - seq_len

    X_list, Y_list = [], []
    for i in range(n_seq):
        X_list.append(features[i:i + seq_len])
        Y_list.append(vol_series[i:i + seq_len])

    X = np.stack(X_list, axis=1)  # (seq_len, n_seq, x_dim)
    Y = np.stack(Y_list, axis=1)  # (seq_len, n_seq, y_dim)

    return X, Y


def train_val_test_split(X, Y, train_frac=0.6, val_frac=0.2):
    """
    Split sequences chronologically (no shuffling - this is time series).
    """
    n = X.shape[1]
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    return {
        'train_x': X[:, :train_end, :],
        'train_y': Y[:, :train_end, :],
        'val_x': X[:, train_end:val_end, :],
        'val_y': Y[:, train_end:val_end, :],
        'test_x': X[:, val_end:, :],
        'test_y': Y[:, val_end:, :],
    }


def train_epoch(model, optimizer, X, Y, epoch, n_epochs, batch_size=64):
    """
    One training epoch with KL annealing.
    """
    model.train()

    # KL annealing: start small, ramp up (prevents posterior collapse)
    if epoch < n_epochs / 2:
        anneal = 0.01
    else:
        anneal = min(1.0, 0.01 + (epoch - n_epochs / 2) / (n_epochs / 2))

    total_loss = 0
    n_batches = 0

    for batch_start in range(0, X.shape[1], batch_size):
        bx = X[:, batch_start:batch_start + batch_size, :]
        by = Y[:, batch_start:batch_start + batch_size, :]

        out = model(bx, by)

        size = bx.shape[0] * bx.shape[1]
        kld = out['kld_gaussian'] + out['kld_category']
        loss = anneal * kld / size + out['nll'] / size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, X, Y):
    """
    Compute validation loss (no gradient).
    """
    model.eval()
    with torch.no_grad():
        out = model(X, Y)
        size = X.shape[0] * X.shape[1]
        loss = (out['kld_gaussian'] + out['kld_category'] + out['nll']) / size
    return loss.item()


def train_ds3m(model, train_x, train_y, val_x=None, val_y=None,
               n_epochs=100, batch_size=64, lr=0.001, patience=20,
               save_path='ds3m_best.pt'):
    """
    Full training loop with early stopping and learning rate scheduling.

    Args:
        model:     DS3M model instance
        train_x:   training inputs,  tensor (seq_len, n_train, x_dim)
        train_y:   training targets, tensor (seq_len, n_train, y_dim)
        val_x:     validation inputs (optional)
        val_y:     validation targets (optional)
        n_epochs:  maximum epochs
        batch_size: mini-batch size
        lr:        initial learning rate
        patience:  early stopping patience
        save_path: where to save best model

    Returns:
        dict with training history
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10
    )

    use_validation = val_x is not None and val_y is not None

    best_metric = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(n_epochs):

        # Train
        train_loss = train_epoch(
            model, optimizer, train_x, train_y, epoch, n_epochs, batch_size
        )
        history['train_loss'].append(train_loss)

        # Validate if provided
        if use_validation:
            val_loss = validate(model, val_x, val_y)
            history['val_loss'].append(val_loss)
            metric = val_loss
        else:
            val_loss = None
            metric = train_loss

        # LR scheduling (use train loss when validation is disabled)
        scheduler.step(metric)

        # Logging
        if epoch % 10 == 0:
            trans = model.get_transition_matrix().detach().cpu().numpy()
            if use_validation:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            else:
                print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"          Transition matrix:\n{np.array2string(trans, precision=3)}")

        # Early stopping
        if metric < best_metric:
            best_metric = metric
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Load best model
    model.load_state_dict(torch.load(save_path, weights_only=True))
    if use_validation:
        print(f"\nTraining complete. Best val loss: {best_metric:.4f}")
    else:
        print(f"\nTraining complete. Best train loss: {best_metric:.4f}")
    print(f"Learned transition matrix:")
    print(model.get_transition_matrix().detach().cpu().numpy())

    return history
