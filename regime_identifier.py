# note:
# think about allowing for predictable stress in future and just blocking unstructured stress

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_sample_weight


class RegimeBlockerXGB:
    
    def __init__(
        self,
        returns: pd.Series,
        extra_features: Optional[pd.DataFrame] = None,
        stress_vol_percentile: float = 90.0,
        stress_drawdown_threshold: float = -0.05,
        calm_vol_percentile: float = 25.0,
        block_threshold: float = 0.5,
        n_splits: int = 5,
        test_size: float = 0.2,
        min_cv_accuracy: float = 0.85,
        random_state: int = 42,
        verbose: bool = True
    ):
        self.stress_vol_percentile = stress_vol_percentile
        self.stress_drawdown_threshold = stress_drawdown_threshold
        self.calm_vol_percentile = calm_vol_percentile
        self.block_threshold = block_threshold
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        # Simplified to binary classification: Normal (0) vs Stress (1)
        # This fixes the "Invalid classes" error when only 2 classes are present
        self.regime_labels = ['Normal', 'Stress']
        self.label_params = None
        self._is_fitted = False
        
        # Auto-fit on instantiation
        self._fit(returns, extra_features, n_splits, test_size, min_cv_accuracy, verbose)
        
    def _compute_rolling_vol(self, returns: pd.Series, window: int) -> pd.Series:
        return returns.rolling(window).std() * np.sqrt(252)
    
    def _compute_rolling_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        cum_returns = (1 + returns).rolling(window).apply(lambda x: np.prod(x), raw=True) - 1
        running_max = cum_returns.rolling(window, min_periods=1).max()
        drawdown = (cum_returns - running_max) / (1 + running_max)
        return drawdown
    
    def _compute_vol_of_vol(self, returns: pd.Series, vol_window: int, vov_window: int) -> pd.Series:
        rolling_vol = self._compute_rolling_vol(returns, vol_window)
        vol_of_vol = rolling_vol.rolling(vov_window).std()
        return vol_of_vol
    
    def make_features(
        self,
        returns: pd.Series,
        extra_features: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DatetimeIndex]:
        df = pd.DataFrame(index=returns.index)
        
        df['vol_5d'] = self._compute_rolling_vol(returns, 5)
        df['vol_20d'] = self._compute_rolling_vol(returns, 20)
        df['vol_60d'] = self._compute_rolling_vol(returns, 60)
        
        df['mean_ret_20d'] = returns.rolling(20).mean()
        df['abs_ret_mean_20d'] = returns.abs().rolling(20).mean()
        df['downside_ret_20d'] = returns[returns < 0].rolling(20).mean().fillna(0)
        
        df['drawdown_20d'] = self._compute_rolling_drawdown(returns, 20)
        df['drawdown_60d'] = self._compute_rolling_drawdown(returns, 60)
        df['max_drawdown_60d'] = returns.rolling(60).apply(
            lambda x: self._compute_rolling_drawdown(pd.Series(x), len(x)).min(), 
            raw=False
        )
        
        df['vol_of_vol_20_5'] = self._compute_vol_of_vol(returns, 5, 20)
        
        df['skew_20d'] = returns.rolling(20).skew()
        df['kurt_20d'] = returns.rolling(20).kurt()
        
        df['vol_ratio_5_20'] = df['vol_5d'] / df['vol_20d']
        df['vol_ratio_20_60'] = df['vol_20d'] / df['vol_60d']
        
        df['large_move_5d'] = (returns.abs().rolling(5).max() > returns.abs().quantile(0.95)).astype(int)
        
        if extra_features is not None:
            extra_features_aligned = extra_features.reindex(df.index)
            df = pd.concat([df, extra_features_aligned], axis=1)
        
        labels = self._create_labels(returns, df)
        
        valid_idx = df.notna().all(axis=1) & labels.notna()
        X = df[valid_idx].copy()
        y = labels[valid_idx].copy()
        aligned_index = returns.index[valid_idx]
        
        self.feature_names = X.columns.tolist()
        
        return X, y, aligned_index
    
    def _create_labels(self, returns: pd.Series, features_df: pd.DataFrame) -> pd.Series:
        labels = pd.Series(index=returns.index, dtype='object')
        labels[:] = 'Normal'
        
        vol_20d = features_df['vol_20d']
        drawdown_20d = features_df['drawdown_20d']
        
        stress_vol_threshold = vol_20d.quantile(self.stress_vol_percentile / 100)
        calm_vol_threshold = vol_20d.quantile(self.calm_vol_percentile / 100)
        
        self.label_params = {
            'stress_vol_threshold': stress_vol_threshold,
            'calm_vol_threshold': calm_vol_threshold,
            'stress_drawdown_threshold': self.stress_drawdown_threshold
        }
        
        # Binary classification: Normal (default) vs Stress
        # Removed 'Calm' category to fix "Invalid classes" error
        stress_mask = (
            (vol_20d > stress_vol_threshold) | 
            (drawdown_20d < self.stress_drawdown_threshold)
        )
        labels[stress_mask] = 'Stress'
        
        # No 'Calm' category - everything non-Stress is 'Normal'
        # This ensures binary classes [0, 1] for XGBClassifier
        
        return labels
    
    def _fit(
        self,
        returns: pd.Series,
        extra_features: Optional[pd.DataFrame] = None,
        n_splits: int = 5,
        test_size: float = 0.2,
        min_cv_accuracy: float = 0.85,
        verbose: bool = True
    ) -> None:
        X_full, y_full, aligned_index_full = self.make_features(returns, extra_features)
        
        # Store for later queries
        self._returns = returns
        self._extra_features = extra_features
        self._X_full = X_full
        self._y_full = y_full
        self._aligned_index = aligned_index_full
        
        split_idx = int(len(X_full) * (1 - test_size))
        X = X_full.iloc[:split_idx]
        y = y_full.iloc[:split_idx]
        self.X_test = X_full.iloc[split_idx:]
        self.y_test = y_full.iloc[split_idx:]
        self.test_index = aligned_index_full[split_idx:]
        
        label_mapping = {label: idx for idx, label in enumerate(self.regime_labels)}
        y_encoded = y.map(label_mapping)
        
        sample_weights = compute_sample_weight('balanced', y_encoded)
        
        self.model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            eval_metric='mlogloss'
        )
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y_encoded.iloc[train_idx], y_encoded.iloc[val_idx]
            w_train = sample_weights[train_idx]
            
            self.model.fit(X_train, y_train, sample_weight=w_train, verbose=False)
            
            y_pred = self.model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            cv_scores.append(acc)
        
        # Validate model quality: mean >= 0.95 AND all folds >= min_cv_accuracy
        mean_acc = np.mean(cv_scores)
        min_fold_acc = min(cv_scores)
        
        if mean_acc < 0.95:
            raise ValueError(
                f"Model quality check failed: Mean CV accuracy {mean_acc:.4f} "
                f"is below required threshold 0.95. "
                f"Consider adjusting labeling parameters or using more data."
            )
        
        if min_fold_acc < min_cv_accuracy:
            raise ValueError(
                f"Model quality check failed: Minimum fold accuracy {min_fold_acc:.4f} "
                f"is below required threshold {min_cv_accuracy:.4f}. "
                f"Consider adjusting labeling parameters or using more data."
            )
        
        if verbose:
            print(f"Model quality validated: Mean={mean_acc:.4f} >= 0.95, All folds >= {min_cv_accuracy:.4f}")
        
        self.model.fit(X, y_encoded, sample_weight=sample_weights, verbose=False)
        
        self._is_fitted = True
        
        if verbose:
            print("\nModel training complete.")
            
            y_test_encoded = self.y_test.map(label_mapping)
            y_pred_test = self.model.predict(self.X_test)
            test_accuracy = accuracy_score(y_test_encoded, y_pred_test)
            print(f"\nHeld-out Test Set Accuracy: {test_accuracy:.4f}")
    
    def isBlocked(self, date: str = None) -> bool:
        if date is None:
            features = self._X_full.iloc[-1]
        else:
            query_dt = pd.to_datetime(date)
            latest_dt = self._aligned_index[-1]
            
            if query_dt > latest_dt:
                features = self._X_full.iloc[-1]
            else:
                # Historical: find features for that date
                if query_dt not in self._aligned_index:
                    valid_dates = self._aligned_index[self._aligned_index >= query_dt]
                    if len(valid_dates) == 0:
                        raise ValueError(f"Date {date} is after all available data")
                    query_dt = valid_dates[0]
                
                idx = self._aligned_index.get_loc(query_dt)
                features = self._X_full.iloc[idx]
        
        # Predict
        X_input = features[self.feature_names].values.reshape(1, -1)
        probs = self.model.predict_proba(X_input)[0]
        prob_dict = {label: prob for label, prob in zip(self.regime_labels, probs)}
        
        return prob_dict['Stress'] > self.block_threshold

'''
usage:

    from preprocess_data import get_log_returns
    
    returns = get_log_returns("s&p_data.csv")
    
    blocker = RegimeBlockerXGB(
        returns=returns,
        stress_vol_percentile=90.0,
        stress_drawdown_threshold=-0.05,
        calm_vol_percentile=25.0,
        block_threshold=0.5,
        random_state=42
    )
    
    or even:
    blocker = RegimeBlockerXGB(returns)
    
    
    print(f"\nBlocked on 2024-06-01: {blocker.isBlocked('2024-06-01')}")
    print(f"Current/Latest - Blocked: {blocker.isBlocked()}")
'''