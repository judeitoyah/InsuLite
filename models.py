"""Baseline regression models, the shared evaluation helper, and training loop.

Mirrors notebook section 7 (Baseline Model Development): Linear Regression,
SVR, Random Forest, XGBoost, LightGBM and a Deep FNN, all scored with the
same metric set so they land in one comparable results table.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from .config import SEED


@dataclass
class EvalResult:
    model: str
    r2: float
    rmse: float
    mae: float
    mape: float
    train_time_s: float
    fitted: Any = field(repr=False)

    def as_dict(self) -> dict:
        return {
            "Model": self.model,
            "R2": self.r2,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "MAPE (%)": self.mape,
            "Train time (s)": self.train_time_s,
        }


def evaluate(name: str, y_true: np.ndarray, y_pred: np.ndarray, train_time_s: float = 0.0, fitted: Any = None) -> EvalResult:
    """Compute the standard metric set (R², RMSE, MAE, MAPE) for one model."""
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return EvalResult(name, float(r2), rmse, mae, mape, round(train_time_s, 3), fitted)


def build_dfnn(n_features: int, lr: float = 1e-3):
    """Deep feedforward network: 256-128-64 with batchnorm/dropout, MSE loss.

    Imported lazily so environments without TensorFlow can still run the rest
    of the pipeline (the Deep FNN baseline is the only stage that needs it).
    """
    from tensorflow import keras
    from tensorflow.keras import layers

    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(256, activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def train_linear_regression(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    t0 = time.time()
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate("Linear Regression", y_test, y_pred, time.time() - t0, model)


def train_svr(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    scaler = StandardScaler()
    Xs_tr = scaler.fit_transform(X_train)
    Xs_te = scaler.transform(X_test)

    t0 = time.time()
    model = SVR(kernel="rbf", C=100, epsilon=0.1, gamma="scale")
    model.fit(Xs_tr, y_train)
    y_pred = model.predict(Xs_te)
    return evaluate("SVR (RBF)", y_test, y_pred, time.time() - t0, model)


def train_random_forest(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    t0 = time.time()
    model = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2,
                                  random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate("Random Forest", y_test, y_pred, time.time() - t0, model)


def train_xgboost(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    t0 = time.time()
    model = xgb.XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="reg:squarederror", eval_metric="rmse",
        random_state=seed, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    y_pred = model.predict(X_test)
    return evaluate("XGBoost", y_test, y_pred, time.time() - t0, model)


def train_lightgbm(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=seed, n_jobs=-1, verbose=-1,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    y_pred = model.predict(X_test)
    return evaluate("LightGBM", y_test, y_pred, time.time() - t0, model)


def train_deep_fnn(X_train, y_train, X_test, y_test, seed: int = SEED) -> EvalResult:
    from tensorflow.keras import callbacks

    scaler = StandardScaler()
    Xnn_tr = scaler.fit_transform(X_train)
    Xnn_te = scaler.transform(X_test)

    t0 = time.time()
    model = build_dfnn(Xnn_tr.shape[1])
    cb = [
        callbacks.EarlyStopping(patience=25, restore_best_weights=True, monitor="val_loss"),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=10, verbose=0),
    ]
    model.fit(Xnn_tr, y_train, validation_split=0.15, epochs=300, batch_size=32,
              callbacks=cb, verbose=0)
    y_pred = model.predict(Xnn_te, verbose=0).flatten()
    return evaluate("Deep FNN", y_test, y_pred, time.time() - t0, model)


BASELINE_TRAINERS = {
    "Linear Regression": train_linear_regression,
    "SVR (RBF)": train_svr,
    "Random Forest": train_random_forest,
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
}


def train_baselines(X_train, y_train, X_test, y_test, seed: int = SEED,
                     include_deep_fnn: bool = False) -> list[EvalResult]:
    """Train every baseline model and return their evaluation results.

    Deep FNN is opt-in (`include_deep_fnn=True`) since it requires TensorFlow
    and is by far the slowest baseline to train.
    """
    results = [trainer(X_train, y_train, X_test, y_test, seed=seed)
               for trainer in BASELINE_TRAINERS.values()]

    if include_deep_fnn:
        results.append(train_deep_fnn(X_train, y_train, X_test, y_test, seed=seed))

    return results
