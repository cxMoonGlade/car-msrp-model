from __future__ import annotations

import argparse
import json
import os
import os
import itertools
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

_xla_flags = os.environ.get("XLA_FLAGS", "")
if "--xla_gpu_autotune_level" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (_xla_flags + " --xla_gpu_autotune_level=0").strip()

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax

jax.config.update("jax_enable_x64", True)

try:
    jax.config.update("jax_platform_name", "gpu")
except Exception:
    # Fall back to default platform if GPU is unavailable.
    pass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BRAND_COUNTRY_MAP: Dict[str, str] = {
    "ACURA": "Asian",
    "ALFA ROMEO": "European",
    "ASTON MARTIN": "European",
    "AUDI": "European",
    "BENTLEY": "European",
    "BMW": "European",
    "BUICK": "American",
    "CADILLAC": "American",
    "CHEVROLET": "American",
    "CHRYSLER": "American",
    "CITROEN": "European",
    "DAEWOO": "Asian",
    "DAIHATSU": "Asian",
    "DODGE": "American",
    "FERRARI": "European",
    "FIAT": "European",
    "FORD": "American",
    "GAZ": "European",
    "GMC": "American",
    "GREATWALL": "Asian",
    "HAVAL": "Asian",
    "HONDA": "Asian",
    "HUMMER": "American",
    "HYUNDAI": "Asian",
    "INFINITI": "Asian",
    "ISUZU": "Asian",
    "JAGUAR": "European",
    "JEEP": "American",
    "KIA": "Asian",
    "LAMBORGHINI": "European",
    "LANCIA": "European",
    "LAND ROVER": "European",
    "LEXUS": "Asian",
    "LINCOLN": "American",
    "MASERATI": "European",
    "MAZDA": "Asian",
    "MERCEDES-BENZ": "European",
    "MERCURY": "American",
    "MINI": "European",
    "MITSUBISHI": "Asian",
    "MOSKVICH": "European",
    "NISSAN": "Asian",
    "OPEL": "European",
    "PEUGEOT": "European",
    "PONTIAC": "American",
    "PORSCHE": "European",
    "RENAULT": "European",
    "ROLLS-ROYCE": "European",
    "ROVER": "European",
    "SAAB": "European",
    "SATURN": "American",
    "SCION": "Asian",
    "SEAT": "European",
    "SKODA": "European",
    "SSANGYONG": "Asian",
    "SUBARU": "Asian",
    "SUZUKI": "Asian",
    "TESLA": "American",
    "TOYOTA": "Asian",
    "UAZ": "European",
    "VAZ": "European",
    "VOLKSWAGEN": "European",
    "VOLVO": "European",
    "ZAZ": "European",
}

PREMIUM_MANUFACTURERS = {
    "ASTON MARTIN",
    "AUDI",
    "BENTLEY",
    "BMW",
    "CADILLAC",
    "FERRARI",
    "INFINITI",
    "JAGUAR",
    "LAMBORGHINI",
    "LAND ROVER",
    "LEXUS",
    "MASERATI",
    "MERCEDES-BENZ",
    "PORSCHE",
    "ROLLS-ROYCE",
    "TESLA",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regression model for car MSRP prediction."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("csv-20251115T062129Z-1-001/csv/car_price_prediction_CLEANED.csv"),
        help="Path to the cleaned car dataset CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model/artifacts"),
        help="Directory to store trained model and metrics.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Fraction of samples allocated to the test split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for the train/test split.",
    )
    parser.add_argument(
        "--ridge-alpha",
        type=float,
        default=10.0,
        help="Regularization strength for Ridge regression.",
    )
    parser.add_argument(
        "--ridge-alpha-grid",
        type=str,
        default="0.1,1,5,10,25",
        help="Comma-separated list of ridge alpha values to evaluate via CV.",
    )
    parser.add_argument(
        "--model-type",
        choices=["ridge", "neural", "lightgbm", "catboost"],
        default="ridge",
        help="Regressor type to train.",
    )
    parser.add_argument(
        "--nn-hidden-dims",
        type=str,
        default="1024,512,256",
        help="Comma-separated hidden layer sizes for the neural model.",
    )
    parser.add_argument(
        "--nn-dropout",
        type=float,
        default=0.1,
        help="Dropout rate for neural model.",
    )
    parser.add_argument(
        "--nn-learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for neural model.",
    )
    parser.add_argument(
        "--nn-weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2) for neural model.",
    )
    parser.add_argument(
        "--nn-epochs",
        type=int,
        default=50,
        help="Epochs for neural training.",
    )
    parser.add_argument(
        "--nn-batch-size",
        type=int,
        default=512,
        help="Batch size for neural training.",
    )
    parser.add_argument(
        "--nn-val-fraction",
        type=float,
        default=0.1,
        help="Fraction of training data used for neural validation.",
    )
    parser.add_argument(
        "--nn-seed",
        type=int,
        default=0,
        help="Random seed used for neural weight init/shuffling.",
    )
    parser.add_argument(
        "--nn-sweep-learning-rates",
        type=str,
        default="",
        help="Comma-separated learning rates for neural sweeps.",
    )
    parser.add_argument(
        "--nn-sweep-dropouts",
        type=str,
        default="",
        help="Comma-separated dropout rates for neural sweeps.",
    )
    parser.add_argument(
        "--nn-sweep-weight-decays",
        type=str,
        default="",
        help="Comma-separated weight decay values for neural sweeps.",
    )
    parser.add_argument(
        "--nn-sweep-batch-sizes",
        type=str,
        default="",
        help="Comma-separated batch sizes for neural sweeps.",
    )
    parser.add_argument(
        "--nn-sweep-hidden-dims",
        type=str,
        default="",
        help="Semicolon-separated hidden layer options (each option comma-separated).",
    )
    parser.add_argument(
        "--lgb-learning-rate-grid",
        type=str,
        default="0.05,0.1",
        help="Comma-separated learning rates for LightGBM.",
    )
    parser.add_argument(
        "--lgb-num-leaves-grid",
        type=str,
        default="31,63,127",
        help="Comma-separated num_leaves values for LightGBM.",
    )
    parser.add_argument(
        "--lgb-max-depth-grid",
        type=str,
        default="None,12,20",
        help="Comma-separated max_depth values (use 'None' for unlimited).",
    )
    parser.add_argument(
        "--lgb-n-estimators-grid",
        type=str,
        default="400,800",
        help="Comma-separated boosting rounds for LightGBM.",
    )
    parser.add_argument(
        "--lgb-subsample-grid",
        type=str,
        default="0.8,1.0",
        help="Comma-separated subsample values for LightGBM.",
    )
    parser.add_argument(
        "--lgb-colsample-bytree-grid",
        type=str,
        default="0.8,1.0",
        help="Comma-separated colsample_bytree values for LightGBM.",
    )
    parser.add_argument(
        "--lgb-min-child-samples-grid",
        type=str,
        default="20,40",
        help="Comma-separated min_child_samples values for LightGBM.",
    )
    parser.add_argument(
        "--lgb-max-combinations",
        type=int,
        default=20,
        help="Max LightGBM hyperparameter combinations to evaluate (<=0 for all).",
    )
    parser.add_argument(
        "--cat-learning-rate-grid",
        type=str,
        default="0.03,0.05",
        help="Comma-separated learning rates for CatBoost.",
    )
    parser.add_argument(
        "--cat-depth-grid",
        type=str,
        default="6,8",
        help="Comma-separated depth values for CatBoost.",
    )
    parser.add_argument(
        "--cat-iterations-grid",
        type=str,
        default="1000,2000",
        help="Comma-separated iteration counts for CatBoost.",
    )
    parser.add_argument(
        "--cat-l2-leaf-reg-grid",
        type=str,
        default="3,5",
        help="Comma-separated L2 leaf regularization values for CatBoost.",
    )
    parser.add_argument(
        "--cat-bagging-temperature-grid",
        type=str,
        default="0.5,1.0",
        help="Comma-separated bagging temperature values for CatBoost.",
    )
    parser.add_argument(
        "--cat-border-count-grid",
        type=str,
        default="128",
        help="Comma-separated border count values for CatBoost.",
    )
    parser.add_argument(
        "--cat-loss-function",
        type=str,
        default="RMSE",
        help='CatBoost loss function (e.g. "RMSE", "MAE", "Quantile:alpha=0.5").',
    )
    parser.add_argument(
        "--cat-eval-metric",
        type=str,
        default="RMSE",
        help="CatBoost evaluation metric.",
    )
    parser.add_argument(
        "--cat-max-combinations",
        type=int,
        default=16,
        help="Max CatBoost hyperparameter combinations to evaluate (<=0 for all).",
    )
    return parser.parse_args()


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")
    df = pd.read_csv(path)
    return df


def normalize_numeric(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(r"[^0-9.]", "", regex=True)
        .replace("", np.nan)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def add_brand_country(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).str.upper().str.strip()
    return normalized.map(BRAND_COUNTRY_MAP).fillna("Other")


def winsorize_series(
    series: pd.Series, lower: float = 0.01, upper: float = 0.99
) -> pd.Series:
    non_null = series.dropna()
    if non_null.empty:
        return series
    lower_bound = non_null.quantile(lower)
    upper_bound = non_null.quantile(upper)
    return series.clip(lower_bound, upper_bound)


def normalize_doors(series: pd.Series) -> pd.Series:
    extracted = series.astype(str).str.extract(r"(\d+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def bucket_prod_year(series: pd.Series) -> pd.Series:
    valid_years = series.dropna()
    if valid_years.empty:
        return pd.Series(pd.NA, index=series.index)
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())
    start = min_year - (min_year % 5)
    bins = list(range(start, max_year + 5, 5))
    if len(bins) < 2:
        bins = [min_year - 1, max_year + 1]
    labels = [f"{bins[i]}-{bins[i+1]-1}" for i in range(len(bins) - 1)]
    return pd.cut(series, bins=bins, labels=labels, include_lowest=True)


def parse_alpha_grid(alpha_string: str) -> List[float]:
    values: List[float] = []
    for token in alpha_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"Invalid alpha value '{token}' in ridge-alpha-grid.")
    return values


def parse_hidden_dims(hidden_string: str) -> Tuple[int, ...]:
    dims: List[int] = []
    for token in hidden_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            dims.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid hidden dimension '{token}'.")
    if not dims:
        dims = [512, 256]
    return tuple(dims)


def parse_float_list(list_string: str) -> List[float]:
    values: List[float] = []
    for token in list_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(float(token))
        except ValueError:
            raise ValueError(f"Invalid float value '{token}'.")
    return values


def parse_int_list(list_string: str) -> List[int]:
    values: List[int] = []
    for token in list_string.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.append(int(token))
        except ValueError:
            raise ValueError(f"Invalid integer value '{token}'.")
    return values


def parse_optional_int_list(list_string: str) -> List[int | None]:
    values: List[int | None] = []
    for token in list_string.split(","):
        token = token.strip()
        if not token:
            continue
        if token.lower() == "none":
            values.append(None)
        else:
            try:
                values.append(int(token))
            except ValueError:
                raise ValueError(f"Invalid integer/None value '{token}'.")
    return values


def parse_hidden_dim_options(option_string: str) -> List[Tuple[int, ...]]:
    options: List[Tuple[int, ...]] = []
    for chunk in option_string.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        options.append(parse_hidden_dims(chunk))
    return options


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["ID"])
    cleaned["Price"] = pd.to_numeric(cleaned["Price"], errors="coerce")
    cleaned["Levy"] = normalize_numeric(cleaned["Levy"]).replace(0, np.nan)
    cleaned["Mileage"] = normalize_numeric(cleaned["Mileage"])
    cleaned["Engine_volume"] = normalize_numeric(cleaned["Engine_volume"])
    cleaned["Prod_year"] = pd.to_numeric(cleaned["Prod_year"], errors="coerce")
    cleaned["Cylinders"] = pd.to_numeric(cleaned["Cylinders"], errors="coerce")
    cleaned["Airbags"] = pd.to_numeric(cleaned["Airbags"], errors="coerce")
    cleaned["BrandCountry"] = add_brand_country(cleaned["Manufacturer"])

    cleaned = cleaned.dropna(subset=["Price", "Prod_year", "Mileage", "Engine_volume"])

    cleaned["Price"] = winsorize_series(cleaned["Price"])
    cleaned["Levy"] = winsorize_series(cleaned["Levy"])
    cleaned["Mileage"] = winsorize_series(cleaned["Mileage"])

    cleaned["Doors_numeric"] = normalize_doors(cleaned["Doors"])
    cleaned["Mileage_log"] = np.log1p(cleaned["Mileage"])
    cleaned["Engine_volume_log"] = np.log1p(cleaned["Engine_volume"])
    cleaned["Prod_year_bin"] = bucket_prod_year(cleaned["Prod_year"])
    model_freq = cleaned["Model"].value_counts()
    cleaned["Model_frequent"] = cleaned["Model"].map(model_freq)
    reference_year = int(cleaned["Prod_year"].max())
    cleaned["Vehicle_age"] = (reference_year - cleaned["Prod_year"]).clip(lower=0)
    cleaned["Mileage_per_year"] = cleaned["Mileage"] / np.maximum(
        cleaned["Vehicle_age"], 1
    )
    cleaned["Mileage_per_year"] = cleaned["Mileage_per_year"].replace(
        [np.inf, -np.inf], np.nan
    )
    cleaned["Mileage_per_year_log"] = np.log1p(cleaned["Mileage_per_year"])
    cylinder_denominator = np.where(cleaned["Cylinders"] > 0, cleaned["Cylinders"], np.nan)
    cleaned["Engine_per_cylinder"] = cleaned["Engine_volume"] / cylinder_denominator
    cleaned["Engine_per_cylinder"] = cleaned["Engine_per_cylinder"].replace(
        [np.inf, -np.inf], np.nan
    )

    premium_mask = (
        cleaned["Manufacturer"].astype(str).str.upper().isin(PREMIUM_MANUFACTURERS)
    )
    cleaned["Is_premium_brand"] = premium_mask.astype(np.int8)

    year_min = cleaned["Prod_year"].min()
    year_max = cleaned["Prod_year"].max()
    if year_max > year_min:
        year_norm = (cleaned["Prod_year"] - year_min) / (year_max - year_min)
    else:
        year_norm = 0.0
    cleaned["Prod_year_sin"] = np.sin(2 * np.pi * year_norm)
    cleaned["Prod_year_cos"] = np.cos(2 * np.pi * year_norm)

    cleaned["Age_x_Mileage_log"] = cleaned["Vehicle_age"] * cleaned["Mileage_log"]
    cleaned["Premium_x_Engine_log"] = cleaned["Is_premium_brand"] * cleaned["Engine_volume_log"]
    cylinders_safe = np.where(cleaned["Cylinders"] > 0, cleaned["Cylinders"], np.nan)
    cleaned["Age_per_cylinder"] = cleaned["Vehicle_age"] / cylinders_safe

    return cleaned


def build_preprocessor(
    numeric_features: List[str], categorical_features: List[str]
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def ensure_dense_array(array) -> np.ndarray:
    if hasattr(array, "toarray"):
        array = array.toarray()
    return np.asarray(array, dtype=np.float64)


def add_bias_column(features: np.ndarray) -> np.ndarray:
    bias = np.ones((features.shape[0], 1), dtype=features.dtype)
    return np.hstack([features, bias])


class PriceMLP(nn.Module):
    hidden_dims: Tuple[int, ...]
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.gelu(x)
            if self.dropout_rate > 0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(1)(x)
        return jnp.squeeze(x, axis=-1)


def create_train_state(
    model: PriceMLP,
    rng: jax.Array,
    feature_dim: int,
    learning_rate: float,
    weight_decay: float,
) -> train_state.TrainState:
    params = model.init(
        rng, jnp.ones((1, feature_dim), dtype=jnp.float32), train=True
    )["params"]
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def iter_minibatches(
    X: np.ndarray, y: np.ndarray, batch_size: int, rng: np.random.Generator
):
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        yield X[batch_idx], y[batch_idx]


@jax.jit
def train_step(
    state: train_state.TrainState,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    dropout_rng: jax.Array,
) -> Tuple[train_state.TrainState, jnp.ndarray]:
    def loss_fn(params):
        preds = state.apply_fn(
            {"params": params}, batch_x, train=True, rngs={"dropout": dropout_rng}
        )
        loss = jnp.mean((preds - batch_y) ** 2)
        return loss

    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)


def predict_step(
    params,
    apply_fn,
    inputs: jnp.ndarray,
) -> jnp.ndarray:
    return apply_fn({"params": params}, inputs, train=False)


def predict_log_prices(
    model: PriceMLP,
    params,
    X: np.ndarray,
    batch_size: int = 4096,
) -> np.ndarray:
    preds: List[np.ndarray] = []
    for start in range(0, X.shape[0], batch_size):
        batch = jnp.asarray(X[start : start + batch_size], dtype=jnp.float32)
        preds.append(np.asarray(predict_step(params, model.apply, batch)))
    return np.concatenate(preds, axis=0)


def create_train_val_split(
    X: np.ndarray,
    y_log: np.ndarray,
    y_raw: np.ndarray,
    val_fraction: float,
    random_state: int,
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if not 0 < val_fraction < 0.5:
        val_fraction = 0.1
    rng = np.random.default_rng(random_state)
    indices = np.arange(X.shape[0])
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return (
        X[train_idx],
        y_log[train_idx],
        y_raw[train_idx],
    ), (
        X[val_idx],
        y_log[val_idx],
        y_raw[val_idx],
    )


def train_neural_model(
    X_train: np.ndarray,
    y_train_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    y_val_raw: np.ndarray,
    model: PriceMLP,
    config: Dict[str, float],
) -> Tuple[dict, List[float], dict]:
    rng = jax.random.PRNGKey(config["seed"])
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        model,
        init_rng,
        feature_dim=X_train.shape[1],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    val_mse_history: List[float] = []
    best_params = state.params
    best_val_mse = np.inf

    for epoch in range(config["epochs"]):
        batch_rng = np.random.default_rng(config["seed"] + epoch)
        for batch_x, batch_y in iter_minibatches(
            X_train, y_train_log, config["batch_size"], batch_rng
        ):
            rng, dropout_rng = jax.random.split(rng)
            batch_x_jax = jnp.asarray(batch_x, dtype=jnp.float32)
            batch_y_jax = jnp.asarray(batch_y, dtype=jnp.float32)
            state = train_step(state, batch_x_jax, batch_y_jax, dropout_rng)

        val_preds_log = predict_log_prices(model, state.params, X_val, batch_size=4096)
        val_preds = np.expm1(val_preds_log)
        val_mse = float(np.mean((y_val_raw - val_preds) ** 2))
        val_mse_history.append(val_mse)
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_params = state.params

    numpy_params = jax.tree_util.tree_map(lambda x: np.asarray(x), best_params)
    model_state = {
        "params": numpy_params,
        "hidden_dims": model.hidden_dims,
        "dropout_rate": model.dropout_rate,
    }
    return model_state, val_mse_history, best_params


def train_ridge_closed_form(
    X: np.ndarray, y_log: np.ndarray, alpha: float
) -> np.ndarray:
    xtx = X.T @ X
    xty = X.T @ y_log
    reg = alpha * np.eye(X.shape[1], dtype=np.float64)
    weights = np.linalg.solve(xtx + reg, xty)
    return weights


def predict_from_weights(X: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return X @ weights


def cross_validate_ridge(
    X: np.ndarray,
    y_log: np.ndarray,
    y_raw: np.ndarray,
    alpha_grid: List[float],
    random_state: int,
    n_splits: int = 5,
) -> Tuple[float, np.ndarray]:
    if not alpha_grid:
        raise ValueError("Alpha grid must contain at least one value.")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_alpha = None
    best_scores = None
    best_mean = np.inf

    for alpha in alpha_grid:
        fold_scores: List[float] = []
        for train_idx, val_idx in kf.split(X):
            weights = train_ridge_closed_form(X[train_idx], y_log[train_idx], alpha)
            val_log_pred = predict_from_weights(X[val_idx], weights)
            val_pred = np.expm1(val_log_pred)
            mse = float(np.mean((y_raw[val_idx] - val_pred) ** 2))
            fold_scores.append(mse)

        fold_scores_arr = np.asarray(fold_scores, dtype=np.float32)
        mean_score = float(np.mean(fold_scores_arr))
        if mean_score < best_mean:
            best_mean = mean_score
            best_alpha = alpha
            best_scores = fold_scores_arr

    if best_alpha is None or best_scores is None:
        raise RuntimeError("Failed to select a ridge model. Check alpha grid.")

    return float(best_alpha), best_scores


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Robustified MAPE: avoid tiny denominators
    denom = np.maximum(np.abs(y_true), 5000.0)  # floor at e.g. $5k
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)

    return {
        "mse": float(mse),
        "rmse": rmse,
        "mae": mae,
        "r2": float(r2),
        "mape": mape,
    }


def compute_outlier_mask(y: np.ndarray, percentile: float = 0.98) -> np.ndarray:
    threshold = np.quantile(y, percentile)
    return y > threshold, threshold


def create_plots(
    predictions: pd.DataFrame, cv_scores: np.ndarray, output_dir: Path
) -> None:
    sns.set_theme(style="whitegrid")

    # Actual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(
        predictions["y_true"],
        predictions["y_pred"],
        alpha=0.3,
        edgecolor="none",
        s=20,
    )
    max_val = max(predictions["y_true"].max(), predictions["y_pred"].max())
    min_val = min(predictions["y_true"].min(), predictions["y_pred"].min())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    ax.set_xlabel("Actual Price (USD)")
    ax.set_ylabel("Predicted Price (USD)")
    ax.set_title("Actual vs. Predicted Prices")
    fig.tight_layout()
    fig.savefig(output_dir / "pred_vs_actual.png", dpi=300)
    plt.close(fig)

    # Residual Histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(predictions["residual"], bins=40, kde=True, ax=ax)
    ax.set_xlabel("Residual (Actual - Predicted)")
    ax.set_title("Residual Distribution")
    fig.tight_layout()
    fig.savefig(output_dir / "residual_histogram.png", dpi=300)
    plt.close(fig)

    # Residual vs Predicted
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        x="y_pred",
        y="residual",
        data=predictions,
        ax=ax,
        alpha=0.3,
        edgecolor="none",
        s=20,
    )
    ax.axhline(0, color="red", linestyle="--")
    ax.set_xlabel("Predicted Price (USD)")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs. Predicted Values")
    fig.tight_layout()
    fig.savefig(output_dir / "residuals_vs_pred.png", dpi=300)
    plt.close(fig)

    # Cross-validation RMSE
    fig, ax = plt.subplots(figsize=(5, 4))
    folds = np.arange(1, len(cv_scores) + 1)
    ax.bar(folds, cv_scores, color="#1f77b4")
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("MSE")
    ax.set_title("Cross-Validation MSE per Fold")
    fig.tight_layout()
    fig.savefig(output_dir / "cv_mse_bar.png", dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataframe(args.data)
    df = clean_dataframe(raw_df)

    feature_columns = [
        "Levy",
        "Manufacturer",
        "Model",
        "Prod_year",
        "Prod_year_bin",
        "Category",
        "Leather_interior",
        "Fuel_type",
        "Engine_volume",
        "Mileage",
        "Cylinders",
        "Gear_box_type",
        "Drive_wheels",
        "Doors_numeric",
        "Wheel",
        "Color",
        "Airbags",
        "BrandCountry",
        "Mileage_log",
        "Engine_volume_log",
        "Model_frequent",
        "Vehicle_age",
        "Mileage_per_year",
        "Mileage_per_year_log",
        "Engine_per_cylinder",
        "Is_premium_brand",
        "Prod_year_sin",
        "Prod_year_cos",
        "Age_x_Mileage_log",
        "Premium_x_Engine_log",
        "Age_per_cylinder",
    ]

    target_column = "Price"

    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing expected columns: {missing_features}")

    X = df[feature_columns]
    y = df[target_column]

    numeric_features = [
        "Levy",
        "Prod_year",
        "Engine_volume",
        "Mileage",
        "Cylinders",
        "Airbags",
        "Doors_numeric",
        "Mileage_log",
        "Engine_volume_log",
        "Model_frequent",
        "Vehicle_age",
        "Mileage_per_year",
        "Mileage_per_year_log",
        "Engine_per_cylinder",
        "Is_premium_brand",
        "Prod_year_sin",
        "Prod_year_cos",
        "Age_x_Mileage_log",
        "Premium_x_Engine_log",
        "Age_per_cylinder",
    ]

    categorical_features = [
        "Manufacturer",
        "Model",
        "Category",
        "Leather_interior",
        "Fuel_type",
        "Gear_box_type",
        "Drive_wheels",
        "Prod_year_bin",
        "Wheel",
        "Color",
        "BrandCountry",
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    X_train_proc = ensure_dense_array(preprocessor.fit_transform(X_train))
    X_test_proc = ensure_dense_array(preprocessor.transform(X_test))

    if args.model_type == "ridge":
        alpha_grid = parse_alpha_grid(args.ridge_alpha_grid) or [args.ridge_alpha]

        X_train_aug = add_bias_column(X_train_proc)
        X_test_aug = add_bias_column(X_test_proc)

        y_train_array = y_train.to_numpy(dtype=np.float64)
        y_test_array = y_test.to_numpy(dtype=np.float64)
        y_train_log = np.log1p(y_train_array)

        best_alpha, cv_mse_scores = cross_validate_ridge(
            X_train_aug,
            y_train_log,
            y_train_array,
            alpha_grid,
            args.random_state,
        )

        weights = train_ridge_closed_form(X_train_aug, y_train_log, best_alpha)
        test_log_pred = predict_from_weights(X_test_aug, weights)
        test_pred = np.expm1(test_log_pred)

        baseline_pred = np.full_like(y_test_array, y_train_array.mean(), dtype=np.float64)
        baseline_metrics = compute_regression_metrics(y_test_array, baseline_pred)
        print("Baseline metrics (predict train mean):", baseline_metrics)

        metrics = compute_regression_metrics(y_test_array, test_pred)
        metrics["baseline_metrics"] = baseline_metrics
        outlier_mask, outlier_threshold = compute_outlier_mask(y_test_array)
        if np.any(~outlier_mask):
            non_outlier_metrics = compute_regression_metrics(
                y_test_array[~outlier_mask], test_pred[~outlier_mask]
            )
            metrics["non_outlier_metrics"] = non_outlier_metrics
            print(
                f"Non-outlier metrics (<= {outlier_threshold:.0f}):",
                non_outlier_metrics,
            )
        metrics["outlier_threshold"] = float(outlier_threshold)
        metrics["outlier_fraction"] = float(outlier_mask.mean())
        metrics["cv_mse_mean"] = float(np.mean(cv_mse_scores))
        metrics["cv_mse_std"] = float(np.std(cv_mse_scores))
        metrics["cv_mse_scores"] = [float(score) for score in cv_mse_scores]
        metrics["train_size"] = int(X_train.shape[0])
        metrics["test_size"] = int(X_test.shape[0])
        metrics["model_type"] = "jax_ridge"
        metrics["ridge_alpha"] = float(best_alpha)
        metrics["ridge_alpha_grid"] = alpha_grid
        metrics["backend"] = jax.default_backend()

        plot_scores = cv_mse_scores
        predictions = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": test_pred,
            }
        )
        predictions["is_outlier"] = outlier_mask
        model_artifact = {
            "weights": np.asarray(weights),
            "alpha": float(best_alpha),
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "bias_term": True,
        }
        model_filename = (
            f"jax_ridge_alpha_{best_alpha}".replace(".", "_") + ".joblib"
        )
    elif args.model_type == "neural":
        X_train_nn = X_train_proc.astype(np.float32)
        X_test_nn = X_test_proc.astype(np.float32)
        y_train_array = y_train.to_numpy(dtype=np.float32)
        y_test_array = y_test.to_numpy(dtype=np.float32)
        y_train_log = np.log1p(y_train_array)

        (X_nn_train, y_nn_train_log, y_nn_train_raw), (
            X_nn_val,
            y_nn_val_log,
            y_nn_val_raw,
        ) = create_train_val_split(
            X_train_nn,
            y_train_log,
            y_train_array,
            args.nn_val_fraction,
            args.random_state,
        )

        hidden_options = parse_hidden_dim_options(args.nn_sweep_hidden_dims)
        if not hidden_options:
            hidden_options = [parse_hidden_dims(args.nn_hidden_dims)]
        lr_grid = parse_float_list(args.nn_sweep_learning_rates) or [
            args.nn_learning_rate
        ]
        dropout_grid = parse_float_list(args.nn_sweep_dropouts) or [args.nn_dropout]
        weight_decay_grid = parse_float_list(args.nn_sweep_weight_decays) or [
            args.nn_weight_decay
        ]
        batch_grid = parse_int_list(args.nn_sweep_batch_sizes) or [args.nn_batch_size]

        sweep_results: List[Dict[str, float | List[int]]] = []
        best_combo = None

        for hidden_dims, lr, dropout, weight_decay, batch_size in itertools.product(
            hidden_options, lr_grid, dropout_grid, weight_decay_grid, batch_grid
        ):
            nn_config = {
                "learning_rate": lr,
                "weight_decay": weight_decay,
                "epochs": args.nn_epochs,
                "batch_size": int(batch_size),
                "seed": args.nn_seed or args.random_state,
            }
            price_mlp = PriceMLP(hidden_dims=hidden_dims, dropout_rate=dropout)

            model_state, val_mse_history, best_params = train_neural_model(
                X_nn_train,
                y_nn_train_log,
                X_nn_val,
                y_nn_val_log,
                y_nn_val_raw,
                price_mlp,
                nn_config,
            )

            best_val_mse = float(np.min(val_mse_history))
            sweep_results.append(
                {
                    "hidden_dims": list(hidden_dims),
                    "learning_rate": float(lr),
                    "weight_decay": float(weight_decay),
                    "dropout_rate": float(dropout),
                    "batch_size": int(batch_size),
                    "val_mse_best": best_val_mse,
                    "val_mse_last": float(val_mse_history[-1]),
                }
            )

            if best_combo is None or best_val_mse < best_combo["best_val_mse"]:
                best_combo = {
                    "hidden_dims": hidden_dims,
                    "learning_rate": lr,
                    "weight_decay": weight_decay,
                    "dropout_rate": dropout,
                    "batch_size": int(batch_size),
                    "val_mse_history": val_mse_history,
                    "best_params": best_params,
                    "model_state": model_state,
                    "best_val_mse": best_val_mse,
                }

        if best_combo is None:
            raise RuntimeError("No neural configurations were evaluated.")

        best_price_mlp = PriceMLP(
            hidden_dims=best_combo["hidden_dims"],
            dropout_rate=best_combo["dropout_rate"],
        )

        test_log_pred = predict_log_prices(
            best_price_mlp, best_combo["best_params"], X_test_nn
        )
        test_pred = np.expm1(test_log_pred)

        baseline_pred = np.full_like(
            y_test_array, y_train_array.mean(dtype=np.float64), dtype=np.float64
        )
        baseline_metrics = compute_regression_metrics(
            y_test_array.astype(np.float64), baseline_pred
        )
        print("Baseline metrics (predict train mean):", baseline_metrics)

        metrics = compute_regression_metrics(y_test_array, test_pred)
        metrics["baseline_metrics"] = baseline_metrics
        metrics["val_mse_history"] = best_combo["val_mse_history"]
        metrics["val_mse_best"] = best_combo["best_val_mse"]
        metrics["val_mse_last"] = float(best_combo["val_mse_history"][-1])
        metrics["train_size"] = int(X_train.shape[0])
        metrics["test_size"] = int(X_test.shape[0])
        metrics["model_type"] = "jax_neural_mlp"
        metrics["hidden_dims"] = list(best_combo["hidden_dims"])
        metrics["dropout_rate"] = float(best_combo["dropout_rate"])
        metrics["learning_rate"] = float(best_combo["learning_rate"])
        metrics["weight_decay"] = float(best_combo["weight_decay"])
        metrics["batch_size"] = int(best_combo["batch_size"])
        metrics["epochs"] = args.nn_epochs
        metrics["backend"] = jax.default_backend()
        if len(sweep_results) > 1:
            metrics["sweep_results"] = sweep_results
        outlier_mask, outlier_threshold = compute_outlier_mask(y_test_array)
        if np.any(~outlier_mask):
            non_outlier_metrics = compute_regression_metrics(
                y_test_array[~outlier_mask], test_pred[~outlier_mask]
            )
            metrics["non_outlier_metrics"] = non_outlier_metrics
            print(
                f"Non-outlier metrics (<= {outlier_threshold:.0f}):",
                non_outlier_metrics,
            )
        metrics["outlier_threshold"] = float(outlier_threshold)
        metrics["outlier_fraction"] = float(outlier_mask.mean())

        plot_scores = np.asarray(best_combo["val_mse_history"], dtype=np.float32)
        predictions = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": test_pred,
            }
        )
        predictions["is_outlier"] = outlier_mask
        model_artifact = {
            "nn_model": best_combo["model_state"],
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "val_mse_history": best_combo["val_mse_history"],
            "best_config": {
                "hidden_dims": list(best_combo["hidden_dims"]),
                "learning_rate": float(best_combo["learning_rate"]),
                "weight_decay": float(best_combo["weight_decay"]),
                "dropout_rate": float(best_combo["dropout_rate"]),
                "batch_size": int(best_combo["batch_size"]),
                "epochs": args.nn_epochs,
            },
            "sweep_results": sweep_results,
        }
        dims_label = "_".join(str(dim) for dim in best_combo["hidden_dims"])
        model_filename = (
            f"jax_mlp_{dims_label}_dropout_{best_combo['dropout_rate']}".replace(
                ".", "_"
            )
            + ".joblib"
        )
    elif args.model_type == "lightgbm":
        X_train_lgb = X_train_proc.astype(np.float32)
        X_test_lgb = X_test_proc.astype(np.float32)
        feature_names = [f"f_{i}" for i in range(X_train_lgb.shape[1])]
        X_train_lgb_df = pd.DataFrame(X_train_lgb, columns=feature_names)
        X_test_lgb_df = pd.DataFrame(X_test_lgb, columns=feature_names)
        y_train_array = y_train.to_numpy(dtype=np.float32)
        y_test_array = y_test.to_numpy(dtype=np.float32)
        y_train_log = np.log1p(y_train_array)

        learning_rates = parse_float_list(args.lgb_learning_rate_grid)
        num_leaves_list = parse_int_list(args.lgb_num_leaves_grid)
        max_depths = parse_optional_int_list(args.lgb_max_depth_grid)
        n_estimators_list = parse_int_list(args.lgb_n_estimators_grid)
        subsamples = parse_float_list(args.lgb_subsample_grid)
        colsample_bytree_list = parse_float_list(args.lgb_colsample_bytree_grid)
        min_child_samples_list = parse_int_list(args.lgb_min_child_samples_grid)

        if not (
            learning_rates
            and num_leaves_list
            and max_depths
            and n_estimators_list
            and subsamples
            and colsample_bytree_list
            and min_child_samples_list
        ):
            raise ValueError("LightGBM grids must not be empty.")

        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
        best_params = None
        best_mean = np.inf
        best_scores = None

        combo_iter = list(
            itertools.product(
            learning_rates,
            num_leaves_list,
            max_depths,
            n_estimators_list,
            subsamples,
            colsample_bytree_list,
            min_child_samples_list,
            )
        )
        if args.lgb_max_combinations > 0 and len(combo_iter) > args.lgb_max_combinations:
            rng = np.random.default_rng(args.random_state)
            selected_idx = rng.choice(
                len(combo_iter), size=args.lgb_max_combinations, replace=False
            )
            combo_iter = [combo_iter[i] for i in selected_idx]

        for (
            lr,
            num_leaves,
            max_depth,
            n_estimators,
            subsample,
            colsample_bytree,
            min_child_samples,
        ) in combo_iter:
            fold_scores: List[float] = []
            for train_idx, val_idx in kf.split(X_train_lgb):
                booster = LGBMRegressor(
                    boosting_type="gbdt",
                    objective="regression",
                    device="gpu",
                    gpu_platform_id=0,
                    gpu_device_id=0,
                    learning_rate=lr,
                    num_leaves=num_leaves,
                    max_depth=-1 if max_depth is None else max_depth,
                    n_estimators=n_estimators,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    min_child_samples=min_child_samples,
                    random_state=args.random_state,
                    n_jobs=-1,
                    verbose=-1,
                )
                booster.fit(
                    X_train_lgb_df.iloc[train_idx],
                    y_train_log[train_idx],
                )
                val_log_pred = booster.predict(X_train_lgb_df.iloc[val_idx])
                val_pred = np.expm1(val_log_pred)
                mse = float(np.mean((y_train_array[val_idx] - val_pred) ** 2))
                fold_scores.append(mse)

            fold_scores_arr = np.asarray(fold_scores, dtype=np.float32)
            mean_score = float(np.mean(fold_scores_arr))
            if mean_score < best_mean:
                best_mean = mean_score
                best_params = {
                    "learning_rate": lr,
                    "num_leaves": num_leaves,
                    "max_depth": max_depth,
                    "n_estimators": n_estimators,
                    "subsample": subsample,
                    "colsample_bytree": colsample_bytree,
                    "min_child_samples": min_child_samples,
                }
                best_scores = fold_scores_arr

        if best_params is None or best_scores is None:
            raise RuntimeError("Failed to select a LightGBM model.")

        final_booster = LGBMRegressor(
            boosting_type="gbdt",
            objective="regression",
            device="gpu",
            gpu_platform_id=0,
            gpu_device_id=0,
            learning_rate=best_params["learning_rate"],
            num_leaves=best_params["num_leaves"],
            max_depth=-1 if best_params["max_depth"] is None else best_params["max_depth"],
            n_estimators=best_params["n_estimators"],
            subsample=best_params["subsample"],
            colsample_bytree=best_params["colsample_bytree"],
            min_child_samples=best_params["min_child_samples"],
            random_state=args.random_state,
            n_jobs=-1,
            verbose=-1,
        )
        final_booster.fit(X_train_lgb_df, y_train_log)

        test_log_pred = final_booster.predict(X_test_lgb_df)
        test_pred = np.expm1(test_log_pred)

        baseline_pred = np.full_like(
            y_test_array, y_train_array.mean(dtype=np.float64), dtype=np.float64
        )
        baseline_metrics = compute_regression_metrics(
            y_test_array.astype(np.float64), baseline_pred
        )
        print("Baseline metrics (predict train mean):", baseline_metrics)

        metrics = compute_regression_metrics(y_test_array, test_pred)
        metrics["baseline_metrics"] = baseline_metrics
        metrics["cv_mse_mean"] = float(np.mean(best_scores))
        metrics["cv_mse_std"] = float(np.std(best_scores))
        metrics["cv_mse_scores"] = [float(score) for score in best_scores]
        metrics["train_size"] = int(X_train.shape[0])
        metrics["test_size"] = int(X_test.shape[0])
        metrics["model_type"] = "lightgbm"
        metrics["lightgbm_best_params"] = {
            key: (value if value is not None else "None") for key, value in best_params.items()
        }
        metrics["backend"] = jax.default_backend()
        metrics["lightgbm_combinations_evaluated"] = int(len(combo_iter))
        outlier_mask, outlier_threshold = compute_outlier_mask(y_test_array)
        if np.any(~outlier_mask):
            non_outlier_metrics = compute_regression_metrics(
                y_test_array[~outlier_mask], test_pred[~outlier_mask]
            )
            metrics["non_outlier_metrics"] = non_outlier_metrics
            print(
                f"Non-outlier metrics (<= {outlier_threshold:.0f}):",
                non_outlier_metrics,
            )
        metrics["outlier_threshold"] = float(outlier_threshold)
        metrics["outlier_fraction"] = float(outlier_mask.mean())

        plot_scores = best_scores
        predictions = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": test_pred,
            }
        )
        predictions["is_outlier"] = outlier_mask
        predictions["is_outlier"] = outlier_mask
        depth_label = (
            "none" if best_params["max_depth"] is None else str(best_params["max_depth"])
        )
        model_artifact = {
            "lightgbm_model": final_booster,
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "best_params": best_params,
        }
        model_filename = (
            f"lightgbm_lr_{best_params['learning_rate']}_leaves_{best_params['num_leaves']}_depth_{depth_label}_estimators_{best_params['n_estimators']}"
            .replace(".", "_")
            + ".joblib"
        )
    elif args.model_type == "catboost":
        X_train_cb = X_train_proc.astype(np.float32)
        X_test_cb = X_test_proc.astype(np.float32)
        y_train_array = y_train.to_numpy(dtype=np.float32)
        y_test_array = y_test.to_numpy(dtype=np.float32)
        y_train_log = np.log1p(y_train_array)

        lr_grid = parse_float_list(args.cat_learning_rate_grid)
        depth_grid = parse_int_list(args.cat_depth_grid)
        iterations_grid = parse_int_list(args.cat_iterations_grid)
        l2_leaf_grid = parse_float_list(args.cat_l2_leaf_reg_grid)
        bagging_temp_grid = parse_float_list(args.cat_bagging_temperature_grid)
        border_count_grid = parse_int_list(args.cat_border_count_grid)

        if not (
            lr_grid
            and depth_grid
            and iterations_grid
            and l2_leaf_grid
            and bagging_temp_grid
            and border_count_grid
        ):
            raise ValueError("CatBoost grids must not be empty.")

        combo_iter = list(
            itertools.product(
                lr_grid,
                depth_grid,
                iterations_grid,
                l2_leaf_grid,
                bagging_temp_grid,
                border_count_grid,
            )
        )
        if args.cat_max_combinations > 0 and len(combo_iter) > args.cat_max_combinations:
            rng = np.random.default_rng(args.random_state + 1)
            selected_idx = rng.choice(
                len(combo_iter), size=args.cat_max_combinations, replace=False
            )
            combo_iter = [combo_iter[i] for i in selected_idx]

        kf = KFold(n_splits=5, shuffle=True, random_state=args.random_state)
        best_params = None
        best_scores = None
        best_mean = np.inf

        for (
            lr,
            depth,
            iterations,
            l2_leaf_reg,
            bagging_temp,
            border_count,
        ) in combo_iter:
            fold_scores: List[float] = []
            for train_idx, val_idx in kf.split(X_train_cb):
                booster = CatBoostRegressor(
                    task_type="GPU",
                    devices="0",
                    loss_function=args.cat_loss_function,
                    eval_metric=args.cat_eval_metric,
                    learning_rate=lr,
                    depth=depth,
                    iterations=iterations,
                    l2_leaf_reg=l2_leaf_reg,
                    bagging_temperature=bagging_temp,
                    border_count=border_count,
                    random_seed=args.random_state,
                    verbose=False,
                    allow_writing_files=False,
                )
                booster.fit(
                    X_train_cb[train_idx],
                    y_train_log[train_idx],
                )
                val_log_pred = booster.predict(X_train_cb[val_idx])
                val_pred = np.expm1(val_log_pred)
                mse = float(np.mean((y_train_array[val_idx] - val_pred) ** 2))
                fold_scores.append(mse)

            fold_scores_arr = np.asarray(fold_scores, dtype=np.float32)
            mean_score = float(np.mean(fold_scores_arr))
            if mean_score < best_mean:
                best_mean = mean_score
                best_scores = fold_scores_arr
                best_params = {
                    "learning_rate": lr,
                    "depth": depth,
                    "iterations": iterations,
                    "l2_leaf_reg": l2_leaf_reg,
                    "bagging_temperature": bagging_temp,
                    "border_count": border_count,
                }

        if best_params is None or best_scores is None:
            raise RuntimeError("Failed to select a CatBoost model.")

        final_booster = CatBoostRegressor(
            task_type="GPU",
            devices="0",
            loss_function=args.cat_loss_function,
            eval_metric=args.cat_eval_metric,
            learning_rate=best_params["learning_rate"],
            depth=best_params["depth"],
            iterations=best_params["iterations"],
            l2_leaf_reg=best_params["l2_leaf_reg"],
            bagging_temperature=best_params["bagging_temperature"],
            border_count=best_params["border_count"],
            random_seed=args.random_state,
            verbose=False,
            allow_writing_files=False,
        )
        final_booster.fit(X_train_cb, y_train_log)
        test_log_pred = final_booster.predict(X_test_cb)
        test_pred = np.expm1(test_log_pred)

        baseline_pred = np.full_like(
            y_test_array, y_train_array.mean(dtype=np.float64), dtype=np.float64
        )
        baseline_metrics = compute_regression_metrics(
            y_test_array.astype(np.float64), baseline_pred
        )
        print("Baseline metrics (predict train mean):", baseline_metrics)

        metrics = compute_regression_metrics(y_test_array, test_pred)
        metrics["baseline_metrics"] = baseline_metrics
        metrics["cv_mse_mean"] = float(np.mean(best_scores))
        metrics["cv_mse_std"] = float(np.std(best_scores))
        metrics["cv_mse_scores"] = [float(score) for score in best_scores]
        metrics["train_size"] = int(X_train.shape[0])
        metrics["test_size"] = int(X_test.shape[0])
        metrics["model_type"] = "catboost_gpu"
        metrics["catboost_best_params"] = best_params
        metrics["catboost_loss_function"] = args.cat_loss_function
        metrics["catboost_eval_metric"] = args.cat_eval_metric
        metrics["backend"] = "gpu"
        metrics["catboost_combinations_evaluated"] = int(len(combo_iter))
        outlier_mask, outlier_threshold = compute_outlier_mask(y_test_array)
        if np.any(~outlier_mask):
            non_outlier_metrics = compute_regression_metrics(
                y_test_array[~outlier_mask], test_pred[~outlier_mask]
            )
            metrics["non_outlier_metrics"] = non_outlier_metrics
            print(
                f"Non-outlier metrics (<= {outlier_threshold:.0f}):",
                non_outlier_metrics,
            )
        metrics["outlier_threshold"] = float(outlier_threshold)
        metrics["outlier_fraction"] = float(outlier_mask.mean())

        plot_scores = best_scores
        predictions = pd.DataFrame(
            {
                "y_true": y_test.to_numpy(),
                "y_pred": test_pred,
            }
        )
        model_artifact = {
            "catboost_model": final_booster,
            "preprocessor": preprocessor,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
            "feature_columns": feature_columns,
            "best_params": best_params,
        }
        model_filename = (
            f"catboost_lr_{best_params['learning_rate']}_depth_{best_params['depth']}_iter_{best_params['iterations']}"
            .replace(".", "_")
            + ".joblib"
        )
    else:
        raise ValueError(f"Unsupported model_type '{args.model_type}'")

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))

    predictions["residual"] = predictions["y_true"] - predictions["y_pred"]
    predictions.to_csv(args.output_dir / "test_predictions.csv", index=False)

    create_plots(predictions, np.asarray(plot_scores), args.output_dir)

    joblib.dump(model_artifact, args.output_dir / model_filename)

    print("Model training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

