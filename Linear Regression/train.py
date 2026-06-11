"""
train.py
--------
Run this script to retrain the pipeline and save a fresh model.pkl.

Usage:
    python train.py
    python train.py --data path/to/custom.csv
"""

import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
NUMERIC_FEATURES     = ["cylinders", "displacement", "horsepower",
                         "weight", "acceleration", "model year"]
CATEGORICAL_FEATURES = ["origin"]
TARGET               = "mpg"
DROP_COLS            = ["car name"]


def load_data(path: str) -> tuple:
    df = pd.read_csv(path)
    df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
    df = df.drop(columns=DROP_COLS, errors="ignore")
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot",  OneHotEncoder(drop="first", sparse_output=False,
                                  handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,     NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model",        Ridge()),
    ])


def train(data_path: str, output_path: str = "model.pkl"):
    print(f"Loading data from: {data_path}")
    X, y = load_data(data_path)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    pipeline = build_pipeline()
    param_grid = {"model__alpha": [0.01, 0.1, 1, 10, 50, 100, 200]}
    gs = GridSearchCV(pipeline, param_grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print("\n── Evaluation ──────────────────────────────")
    print(f"  Best alpha : {gs.best_params_['model__alpha']}")
    print(f"  MAE        : {mae:.3f} mpg")
    print(f"  RMSE       : {rmse:.3f} mpg")
    print(f"  R²         : {r2:.4f}")
    print("────────────────────────────────────────────")

    joblib.dump(best, output_path)
    print(f"\n✅ Model saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Auto MPG regression pipeline.")
    parser.add_argument("--data",   default="auto-mpg.csv", help="Path to CSV dataset")
    parser.add_argument("--output", default="model.pkl",    help="Output path for model")
    args = parser.parse_args()
    train(args.data, args.output)
