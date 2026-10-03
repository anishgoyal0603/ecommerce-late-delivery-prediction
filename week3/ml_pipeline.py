"""
Week 3 - ML model development and evaluation pipeline (template)

This is the code version of the plan in the Week 3 report. It builds a
scikit-learn Pipeline (preprocessing + model), compares several models with
time-based cross-validation, tunes the best one, picks a decision threshold
and evaluates it once on a hold-out test set.

Usage:
    python ml_pipeline.py --demo                # runs on a simulated sample
    python ml_pipeline.py --data orders_clean.csv

NOTE: the --demo option only checks that the pipeline runs end to end.
Its scores mean nothing because the data is simulated.

Author: Anish Goyal
"""
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import TimeSeriesSplit, cross_validate, RandomizedSearchCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             average_precision_score, precision_recall_curve, f1_score)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

TARGET = "is_late"
DATE_COL = "order_purchase_timestamp"
NUMERIC = ["distance_km", "estimated_days", "approval_hours", "price", "freight_value",
           "freight_ratio", "product_weight_g", "product_volume_cm3", "n_items", "n_sellers"]
SKEWED = ["price", "freight_value", "product_weight_g", "product_volume_cm3", "distance_km"]
CATEGORICAL = ["customer_region", "payment_type", "purchase_month", "purchase_weekday",
               "same_state", "is_peak_season"]


# ---------------------------------------------------------------- data
def make_demo_data(n=6000, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime("2017-01-01") + pd.to_timedelta(np.sort(rng.integers(0, 600, n)), unit="D")
    df = pd.DataFrame({
        DATE_COL: dates,
        "distance_km": rng.gamma(2, 300, n),
        "estimated_days": np.clip(rng.normal(23, 8, n), 5, 60).round(),
        "approval_hours": rng.exponential(10, n),
        "price": rng.lognormal(4.5, 0.7, n),
        "product_weight_g": rng.lognormal(7, 1, n),
        "product_volume_cm3": rng.lognormal(8.5, 1, n),
        "n_items": rng.choice([1, 1, 1, 2, 3], n),
        "n_sellers": rng.choice([1, 1, 1, 1, 2], n),
        "customer_region": rng.choice(["Southeast", "South", "Northeast", "Centre-West", "North"], n,
                                      p=[.6, .15, .13, .07, .05]),
        "payment_type": rng.choice(["credit_card", "boleto", "voucher", "debit_card"], n, p=[.74, .2, .04, .02]),
        "same_state": rng.choice([0, 1], n, p=[.64, .36]),
    })
    df["freight_value"] = np.clip(8 + df["distance_km"] * 0.02 + rng.normal(0, 4, n), 1, None)
    df["freight_ratio"] = df["freight_value"] / df["price"]
    df["purchase_month"] = df[DATE_COL].dt.month
    df["purchase_weekday"] = df[DATE_COL].dt.weekday
    df["is_peak_season"] = df["purchase_month"].isin([11, 12]).astype(int)
    logit = (-3.3 + df["distance_km"] / 800 - df["estimated_days"] / 35 + df["is_peak_season"] * 0.8
             + df["customer_region"].isin(["Northeast", "North"]) * 0.7)
    df[TARGET] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df.loc[rng.choice(n, 80, replace=False), "product_weight_g"] = np.nan
    return df


def time_split(df, valid_frac=0.15, test_frac=0.15):
    """Split by date: oldest -> train, middle -> validation, newest -> test."""
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    n = len(df)
    i_val, i_test = int(n * (1 - valid_frac - test_frac)), int(n * (1 - test_frac))
    return df.iloc[:i_val], df.iloc[i_val:i_test], df.iloc[i_test:]


# ---------------------------------------------------------------- preprocessing
def build_preprocessor():
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
    ])
    # log only applied to the skewed columns, the rest are only imputed + scaled
    plain_numeric = [c for c in NUMERIC if c not in SKEWED]
    plain_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("skewed", numeric_pipe, SKEWED),
        ("numeric", plain_pipe, plain_numeric),
        ("categorical", categorical_pipe, CATEGORICAL),
    ])


# ---------------------------------------------------------------- models
def candidate_models(pos_weight):
    models = {
        "Dummy (baseline)": DummyClassifier(strategy="stratified", random_state=0),
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=0),
        "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=0),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=5,
                                          scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=0)
    else:
        models["Gradient Boosting (HistGB)"] = HistGradientBoostingClassifier(class_weight="balanced",
                                                                              random_state=0)
    return models


def compare_models(X, y, cv):
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    rows = []
    for name, model in candidate_models(pos_weight).items():
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
        scores = cross_validate(pipe, X, y, cv=cv,
                                scoring=["roc_auc", "average_precision", "recall", "precision", "f1"])
        rows.append({"model": name, **{k.replace("test_", ""): v.mean()
                                        for k, v in scores.items() if k.startswith("test_")}})
    return pd.DataFrame(rows).sort_values("average_precision", ascending=False).round(3)


def tune_best(X, y, cv):
    pos_weight = (y == 0).sum() / max((y == 1).sum(), 1)
    if HAS_XGB:
        model = XGBClassifier(scale_pos_weight=pos_weight, eval_metric="aucpr", random_state=0)
        grid = {"model__n_estimators": [200, 400, 800], "model__learning_rate": [0.01, 0.05, 0.1],
                "model__max_depth": [3, 5, 7], "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0], "model__min_child_weight": [1, 5, 10]}
    else:
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=0)
        grid = {"model__learning_rate": [0.03, 0.05, 0.1], "model__max_depth": [3, 5, None],
                "model__max_leaf_nodes": [15, 31, 63], "model__min_samples_leaf": [20, 50, 100],
                "model__l2_regularization": [0.0, 0.1, 1.0]}
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)])
    search = RandomizedSearchCV(pipe, grid, n_iter=15, scoring="average_precision", cv=cv,
                                random_state=0, n_jobs=-1)
    search.fit(X, y)
    return search


def pick_threshold(model, X_val, y_val, min_precision=0.30):
    """Highest recall threshold that still keeps precision above a minimum."""
    prob = model.predict_proba(X_val)[:, 1]
    prec, rec, thr = precision_recall_curve(y_val, prob)
    ok = np.where(prec[:-1] >= min_precision)[0]
    if len(ok) == 0:
        return 0.5
    best = ok[np.argmax(rec[:-1][ok])]
    return float(thr[best])


# ---------------------------------------------------------------- main
def main(df):
    train, valid, test = time_split(df)
    features = NUMERIC + CATEGORICAL
    X_tr, y_tr = train[features], train[TARGET]
    X_va, y_va = valid[features], valid[TARGET]
    X_te, y_te = test[features], test[TARGET]
    print(f"train {len(train)} | valid {len(valid)} | test {len(test)} | late rate {y_tr.mean():.1%}")

    cv = TimeSeriesSplit(n_splits=4)
    print("\nModel comparison (time-series CV on train):")
    print(compare_models(X_tr, y_tr, cv).to_string(index=False))

    search = tune_best(X_tr, y_tr, cv)
    print("\nBest params:", search.best_params_)
    best = search.best_estimator_

    threshold = pick_threshold(best, X_va, y_va)
    print(f"Chosen threshold (from validation set): {threshold:.3f}")

    # final model on train + valid, evaluated ONCE on the untouched test set
    best.fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
    prob = best.predict_proba(X_te)[:, 1]
    pred = (prob >= threshold).astype(int)
    print("\nTest set results")
    print(f"ROC-AUC: {roc_auc_score(y_te, prob):.3f}   PR-AUC: {average_precision_score(y_te, prob):.3f}"
          f"   F1: {f1_score(y_te, pred):.3f}")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, target_names=["on time", "late"], digits=3))

    joblib.dump({"pipeline": best, "threshold": threshold, "features": features}, "late_delivery_model.pkl")
    print("Saved: late_delivery_model.pkl")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to cleaned order-level CSV")
    ap.add_argument("--demo", action="store_true", help="run on simulated data")
    args = ap.parse_args()
    if args.demo or not args.data:
        print("Running in DEMO mode on simulated data (scores are not meaningful)\n")
        data = make_demo_data()
    else:
        data = pd.read_csv(args.data, parse_dates=[DATE_COL])
    main(data)
