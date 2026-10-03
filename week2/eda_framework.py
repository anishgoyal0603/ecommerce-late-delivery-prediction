"""
Reusable EDA framework (Week 2)

These functions work on any pandas DataFrame, not only the e-commerce data.
The idea is to run the same checks in the same order on every dataset:

    1. overview()          -> shape, dtypes, duplicates
    2. missing_report()    -> missing values per column
    3. outlier_report()    -> IQR based outlier count for numeric columns
    4. plot_numeric()      -> histogram + boxplot for each numeric column
    5. plot_categorical()  -> bar chart of top categories
    6. plot_vs_target()    -> how each feature relates to the target
    7. plot_correlation()  -> correlation heatmap

Author: Anish Goyal
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="Set2")


def split_columns(df, max_unique_for_cat=20):
    """Separate columns into numeric, categorical and datetime."""
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime"]).columns.tolist()
    numeric_cols, categorical_cols = [], []
    for col in df.columns:
        if col in datetime_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > max_unique_for_cat:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)
    return numeric_cols, categorical_cols, datetime_cols


def overview(df):
    print(f"Rows: {df.shape[0]:,}   Columns: {df.shape[1]}")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "unique": df.nunique(),
        "missing": df.isna().sum(),
    })
    return summary


def missing_report(df):
    miss = df.isna().sum()
    pct = (miss / len(df) * 100).round(2)
    report = pd.DataFrame({"missing": miss, "percent": pct})
    report = report[report["missing"] > 0].sort_values("percent", ascending=False)

    # simple rule of thumb I am following for treatment
    def suggest(p):
        if p < 5:
            return "drop rows / simple impute"
        if p < 30:
            return "impute (median / mode / group-wise)"
        return "consider dropping column"
    report["suggestion"] = report["percent"].apply(suggest)
    return report


def outlier_report(df, cols=None):
    cols = cols or split_columns(df)[0]
    rows = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = ((df[col] < low) | (df[col] > high)).sum()
        rows.append([col, round(low, 2), round(high, 2), n_out, round(n_out / len(df) * 100, 2)])
    return pd.DataFrame(rows, columns=["column", "lower", "upper", "outliers", "percent"])


def _save(fig, out_dir, name):
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, name), dpi=150, bbox_inches="tight")


def plot_numeric(df, cols=None, out_dir=None):
    cols = cols or split_columns(df)[0]
    for col in cols:
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {col}")
        sns.boxplot(x=df[col], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        _save(fig, out_dir, f"num_{col}.png")
        plt.show() if not out_dir else plt.close(fig)


def plot_categorical(df, cols=None, top_n=10, out_dir=None):
    cols = cols or split_columns(df)[1]
    for col in cols:
        counts = df[col].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=counts.values, y=counts.index.astype(str), ax=ax)
        ax.set_title(f"Top {top_n} values of {col}")
        ax.set_xlabel("count")
        plt.tight_layout()
        _save(fig, out_dir, f"cat_{col}.png")
        plt.show() if not out_dir else plt.close(fig)


def plot_vs_target(df, target, out_dir=None):
    """Bivariate view of every feature against a binary target."""
    num_cols, cat_cols, _ = split_columns(df.drop(columns=[target]))
    for col in num_cols:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        sns.boxplot(data=df, x=target, y=col, ax=ax)
        ax.set_title(f"{col} by {target}")
        _save(fig, out_dir, f"target_{col}.png")
        plt.show() if not out_dir else plt.close(fig)
    for col in cat_cols:
        rate = df.groupby(col)[target].mean().sort_values(ascending=False).head(10) * 100
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=rate.values, y=rate.index.astype(str), ax=ax)
        ax.set_xlabel(f"% {target}")
        ax.set_title(f"{target} rate by {col}")
        _save(fig, out_dir, f"target_{col}.png")
        plt.show() if not out_dir else plt.close(fig)


def plot_correlation(df, out_dir=None):
    corr = df.select_dtypes(include=np.number).corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation heatmap")
    _save(fig, out_dir, "correlation.png")
    plt.show() if not out_dir else plt.close(fig)
    return corr


def run_full_eda(df, target=None, out_dir="eda_output"):
    """Runs the full framework in one go and saves all charts to out_dir."""
    print(overview(df), "\n")
    print(missing_report(df), "\n")
    print(outlier_report(df), "\n")
    plot_numeric(df, out_dir=out_dir)
    plot_categorical(df, out_dir=out_dir)
    if target:
        plot_vs_target(df, target, out_dir=out_dir)
    plot_correlation(df, out_dir=out_dir)
    print(f"Charts saved in: {out_dir}/")
