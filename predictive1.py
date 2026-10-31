import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

from scipy.stats import pearsonr
from scipy.cluster.hierarchy import linkage, fcluster

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    log_loss,
    roc_auc_score
)

"""Startup funding analysis project.

The script loads and cleans the dataset, explores funding patterns, compares
machine-learning models, groups similar records, and saves results in a
folder named ``results`` beside this Python file.
"""

warnings.filterwarnings("ignore")

# =========================================================
# Basic project settings
# =========================================================

DATA_FILE = Path(
    r"C:\Users\hp\Downloads\archive (12)\india_startup_funding_sample.csv"
)

BASE_DIR = Path(__file__).resolve().parent
RESULT_DIR = BASE_DIR / "results"

RESULT_DIR.mkdir(exist_ok=True)

RANDOM_STATE = 42

print("=" * 65)
print(" Startup Funding Prediction and Investment Analytics")
print("=" * 65)


# =========================================================
# Step 1: Open the startup funding dataset
# =========================================================

def load_data():

    if not DATA_FILE.exists():
        print("\nERROR: Dataset not found.")
        print("Expected location:", DATA_FILE)
        return None

    try:
        df = pd.read_csv(DATA_FILE, low_memory=False)
    except Exception as error:
        print("Could not read the dataset:", error)
        return None

    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    aliases = {
        "company": "company_name",
        "startup": "company_name",
        "startup_name": "company_name",
        "date": "announced_date",
        "funding_date": "announced_date",
        "funding_year": "year",
        "funding_amount": "amount_usd",
        "amount": "amount_usd",
        "funding_amount_usd": "amount_usd",
        "amount_in_usd": "amount_usd",
        "investment_amount": "amount_usd",
        "funding_stage": "stage",
        "industry": "sector",
        "location": "city",
        "city_name": "city",
        "number_of_investors": "investor_count",
        "num_investors": "investor_count"
    }

    for old, new in aliases.items():
        if old in df.columns and new not in df.columns:
            df.rename(columns={old: new}, inplace=True)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)

        df[col] = df[col].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    print("\nDataset loaded successfully.")
    print("-" * 50)
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("\nColumn names:")
    print(df.columns.tolist())

    return df


# =========================================================
# Step 2: Get a quick look at the data
# =========================================================

def inspect_data(df):

    print("\nFIRST FIVE RECORDS")
    print("-" * 50)
    print(df.head().to_string(index=False))

    print("\nDATA TYPES")
    print("-" * 50)
    print(df.dtypes)

    print("\nMISSING VALUES")
    print("-" * 50)
    print(df.isnull().sum().sort_values(ascending=False))

    print("\nDUPLICATE ROWS")
    print("-" * 50)
    print(df.duplicated().sum())

    print("\nSTATISTICAL SUMMARY")
    print("-" * 50)
    print(df.describe(include="all").to_string())

    df.isnull().sum().to_csv(
        RESULT_DIR / "missing_values_before_cleaning.csv"
    )


# =========================================================
# Step 3: Clean the data before analysis
# =========================================================

def clean_data(df):

    df = df.copy()
    initial_rows = len(df)

    df.drop_duplicates(inplace=True)

    if "announced_date" in df.columns:
        df["announced_date"] = pd.to_datetime(
            df["announced_date"],
            errors="coerce"
        )

    numeric_columns = [
        "amount_usd",
        "founded_year",
        "year",
        "investor_count",
        "news_mentions"
    ]

    for col in numeric_columns:

        if col not in df.columns:
            continue

        if df[col].dtype == "object":

            df[col] = df[col].astype(str).str.replace(
                r"[$,\s]",
                "",
                regex=True
            )

            df[col] = df[col].replace(
                ["", "nan", "None", "NA", "N/A", "-"],
                np.nan
            )

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    if "announced_date" in df.columns:

        if "year" not in df.columns:
            df["year"] = df["announced_date"].dt.year

        else:
            df["year"] = df["year"].fillna(
                df["announced_date"].dt.year
            )

    if "year" in df.columns and "founded_year" in df.columns:

        df["startup_age"] = (
            df["year"] - df["founded_year"]
        )

        df.loc[
            (df["startup_age"] < 0)
            | (df["startup_age"] > 100),
            "startup_age"
        ] = np.nan

    if "amount_usd" in df.columns:

        df.loc[
            df["amount_usd"] <= 0,
            "amount_usd"
        ] = np.nan

        df.dropna(
            subset=["amount_usd"],
            inplace=True
        )

    if "company_name" in df.columns:
        df.dropna(
            subset=["company_name"],
            inplace=True
        )

    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.to_csv(
        RESULT_DIR / "cleaned_dataset.csv",
        index=False
    )

    df.isnull().sum().to_csv(
        RESULT_DIR / "missing_values_after_cleaning.csv"
    )

    print("\nData cleaning is complete.")
    print("-" * 50)
    print("Rows before cleaning:", initial_rows)
    print("Rows after cleaning:", len(df))
    print("Rows removed:", initial_rows - len(df))
    print("Cleaned dataset saved.")

    return df


# =========================================================
# Step 4: Explore patterns in the dataset
# =========================================================

def perform_eda(df):

    print("\nExploring the dataset")
    print("=" * 50)

    # Count the sectors that appear most often in the dataset.

    if "sector" in df.columns:

        sector_counts = (
            df["sector"]
            .fillna("Unknown")
            .value_counts()
            .head(10)
        )

        print("\nTOP 10 SECTORS")
        print(sector_counts)

        sector_counts.to_csv(
            RESULT_DIR / "top_sectors.csv"
        )

        plt.figure(figsize=(10, 6))

        sector_counts.sort_values().plot(
            kind="barh"
        )

        plt.title("Top 10 Startup Sectors")
        plt.xlabel("Number of Funding Rounds")
        plt.ylabel("Sector")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / "top_sectors.png"
        )

        plt.show()

    # See how the number of funding rounds changes over time.

    if "year" in df.columns:

        yearly = (
            df.dropna(subset=["year"])
            .groupby("year")
            .size()
        )

        yearly.to_csv(
            RESULT_DIR / "yearly_funding_rounds.csv"
        )

        if not yearly.empty:

            plt.figure(figsize=(10, 5))

            yearly.plot(
                kind="line",
                marker="o"
            )

            plt.title("Startup Funding Rounds by Year")
            plt.xlabel("Year")
            plt.ylabel("Number of Funding Rounds")
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(
                RESULT_DIR / "yearly_funding_trend.png"
            )

            plt.show()

    # Compare how often each funding stage appears.

    if "stage" in df.columns:

        stage_counts = (
            df["stage"]
            .fillna("Unknown")
            .value_counts()
            .head(15)
        )

        print("\nFUNDING STAGE SUMMARY")
        print(stage_counts)

        stage_counts.to_csv(
            RESULT_DIR / "funding_stage_summary.csv"
        )

        plt.figure(figsize=(10, 6))

        stage_counts.sort_values().plot(
            kind="barh"
        )

        plt.title("Funding Rounds by Stage")
        plt.xlabel("Number of Funding Rounds")
        plt.ylabel("Funding Stage")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / "funding_stage.png"
        )

        plt.show()

    # Funding amounts can vary a lot, so use a log scale for the chart.

    if "amount_usd" in df.columns:

        amounts = df["amount_usd"].dropna()

        if not amounts.empty:

            print("\nFUNDING AMOUNT STATISTICS")
            print(amounts.describe())

            plt.figure(figsize=(10, 5))

            sns.histplot(
                np.log10(amounts),
                bins=30,
                kde=True
            )

            plt.title("Distribution of Log10 Funding Amount")
            plt.xlabel("Log10 Funding Amount in USD")
            plt.ylabel("Number of Funding Rounds")
            plt.tight_layout()

            plt.savefig(
                RESULT_DIR / "funding_distribution.png"
            )

            plt.show()

    # Check how numerical columns move in relation to one another.

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] >= 2:

        correlation = numeric_df.corr()

        correlation.to_csv(
            RESULT_DIR / "correlation_matrix.csv"
        )

        plt.figure(figsize=(10, 7))

        sns.heatmap(
            correlation,
            annot=True,
            cmap="coolwarm",
            fmt=".2f"
        )

        plt.title("Correlation Heatmap")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / "correlation_heatmap.png"
        )

        plt.show()

    # Measure the relationship between investor count and funding amount.

    if (
        "amount_usd" in df.columns
        and "investor_count" in df.columns
    ):

        pair = df[
            ["amount_usd", "investor_count"]
        ].dropna()

        if len(pair) >= 3:

            coefficient, p_value = pearsonr(
                pair["investor_count"],
                pair["amount_usd"]
            )

            print("\nPEARSON CORRELATION")
            print(
                "Investor count vs funding amount:",
                round(coefficient, 4)
            )
            print("P-value:", round(p_value, 6))


# =========================================================
# Step 5: Choose the information used by the models
# =========================================================

def prepare_features(df):

    if "amount_usd" not in df.columns:

        print("\nERROR: amount_usd column not found.")
        print("Available columns:", df.columns.tolist())

        return None, None

    model_df = df[
        df["amount_usd"].notna()
        & (df["amount_usd"] > 0)
    ].copy()

    if len(model_df) < 30:

        print(
            "\nNot enough valid records for machine learning."
        )

        return None, None

    preferred_features = [
        "sector",
        "stage",
        "city",
        "year",
        "founded_year",
        "startup_age",
        "investor_count",
        "news_mentions"
    ]

    features = [
        col for col in preferred_features
        if col in model_df.columns
    ]

    if not features:

        print("\nNo suitable predictor columns found.")
        print("Available columns:", model_df.columns.tolist())

        return None, None

    X = model_df[features].copy()
    y = model_df["amount_usd"].copy()

    X = X.dropna(
        axis=1,
        how="all"
    )

    if X.empty:

        print("\nAll selected features are missing.")

        return None, None

    print("\nFEATURES USED FOR PREDICTION")
    print(X.columns.tolist())
    print("Target: amount_usd")
    print("Records:", len(X))

    return X, y


# =========================================================
# Step 6: Prepare text and numeric columns for machine learning
# =========================================================

def build_preprocessor(X, scale=True):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        exclude=np.number
    ).columns.tolist()

    if scale:

        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]

    else:

        numeric_steps = [
            ("imputer", SimpleImputer(strategy="median"))
        ]

    categorical_steps = [
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                Pipeline(numeric_steps),
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                Pipeline(categorical_steps),
                categorical_features
            )
        )

    return ColumnTransformer(
        transformers=transformers
    )


# =========================================================
# Step 7: Estimate startup funding amounts
# =========================================================

def run_regression(df):

    print("\nTraining regression models")
    print("=" * 50)

    X, y = prepare_features(df)

    if X is None:
        return

    y_log = np.log1p(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_log,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    results = []

    # ---------------------------------------------
    # Simple Linear Regression
    # ---------------------------------------------

    if numeric_features:

        feature = numeric_features[0]

        simple_pipeline = Pipeline([
            (
                "preprocessing",
                build_preprocessor(
                    X[[feature]]
                )
            ),
            (
                "model",
                LinearRegression()
            )
        ])

        try:

            simple_pipeline.fit(
                X_train[[feature]],
                y_train
            )

            pred_log = simple_pipeline.predict(
                X_test[[feature]]
            )

            actual = np.expm1(
                y_test.to_numpy()
            )

            predicted = np.maximum(
                np.expm1(pred_log),
                0
            )

            results.append(
                evaluate_regression(
                    "Simple Linear Regression",
                    actual,
                    predicted
                )
            )

        except Exception as error:

            print(
                "Simple Linear Regression failed:",
                error
            )

    else:

        print(
            "Simple Linear Regression skipped: "
            "no numerical predictor exists."
        )

    # ---------------------------------------------
    # Multiple Linear Regression
    # ---------------------------------------------

    try:

        multiple_pipeline = Pipeline([
            (
                "preprocessing",
                build_preprocessor(X)
            ),
            (
                "model",
                LinearRegression()
            )
        ])

        multiple_pipeline.fit(
            X_train,
            y_train
        )

        pred_log = multiple_pipeline.predict(
            X_test
        )

        actual = np.expm1(
            y_test.to_numpy()
        )

        predicted = np.maximum(
            np.expm1(pred_log),
            0
        )

        results.append(
            evaluate_regression(
                "Multiple Linear Regression",
                actual,
                predicted
            )
        )

    except Exception as error:

        print(
            "Multiple Linear Regression failed:",
            error
        )

    # ---------------------------------------------
    # Polynomial Regression
    # Uses one numerical predictor to limit
    # computational cost.
    # ---------------------------------------------

    if numeric_features:

        feature = numeric_features[0]

        polynomial_pipeline = Pipeline([
            (
                "preprocessing",
                build_preprocessor(
                    X[[feature]]
                )
            ),
            (
                "polynomial",
                PolynomialFeatures(
                    degree=2,
                    include_bias=False
                )
            ),
            (
                "model",
                LinearRegression()
            )
        ])

        try:

            polynomial_pipeline.fit(
                X_train[[feature]],
                y_train
            )

            pred_log = polynomial_pipeline.predict(
                X_test[[feature]]
            )

            actual = np.expm1(
                y_test.to_numpy()
            )

            predicted = np.maximum(
                np.expm1(pred_log),
                0
            )

            results.append(
                evaluate_regression(
                    "Polynomial Regression",
                    actual,
                    predicted
                )
            )

        except Exception as error:

            print(
                "Polynomial Regression failed:",
                error
            )

    if results:

        results_df = pd.DataFrame(results)

        results_df.to_csv(
            RESULT_DIR / "regression_results.csv",
            index=False
        )

        print("\nREGRESSION COMPARISON")
        print(results_df.to_string(index=False))

        plt.figure(figsize=(10, 5))

        plt.bar(
            results_df["Model"],
            results_df["RMSE"]
        )

        plt.title("Regression Model Comparison")
        plt.xlabel("Model")
        plt.ylabel("RMSE")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / "regression_comparison.png"
        )

        plt.show()

    # ---------------------------------------------
    # OLS statistical regression
    # ---------------------------------------------

    if numeric_features:

        feature = numeric_features[0]

        try:

            stats_df = df[
                [feature, "amount_usd"]
            ].replace(
                [np.inf, -np.inf],
                np.nan
            ).dropna()

            stats_df = stats_df[
                stats_df["amount_usd"] > 0
            ]

            if len(stats_df) >= 10:

                stats_df["log_amount"] = np.log1p(
                    stats_df["amount_usd"]
                )

                X_ols = sm.add_constant(
                    stats_df[[feature]]
                )

                y_ols = stats_df["log_amount"]

                ols_model = sm.OLS(
                    y_ols,
                    X_ols
                ).fit()

                with open(
                    RESULT_DIR / "ols_summary.txt",
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write(
                        ols_model.summary().as_text()
                    )

                print("\nOLS REGRESSION SUMMARY")
                print(ols_model.summary())

        except Exception as error:

            print("OLS regression failed:", error)


def evaluate_regression(model_name, actual, predicted):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    mse = mean_squared_error(
        actual,
        predicted
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        actual,
        predicted
    )

    print("\nMODEL:", model_name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R-squared:", round(r2, 4))

    return {
        "Model": model_name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# =========================================================
# Step 8: Group funding rounds into low, medium, and high categories
# =========================================================

def run_classification(df):

    print("\nTraining classification models")
    print("=" * 50)

    X, y = prepare_features(df)

    if X is None:
        return

    # Split funding amounts into three groups using the one-third and two-third quantiles.

    q1 = y.quantile(1 / 3)
    q2 = y.quantile(2 / 3)

    if q1 >= q2:

        print(
            "Classification skipped because funding "
            "amount thresholds are not distinct."
        )

        return

    target = pd.Series(
        np.where(
            y <= q1,
            "Low Funding",
            np.where(
                y <= q2,
                "Medium Funding",
                "High Funding"
            )
        ),
        index=y.index
    )

    print("\nFUNDING CATEGORY THRESHOLDS")
    print("Low funding: up to", round(q1, 2))
    print("Medium funding: up to", round(q2, 2))
    print("High funding: above", round(q2, 2))

    print("\nCATEGORY COUNTS")
    print(target.value_counts())

    # Keep the class proportions similar in the training and testing sets.
    # Each category needs enough records for this split to work.

    counts = target.value_counts()

    if counts.min() < 2:

        print(
            "Classification skipped because at least "
            "one category has fewer than two records."
        )

        return

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=target
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5
        ),
        "Naive Bayes": GaussianNB(),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=RANDOM_STATE
        ),
        "Support Vector Machine": SVC(
            probability=True,
            random_state=RANDOM_STATE
        )
    }

    results = []

    best_score = -1
    best_name = None
    best_predictions = None
    best_actual = None
    best_labels = None

    for name, model in models.items():

        try:

            pipeline = Pipeline([
                (
                    "preprocessing",
                    build_preprocessor(X)
                ),
                (
                    "model",
                    model
                )
            ])

            pipeline.fit(
                X_train,
                y_train
            )

            predictions = pipeline.predict(
                X_test
            )

            accuracy = accuracy_score(
                y_test,
                predictions
            )

            precision = precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )

            probabilities = pipeline.predict_proba(
                X_test
            )

            try:

                loss = log_loss(
                    y_test,
                    probabilities,
                    labels=pipeline.classes_
                )

            except Exception:

                loss = np.nan

            try:

                auc = roc_auc_score(
                    y_test,
                    probabilities,
                    multi_class="ovr",
                    average="weighted",
                    labels=pipeline.classes_
                )

            except Exception:

                auc = np.nan

            print("\nMODEL:", name)
            print("Accuracy:", round(accuracy, 4))
            print("Precision:", round(precision, 4))
            print("Recall:", round(recall, 4))
            print("F1-score:", round(f1, 4))
            print(
                "Log Loss:",
                round(loss, 4)
                if np.isfinite(loss)
                else "Not available"
            )
            print(
                "ROC-AUC:",
                round(auc, 4)
                if np.isfinite(auc)
                else "Not available"
            )

            print("\nCLASSIFICATION REPORT")
            print(
                classification_report(
                    y_test,
                    predictions,
                    zero_division=0
                )
            )

            results.append({
                "Model": name,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1_Score": f1,
                "Log_Loss": loss,
                "ROC_AUC": auc
            })

            if accuracy > best_score:

                best_score = accuracy
                best_name = name
                best_predictions = predictions
                best_actual = y_test
                best_labels = list(pipeline.classes_)

        except Exception as error:

            print("\nModel failed:", name, error)

    if not results:
        print("No classification model completed successfully.")
        return

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        RESULT_DIR / "classification_results.csv",
        index=False
    )

    print("\nCLASSIFICATION COMPARISON")
    print(results_df.to_string(index=False))

    plt.figure(figsize=(10, 5))

    plt.bar(
        results_df["Model"],
        results_df["Accuracy"]
    )

    plt.title("Classification Model Comparison")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "classification_comparison.png"
    )

    plt.show()

    # Show where the best-performing model made correct and incorrect predictions.

    if best_predictions is not None:

        cm = confusion_matrix(
            best_actual,
            best_predictions,
            labels=best_labels
        )

        print("\nBEST MODEL:", best_name)
        print("\nCONFUSION MATRIX")
        print(cm)

        plt.figure(figsize=(7, 5))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=best_labels,
            yticklabels=best_labels
        )

        plt.title("Confusion Matrix - " + best_name)
        plt.xlabel("Predicted Category")
        plt.ylabel("Actual Category")
        plt.tight_layout()

        plt.savefig(
            RESULT_DIR / "confusion_matrix.png"
        )

        plt.show()


# =========================================================
# Step 9: Group similar funding records with K-Means
# =========================================================

def run_kmeans(df):

    print("\nFinding groups with K-Means")
    print("=" * 50)

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.dropna(
        axis=1,
        how="all"
    )

    numeric = numeric.loc[
        :,
        numeric.nunique(dropna=True) > 1
    ]

    if numeric.shape[1] < 2 or len(numeric) < 10:

        print("Not enough numerical data for K-Means.")
        return

    numeric = numeric.fillna(
        numeric.median()
    )

    numeric = numeric.sample(
        n=min(2000, len(numeric)),
        random_state=RANDOM_STATE
    )

    scaled = StandardScaler().fit_transform(
        numeric
    )

    k_values = range(
        2,
        min(8, len(scaled))
    )

    inertias = []

    for k in k_values:

        model = KMeans(
            n_clusters=k,
            random_state=RANDOM_STATE,
            n_init=10
        )

        model.fit(scaled)

        inertias.append(
            model.inertia_
        )

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(k_values),
        inertias,
        marker="o"
    )

    plt.title("K-Means Elbow Method")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "kmeans_elbow.png"
    )

    plt.show()

    # Use three groups for the final K-Means result so the groups are easy to compare.

    model = KMeans(
        n_clusters=3,
        random_state=RANDOM_STATE,
        n_init=10
    )

    labels = model.fit_predict(
        scaled
    )

    output = numeric.copy()
    output["cluster"] = labels

    output.to_csv(
        RESULT_DIR / "kmeans_clusters.csv",
        index=False
    )

    print("\nCLUSTER SIZES")
    print(pd.Series(labels).value_counts().sort_index())

    plt.figure(figsize=(8, 6))

    plt.scatter(
        scaled[:, 0],
        scaled[:, 1],
        c=labels
    )

    plt.title("K-Means Funding Clusters")
    plt.xlabel("First Numerical Feature (scaled)")
    plt.ylabel("Second Numerical Feature (scaled)")
    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / "kmeans_clusters.png"
    )

    plt.show()


# =========================================================
# Step 10: Compare hierarchical clustering methods
# =========================================================

def run_hierarchical(df):

    print("\nRunning hierarchical clustering")
    print("=" * 50)

    numeric = df.select_dtypes(
        include=np.number
    ).copy()

    numeric = numeric.replace(
        [np.inf, -np.inf],
        np.nan
    )

    numeric = numeric.dropna(
        axis=1,
        how="all"
    )

    numeric = numeric.loc[
        :,
        numeric.nunique(dropna=True) > 1
    ]

    if numeric.shape[1] < 2 or len(numeric) < 10:

        print("Not enough numerical data for clustering.")
        return

    numeric = numeric.fillna(
        numeric.median()
    )

    numeric = numeric.sample(
        n=min(300, len(numeric)),
        random_state=RANDOM_STATE
    )

    scaled = StandardScaler().fit_transform(
        numeric
    )

    methods = [
        "single",
        "complete",
        "average",
        "centroid"
    ]

    for method in methods:

        try:

            tree = linkage(
                scaled,
                method=method
            )

            labels = fcluster(
                tree,
                t=3,
                criterion="maxclust"
            )

            output = numeric.copy()
            output["cluster"] = labels

            output.to_csv(
                RESULT_DIR /
                ("hierarchical_" + method + ".csv"),
                index=False
            )

            print(
                method,
                "linkage cluster sizes:",
                pd.Series(labels).value_counts().to_dict()
            )

        except Exception as error:

            print(
                method,
                "linkage failed:",
                error
            )

    print("Hierarchical clustering completed.")


# =========================================================
# Step 11: Find common links between sectors and funding stages
# =========================================================

def run_association_rules(df):

    print("\nChecking links between sectors and funding stages")
    print("=" * 50)

    if "sector" not in df.columns or "stage" not in df.columns:

        print(
            "Association analysis requires both "
            "'sector' and 'stage' columns."
        )

        return

    data = df[
        ["sector", "stage"]
    ].dropna().copy()

    if len(data) < 10:

        print("Not enough records for association rules.")
        return

    data = data.sample(
        n=min(5000, len(data)),
        random_state=RANDOM_STATE
    )

    # Represent every record as two items:
    # Sector=... and Stage=...

    sector_items = (
        "Sector=" + data["sector"].astype(str)
    )

    stage_items = (
        "Stage=" + data["stage"].astype(str)
    )

    total = len(data)

    # Count individual items.

    sector_counts = sector_items.value_counts()
    stage_counts = stage_items.value_counts()

    # Count pairs of sector and stage.

    pair_counts = (
        pd.DataFrame({
            "sector_item": sector_items,
            "stage_item": stage_items
        })
        .value_counts()
        .reset_index(name="pair_count")
    )

    rules = []

    for _, row in pair_counts.iterrows():

        sector = row["sector_item"]
        stage = row["stage_item"]
        pair_count = row["pair_count"]

        # Skip sector-stage combinations that appear in less than 1% of the sampled records.

        support = pair_count / total

        if support < 0.01:
            continue

        sector_count = sector_counts.get(
            sector,
            0
        )

        stage_count = stage_counts.get(
            stage,
            0
        )

        if sector_count == 0 or stage_count == 0:
            continue

        # First rule: if a record belongs to this sector, how often does this stage appear?

        confidence_1 = pair_count / sector_count

        stage_support = stage_count / total

        lift_1 = (
            confidence_1 / stage_support
            if stage_support > 0
            else 0
        )

        rules.append({
            "Antecedent": sector,
            "Consequent": stage,
            "Support": support,
            "Confidence": confidence_1,
            "Lift": lift_1
        })

        # Reverse the rule: if a record has this stage, how often does this sector appear?

        confidence_2 = pair_count / stage_count

        sector_support = sector_count / total

        lift_2 = (
            confidence_2 / sector_support
            if sector_support > 0
            else 0
        )

        rules.append({
            "Antecedent": stage,
            "Consequent": sector,
            "Support": support,
            "Confidence": confidence_2,
            "Lift": lift_2
        })

    if not rules:

        print(
            "No sector-stage pairs met the 1% support threshold."
        )

        return

    rules_df = pd.DataFrame(rules)

    rules_df = rules_df.drop_duplicates(
        subset=["Antecedent", "Consequent"]
    )

    rules_df = rules_df.sort_values(
        by="Lift",
        ascending=False
    )

    rules_df.to_csv(
        RESULT_DIR / "association_rules.csv",
        index=False
    )

    print("\nTOP 10 ASSOCIATION RULES")

    print(
        rules_df.head(10).to_string(index=False)
    )

    print(
        "\nAssociation rules saved to association_rules.csv"
    )


# =========================================================
# Step 12: Run the complete project in order
# =========================================================

def main():

    df = load_data()

    if df is None:
        return

    if df.empty:

        print("The dataset contains no records.")
        return

    inspect_data(df)

    df = clean_data(df)

    if df.empty:

        print("No usable records remain after cleaning.")
        return

    perform_eda(df)

    run_regression(df)

    run_classification(df)

    run_kmeans(df)

    run_hierarchical(df)

    run_association_rules(df)

    print("\n" + "=" * 65)
    print(" Project run completed.")
    print("=" * 65)

    print("\nResults folder:")
    print(RESULT_DIR)

    print("\nImportant output files:")
    print("1. cleaned_dataset.csv")
    print("2. regression_results.csv")
    print("3. classification_results.csv")
    print("4. correlation_matrix.csv")
    print("5. ols_summary.txt")
    print("6. kmeans_clusters.csv")
    print("7. association_rules.csv")

    print("\nOpen the IDLE Shell to review the results and any messages.")


if __name__ == "__main__":
    main()
