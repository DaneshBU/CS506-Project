import pandas as pd
from sklearn.model_selection import train_test_split

DATA_URL = "https://raw.githubusercontent.com/DaneshBU/CS506-Project/main/data/credit_risk_dataset.csv"


def test_dataset_loads():
    df = pd.read_csv(DATA_URL)

    assert not df.empty
    assert "loan_status" in df.columns


def test_target_is_binary():
    df = pd.read_csv(DATA_URL)

    values = set(df["loan_status"].dropna().unique())

    assert values.issubset({0, 1})


def test_no_duplicate_rows_after_cleaning():
    df = pd.read_csv(DATA_URL)

    cleaned_df = df.drop_duplicates()

    assert cleaned_df.shape[0] <= df.shape[0]


def test_train_validation_test_split_sizes():
    df = pd.read_csv(DATA_URL).drop_duplicates()

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=42,
        stratify=y_train_val,
    )

    assert len(X_train) > 0
    assert len(X_val) > 0
    assert len(X_test) > 0

    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_required_feature_types_exist():
    df = pd.read_csv(DATA_URL)

    X = df.drop(columns=["loan_status"])

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    assert len(numeric_features) > 0
    assert len(categorical_features) > 0