"""
ITA202 - Assignment 1: Titanic data preprocessing
Run: python preprocess_data.py

The script downloads Titanic data if needed, splits train/test first,
fits preprocessing steps on the training set only, and exports processed CSVs.
"""

from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "titanic.csv"
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


class IQRCapper(BaseEstimator, TransformerMixin):
    """Giới hạn ngoại lệ theo ngưỡng Q1 - 1.5*IQR và Q3 + 1.5*IQR."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.n_features_in_ = X.shape[1]
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * iqr
        self.upper_ = self.q3_ + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Bao toan ten cot de sklearn co the lay ten dac trung sau Pipeline.
        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]
        return np.asarray(input_features, dtype=object)


def main():
    if not DATA_PATH.exists():
        print("Dang tai du lieu Titanic...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    df = pd.read_csv(DATA_PATH)
    print("===== KHAM PHA DU LIEU =====")
    print("Kich thuoc du lieu goc:", df.shape)
    print("\nSo gia tri thieu theo cot:")
    print(df.isna().sum())

    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target_col = "survived"
    data = df[feature_cols + [target_col]].dropna(subset=[target_col]).copy()

    X = data[feature_cols]
    y = data[target_col].astype(int)

    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    # Chia train/test truoc khi fit cac buoc tien xu ly de tranh ro ri du lieu.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier", IQRCapper()),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])

    # Chi hoc tham so tien xu ly tren tap train.
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()
    train_df = pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index)
    test_df = pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index)
    train_df[target_col] = y_train
    test_df[target_col] = y_test

    train_path = OUTPUT_DIR / "titanic_processed_train.csv"
    test_path = OUTPUT_DIR / "titanic_processed_test.csv"
    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    summary_path = OUTPUT_DIR / "preprocessing_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("ITA202 - KET QUA TIEN XU LY DU LIEU TITANIC\n")
        f.write("=" * 55 + "\n")
        f.write(f"So dong du lieu goc: {len(df)}\n")
        f.write(f"So cot du lieu goc: {df.shape[1]}\n")
        f.write(f"So mau train: {len(train_df)}\n")
        f.write(f"So mau test: {len(test_df)}\n")
        f.write(f"So cot sau One-Hot Encoding (khong tinh nhan): {len(feature_names)}\n")
        f.write("Xu ly thieu: median cho cot so, most_frequent cho cot phan loai.\n")
        f.write("Ngoai le: gioi han theo IQR; cot so: StandardScaler.\n")
        f.write("Cot phan loai: One-Hot Encoding.\n")
        f.write("Da chia train/test truoc khi fit tien xu ly de tranh data leakage.\n")

    print("\n===== KET QUA TIEN XU LY =====")
    print("Train:", train_df.shape)
    print("Test:", test_df.shape)
    print("Da luu:", train_path)
    print("Da luu:", test_path)
    print("Da luu:", summary_path)
    print("\nHOAN TAT TIEN XU LY DU LIEU!")


if __name__ == "__main__":
    main()
