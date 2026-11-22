"""
ITA202 - Titanic Survival Prediction
Chạy: python run_project.py
Chương trình tự tải dataset CSV, khám phá dữ liệu, huấn luyện và đánh giá mô hình.
"""

from pathlib import Path
import urllib.request
import warnings
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")

# 1. Khai báo đường dẫn dự án để chương trình chạy được ở nhiều máy.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
DATA_PATH = DATA_DIR / "titanic.csv"
OUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# 2. Tải dữ liệu từ kho dữ liệu công khai nếu chưa có file CSV.
DATA_URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"

def download_dataset():
    """Tải dữ liệu Titanic và lưu vào thư mục data."""
    if not DATA_PATH.exists():
        print("Đang tải Titanic dataset...")
        try:
            urllib.request.urlretrieve(DATA_URL, DATA_PATH)
            print(f"Đã lưu dữ liệu tại: {DATA_PATH}")
        except Exception as exc:
            raise RuntimeError(
                "Không tải được dữ liệu. Hãy kiểm tra Internet rồi chạy lại. "
                f"Nguồn: {DATA_URL}"
            ) from exc

# 3. Transformer tự xây dựng để giới hạn ngoại lệ theo IQR.
class IQRCapper(BaseEstimator, TransformerMixin):
    """Giới hạn giá trị số nằm ngoài [Q1 - 1.5*IQR, Q3 + 1.5*IQR]."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.q1_ = np.nanpercentile(X, 25, axis=0)
        self.q3_ = np.nanpercentile(X, 75, axis=0)
        iqr = self.q3_ - self.q1_
        self.lower_ = self.q1_ - 1.5 * iqr
        self.upper_ = self.q3_ + 1.5 * iqr
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

def main():
    download_dataset()

    # 4. Đọc dữ liệu bằng Pandas.
    df = pd.read_csv(DATA_PATH)
    print("\n========== KHÁM PHÁ DỮ LIỆU ==========")
    print("Kích thước dữ liệu (rows, columns):", df.shape)
    print("\nThông tin dữ liệu:")
    df.info()
    print("\nThống kê mô tả:")
    print(df.describe(include="all").T)
    print("\nSố lượng giá trị thiếu:")
    print(df.isna().sum())

    # 5. Chọn các thuộc tính có ý nghĩa, loại bỏ cột định danh/không cần thiết.
    # Không dùng alive vì đây là thông tin trùng trực tiếp với nhãn survived.
    feature_cols = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target_col = "survived"
    data = df[feature_cols + [target_col]].copy()
    data = data.dropna(subset=[target_col])

    X = data[feature_cols]
    y = data[target_col].astype(int)

    numeric_features = ["pclass", "age", "sibsp", "parch", "fare"]
    categorical_features = ["sex", "embarked"]

    # 6. Chia train/test trước khi học tham số tiền xử lý để tránh rò rỉ dữ liệu.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print("\nKích thước X_train:", X_train.shape)
    print("Kích thước X_test:", X_test.shape)

    # 7. Tạo quy trình tiền xử lý cho cột số và cột phân loại.
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  # Điền thiếu bằng trung vị.
        ("outlier", IQRCapper()),                       # Giới hạn ngoại lệ theo IQR.
        ("scaler", StandardScaler())                    # Đưa số liệu về cùng thang đo.
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền bằng giá trị phổ biến.
        ("onehot", OneHotEncoder(handle_unknown="ignore"))     # Mã hóa biến phân loại.
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # 8. Giai đoạn 1: mô hình cơ sở Logistic Regression.
    baseline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ])
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    def calculate_metrics(name, y_true, y_pred):
        return {
            "model": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0)
        }

    results = [calculate_metrics("Logistic Regression", y_test, baseline_pred)]

    # 9. Giai đoạn 2: GridSearchCV để tìm tham số tốt hơn.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "KNN": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [3, 5, 7, 9],
             "model__weights": ["uniform", "distance"]}
        ),
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {"model__max_depth": [3, 5, 8, None],
             "model__min_samples_split": [2, 5, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {"model__n_estimators": [100, 200],
             "model__max_depth": [None, 5, 10],
             "model__min_samples_split": [2, 5]}
        )
    }

    best_estimators = {"Logistic Regression": baseline}
    search_details = {}
    for name, (estimator, params) in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", estimator)
        ])
        search = GridSearchCV(
            estimator=pipe,
            param_grid=params,
            scoring="accuracy",
            cv=cv,
            n_jobs=-1,
            refit=True
        )
        print(f"\nĐang tìm tham số tốt nhất cho {name}...")
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        pred = best_model.predict(X_test)
        results.append(calculate_metrics(name, y_test, pred))
        best_estimators[name] = best_model
        search_details[name] = {
            "best_params": search.best_params_,
            "best_cv_accuracy": float(search.best_score_)
        }

    # 10. So sánh kết quả trên cùng tập test và chọn mô hình theo Accuracy.
    result_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    result_df.to_csv(OUT_DIR / "model_comparison.csv", index=False)
    print("\n========== SO SÁNH MÔ HÌNH ==========")
    print(result_df.to_string(index=False))

    best_name = result_df.iloc[0]["model"]
    final_model = best_estimators[best_name]
    final_pred = final_model.predict(X_test)
    final_cm = confusion_matrix(y_test, final_pred)

    # 11. Lưu mô hình cuối cùng bằng Joblib.
    joblib.dump(final_model, OUT_DIR / "titanic_best_model.pkl")

    # 12. Xuất báo cáo văn bản với số liệu được tính từ lần chạy này.
    summary = {
        "dataset_rows": int(df.shape[0]),
        "dataset_columns": int(df.shape[1]),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "missing_values": {k: int(v) for k, v in df.isna().sum().items()},
        "model_comparison": result_df.to_dict(orient="records"),
        "best_model": best_name,
        "best_model_confusion_matrix": final_cm.tolist(),
        "best_model_classification_report": classification_report(
            y_test, final_pred, zero_division=0, output_dict=True
        ),
        "grid_search": search_details
    }
    with open(OUT_DIR / "experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with open(OUT_DIR / "experiment_report.txt", "w", encoding="utf-8") as f:
        f.write("KET QUA THUC NGHIEM - ITA202 TITANIC\n")
        f.write("=" * 55 + "\n")
        f.write(f"So dong du lieu: {df.shape[0]}\n")
        f.write(f"So cot du lieu: {df.shape[1]}\n")
        f.write(f"So mau train: {len(X_train)}; so mau test: {len(X_test)}\n\n")
        f.write("SO SANH MO HINH:\n")
        f.write(result_df.to_string(index=False))
        f.write(f"\n\nMo hinh tot nhat theo Accuracy: {best_name}\n")
        f.write(f"Confusion Matrix:\n{final_cm}\n")
        f.write("\nClassification report:\n")
        f.write(classification_report(y_test, final_pred, zero_division=0))
        f.write("\nTham so GridSearchCV:\n")
        f.write(json.dumps(search_details, ensure_ascii=False, indent=2))

    # 13. Trực quan hóa ma trận nhầm lẫn và so sánh Accuracy.
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(final_cm)
    ax.set_title(f"Confusion Matrix - {best_name}")
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_ylabel("Nhãn thực tế")
    ax.set_xticks([0, 1], ["Không sống sót", "Sống sót"])
    ax.set_yticks([0, 1], ["Không sống sót", "Sống sót"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(final_cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(result_df["model"], result_df["accuracy"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("So sánh Accuracy trên tập Test")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "model_accuracy.png", dpi=160)
    plt.close(fig)

    print("\nĐã hoàn thành. Các file kết quả nằm trong thư mục outputs/.")
    print("Mô hình tốt nhất:", best_name)
    print("Accuracy:", round(float(result_df.iloc[0]["accuracy"]), 4))

if __name__ == "__main__":
    main()
