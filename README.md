# ITA202 - Titanic Survival Prediction

## 1. Mục tiêu
Xây dựng quy trình Machine Learning phân loại khả năng sống sót của hành khách Titanic, bám sát 2 giai đoạn trong đề ITA202.

## 2. Cấu trúc
- `run_project.py`: toàn bộ code, có chú thích tiếng Việt.
- `data/titanic.csv`: được tự tải ở lần chạy đầu tiên.
- `outputs/model_comparison.csv`: bảng so sánh mô hình.
- `outputs/experiment_results.json`: số liệu chi tiết.
- `outputs/experiment_report.txt`: báo cáo kết quả thực nghiệm.
- `outputs/titanic_best_model.pkl`: mô hình tốt nhất.
- `outputs/confusion_matrix.png`: ma trận nhầm lẫn.
- `outputs/model_accuracy.png`: biểu đồ Accuracy.

## 3. Cách chạy trên Windows
1. Cài Python 3.10 trở lên.
2. Giải nén thư mục dự án.
3. Mở CMD/Terminal tại thư mục `ITA202_Titanic_Project`.
4. Tạo môi trường ảo (khuyến nghị): `python -m venv .venv`
5. Kích hoạt: `.venv\Scripts\activate`
6. Cài thư viện: `pip install -r requirements.txt`
7. Chạy: `python run_project.py`

Lần chạy đầu cần Internet để tải CSV từ:
https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv

## 4. Thiết kế thí nghiệm
- Nhãn: `survived` (0 = không sống sót, 1 = sống sót).
- Thuộc tính: pclass, sex, age, sibsp, parch, fare, embarked.
- Chia train/test: 80%/20%, stratify theo nhãn, random_state=42.
- Logistic Regression là mô hình cơ sở.
- KNN, Decision Tree, Random Forest được tối ưu bằng GridSearchCV với Stratified 5-Fold CV.
- Imputation, IQR capping, scaling và One-Hot Encoding nằm trong Pipeline để hạn chế rò rỉ dữ liệu.
- Chọn mô hình cuối theo Accuracy trên tập test; vẫn báo cáo Precision, Recall, F1 và Confusion Matrix.

## 5. Lưu ý học thuật
Đây là bộ khung dự án có thể chạy lại. Hãy tự chạy trên máy, đọc kết quả sinh ra, chỉnh sửa nhận xét theo số liệu thực tế và ghi rõ nguồn dữ liệu trong báo cáo. Không điền số liệu dự đoán trước khi chương trình chạy.
