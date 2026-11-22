# Hướng dẫn hiểu code và chuẩn bị bảo vệ ITA202

## 1. Bài toán
Đây là bài toán phân loại có giám sát (Supervised Learning). Mỗi dòng là một hành khách; `survived` là nhãn cần dự đoán: 0 hoặc 1.

## 2. Vì sao chọn các cột này?
- `pclass`: hạng vé.
- `sex`: giới tính.
- `age`: tuổi.
- `sibsp`, `parch`: số người thân đi cùng.
- `fare`: giá vé.
- `embarked`: cảng lên tàu.
Các cột định danh như tên, mã vé không dùng trong mô hình cơ bản. `alive` không dùng vì trùng thông tin với `survived`, gây rò rỉ nhãn.

## 3. Các bước xử lý
1. `read_csv`: đọc CSV thành DataFrame.
2. `shape`, `info`, `describe`: xem kích thước, kiểu dữ liệu, thống kê.
3. `isna().sum()`: đếm giá trị thiếu.
4. `train_test_split`: chia 80% train và 20% test; `stratify=y` giữ tỷ lệ hai lớp tương tự nhau.
5. `SimpleImputer`: điền tuổi/giá vé/số lượng bị thiếu bằng trung vị; cột chữ điền bằng giá trị phổ biến nhất.
6. `IQRCapper`: giới hạn giá trị ngoại lệ dựa trên Q1 - 1.5*IQR và Q3 + 1.5*IQR. Đây là xử lý giảm ảnh hưởng ngoại lệ, không xóa hàng.
7. `StandardScaler`: chuẩn hóa biến số để KNN và Logistic Regression không bị biến có thang đo lớn chi phối.
8. `OneHotEncoder`: chuyển biến chữ thành cột số.
9. `Pipeline` và `ColumnTransformer`: gói các bước xử lý cùng mô hình; fit các tham số tiền xử lý trên train, rồi áp dụng cho test.
10. `GridSearchCV`: thử các tổ hợp tham số, đánh giá bằng 5-fold cross-validation trên tập train.
11. Đánh giá cuối cùng trên tập test chưa dùng trong quá trình chọn tham số.

## 4. Ý nghĩa chỉ số
- Accuracy = số dự đoán đúng / tổng số mẫu.
- Precision: trong các trường hợp mô hình dự đoán sống sót, tỷ lệ dự đoán đúng.
- Recall: trong các trường hợp thực tế sống sót, mô hình tìm ra được bao nhiêu.
- F1: trung bình điều hòa của Precision và Recall.
- Confusion Matrix: bảng đối chiếu nhãn thực tế và dự đoán. Hàng là nhãn thật, cột là nhãn dự đoán.

## 5. Câu hỏi giảng viên có thể hỏi
**Q: Vì sao đây là supervised learning?**  
A: Vì dữ liệu huấn luyện đã có nhãn `survived`, mô hình học mối quan hệ giữa thuộc tính và nhãn.

**Q: Vì sao chia train/test?**  
A: Train dùng để học; test dùng để kiểm tra khả năng dự đoán trên dữ liệu chưa dùng để huấn luyện.

**Q: Vì sao không điền thiếu trước khi chia dữ liệu?**  
A: Nếu tính trung vị trên toàn bộ dữ liệu sẽ làm thông tin test lọt vào quá trình học. Đặt imputer trong Pipeline giúp chỉ fit trên train.

**Q: Vì sao cần Scaling?**  
A: Các thuộc tính có đơn vị và khoảng giá trị khác nhau. Scaling giúp thuật toán dựa trên khoảng cách/ hệ số hoạt động ổn định hơn.

**Q: GridSearchCV làm gì?**  
A: Thử các bộ tham số được khai báo và chọn bộ có điểm cross-validation tốt nhất.

**Q: Vì sao không dùng cột `alive`?**  
A: `alive` biểu diễn trực tiếp kết quả sống/chết, tương đương nhãn `survived`; dùng nó sẽ làm mô hình biết trước đáp án (data leakage).

**Q: Accuracy cao có nghĩa mô hình hoàn hảo không?**  
A: Không. Cần xem thêm Precision, Recall, F1 và Confusion Matrix, đồng thời lưu ý tập Titanic có thể không cân bằng lớp.

## 6. Việc cần làm sau khi chạy
- Mở `outputs/experiment_report.txt`.
- Chép số liệu thực tế vào báo cáo Word và slide.
- Giải thích mô hình có Accuracy cao nhất, nhưng không kết luận chỉ dựa vào Accuracy.
- Thử thay `random_state` hoặc tham số để hiểu tính ổn định; ghi rõ mọi thay đổi.
