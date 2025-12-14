<h1 align="center">HousePrice-MLOps</h1>

<h3 align="center">
Dự án dự đoán giá nhà được xây dựng và triển khai theo quy trình MLOps hiện đại, 
kết hợp tự động hóa pipeline, container hóa và triển khai trên nền tảng đám mây.
</h3>

---
#### **Thiết lập môi trường ảo cho Python**
**Bước 1**: Tạo môi trường ảo (venv)
```
python -m venv venv
```
**Bước 2**: Vào venv 
```
venv/Scripts/activate
```
**Bước 3**: Tạo file requirement.txt.

**Bước 4**: Tải các thư viện của file requirement.txt cấu hình thư viện cho dự án
```
pip install -r requirements.txt
```
***Lưu ý***: Nếu bạn muốn mở rộng dự án thì sau khi pip install 1 thư viện bất kì ngoài các thư viện đã có sẵn trong file `requirements.txt` thì thêm các thư viện được cài thêm vào file `requirements.txt` như sau:
```
pip freeze > requirements.txt
```

#### **THÔNG TIN DỰ ÁN**

Dự án sử dụng tập dữ liệu chuẩn về dự đoán giá nhà là : `AmesHousing`

**1. Tổng quan Dataset**
Dataset AmesHousing là bộ dữu liệu về nhà ở tại thành phố Ames, Lowa(Mỹ) là một dataset thay thế chất lượng cao cho Boston Housing trong các bài toán dự đoán giá nhà.

***Mục tiêu***: Dự đoán giá nhà dựa trên các đặc trưng về cấu trúc nhà, tiện ích, chất lượng, vị trí,...

**2. Thông tin dữ liệu của `AmesHousing.csv`**

***Số lượng:***
- Số dòng(bảng ghi): 2931 dòng
- Số cột(biến): 82 cột (bao gồm cả biến target: salePrice)

**3. Mô tả biến và nhóm cột**

***Nhóm Indentification***
| Order                 | PID                     |
|-----------------------|-------------------------|
| Số thứ tự của bảng ghi| Mã thửa đất/bất động sản|





