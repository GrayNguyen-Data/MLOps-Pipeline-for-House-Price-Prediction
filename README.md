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

***Nhóm 1: Tiện nghi/hệ thống kỹ thuật trong nhà***

Các cột nói về tiện nghi, điều hòa
|Name|Describle|
|-|-|
| **Central Air**| Có điều hòa giữa nhà hay không (Y/N)|
| **Electrical** | Hệ thống điện chính(Ví du: SBrkr, FuseA)|
| **Functional**|Tình trạng chức năng tổng thể của nhà|
| **Paved Drive**| Lối xe vào (driveway) có được lát nhựa/bê tổng không|

---
***Nhóm 2: Diện tích và không gian sử dụng***

Các cột diện tích mặt sàn và không gian sống

|Name|Describle|
|-|-|
| **1st Flr SF**|Diện tích sàn tầng 1(square feet)|
| **2nd Flr SF**|Diện tích sàn tầng 2|
| **Low Qual Fin SF**|Diện tích sàn hoàn thiện chất lượng thấp|
| **Gr Liv Area**|Diện tích sử dụng trên mặt đất không tính tầng hầm|
| **Wood Deck SF**|Diện tích sàn gỗ(desk)|
| **Open Porch SF** |Diện tích hiên mở|
| **Enclosed Porch**|Diện tích hiên kín|
| **3Ssn Porch**|Diện tích hiên 3 mùa|
| **Screen Porch**|Diện tích hiên có lưới chắn|
| **Pool Area**|Diện tích hồ bơi|
| **Garage Area**|Diện tích gara|

---

***Nhóm 3: Phòng/Bố cục bên trong***

Các cột về số lượng phòng

|Name|Describle|
|-|-|
| **Bsmt Full Bath**|Số phòng tắm đầy đủ ở tầng hầm|
| **Bsmt Half Bath**|Số phòng tắm nửa ở tầng hầm|
| **Full Bath**|Số phòng tắm đầy đủ trên mặt đất|
| **Half Bath**|Số phòng tắm nửa (toilet, không đủ tiện nghi tắm)|
| **Bedroom AbvGr**|Số phòng ngủ trên mặt đất|
| **Kitchen AbvGr** |Số bếp trên mặt đất|
| **TotRms AbvGrd**|Tổng số phòng trên mặt đất (không tính phòng tắm)|
| **Fireplaces**|Số lò sưởi|
| **Garage Cars**|Sức chứa gara tính theo số xe|

---

***Nhóm 4: Chất lượng và tình trạng hoàn thiện***

Các cột mang tính đánh giá chất lượng

|Name|Describle|
|-|-|
| **Kitchen Qual**|Chất lượng bếp|
| **Fireplace Qu**|Chất lượng lò sưởi|
| **Garage Qual**| Chất lượng gara|
| **Garage Cond**|Tình trạng gara (condition)|
| **Pool QC**|Chất lượng hồ bơi|
| **Fence** |Loại hàng rào|
| **Misc Feature**|Đặc điểm phụ thêm (shed, elevator, …)|

---

***Nhóm 5: Thông tin gara***

|Name|Describle|
|-|-|
| **Garage Type**|Loại gara|
| **Garage Yr Blt**| Năm xây gara|
| **Garage Finish**| Mức độ hoàn thiện nội thất gara|
| **Garage Cars**|Sức chứa |
| **Garage Area**|Diện tích|
| **Garage Qual** |Chất lượng|
| **Garage Cond**|Tình trạng|

---

***Nhóm 6: Tiện nghi bên ngoài/Ngoại thất***

|Name|Describle|
|-|-|
| **Misc Feature**|Các tiện nghi khác |
| **Misc Val**|Giá trị ước tính của tiện ích phụ|
| **Các thuộc tính đã có**| Oử trên|

---

***Nhóm 7: Thông tin gara***

Các cột về thời điểm bán và loại giao dịch:
|Name|Describle|
|-|-|
| **Mo Sold**|Tháng bán|
| **Yr Sold**|Năm bán|
| **Sale Type**|Loại giao dịch|
| **Sale Condition**|Tình trạng giao dịch|

---

***Nhóm 8: Target***

`SalePrice:` Giá bán ngôi nhà (biến mục tiêu khi làm mô hình dự đoán).

Download file.csv: [AmesHousing.csv](D:\Project_Portfolio\HOUSE-PRICE-MLOPS\unzip_dataset\AmesHousing.csv) 