from abc import ABC, abstractmethod
import pandas as pd
import zipfile
import os


"""Product - Sản phẩm trừu tượng -- Chứa rất nhiều nguồn dữ liệu"""
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Một phương pháp trừu tượng để nhập dữ liệu từ 1 file"""
        pass

"""Concrete - Sản phẩm cụ thể - File dữ liệu cụ thể"""
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: str) -> pd.DataFrame:
        """Giải nén file zip và trả về dữ liệu về dạng DataFrame"""
        if not file_path.endswith(".zip"):
            raise ValueError("File được cung cấp không phải là file zip.")

        """Giải nén file zip"""
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall("unzip_dataset")

        """Tìm các file csv sau khi được unzip"""
        unzip_file = os.listdir("unzip_dataset")
        csv_files = [csv_file for csv_file in unzip_file if csv_file.endswith(".csv")]

        if (len(csv_files) == 0):
            raise FileNotFoundError("Không tồn tại bất kì file CVS nào.")
        if (len(csv_files) > 1):
            raise ValueError("Có nhiều file CVS. Vui lòng chỉ định 1 file.")
        
        csv_file_path = os.path.join("unzip_dataset", csv_files[0])
        df = pd.read_csv(csv_file_path)

        return df

class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """Nếu là file zip thì trả về đối tượng Data Ingestor còn không thì báo lỗi -> đảm bảo tính đa hình
           File_extension được định nghĩa là chuỗi kí tự đứng sau dấu chấm cuối cùng trong file, dùng để chỉ cái định dạng của file là gì."""
        if file_extension == ".zip":
            return ZipDataIngestor()
        else:
            raise ValueError(f"Tệp {file_extension} không được sử dụng cho dự án này.")

#Test
if __name__ == "__main__":

    # Đã test
    file_path = "D:\\Project_Portfolio\\HOUSE-PRICE-MLOPS\\data\\storage.zip"
    file_extension = os.path.splitext(file_path)[1]
    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)
    df = data_ingestor.ingest(file_path)
    # print(df.head())

    # pass





        






