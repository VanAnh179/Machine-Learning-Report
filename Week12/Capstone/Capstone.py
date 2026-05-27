import pandas as pd
import os
from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

# 1. Tải dữ liệu (giả sử file có định dạng user, item, rating)
# Cần điều chỉnh rating_scale phù hợp với dữ liệu thực tế
data_dir = os.path.dirname(os.path.abspath(__file__))
reader = Reader(rating_scale=(1, 5)) 
data = Dataset.load_from_df(pd.read_csv(os.path.join(data_dir, 'course_ratings.csv'))[['user', 'item', 'rating']], reader)

# 2. Chia dữ liệu Train/Test
trainset, testset = train_test_split(data, test_size=0.3)

# 3. Cấu hình và huấn luyện mô hình KNN
# Sử dụng Cosine similarity và User-based approach
sim_options = {
    'name': 'cosine',
    'user_based': False  # Item-based: tìm item tương tự (126 items < 33901 users)
}

model = KNNBasic(k=40, sim_options=sim_options) # k là số hàng xóm
model.fit(trainset)

# 4. Dự đoán và đánh giá
predictions = model.test(testset)
rmse_score = accuracy.rmse(predictions) # Tính RMSE để đánh giá độ chính xác

print(f"Độ lỗi RMSE của hệ thống gợi ý: {rmse_score}")

# Ví dụ dự đoán cho một User cụ thể
uid = str(1) # ID người dùng
iid = str(101) # ID khóa học
pred = model.predict(uid, iid)
print(f"Dự đoán đánh giá của User {uid} cho Course {iid}: {pred.est}")