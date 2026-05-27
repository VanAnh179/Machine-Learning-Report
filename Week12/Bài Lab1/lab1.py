import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
from pathlib import Path

matplotlib.use('Agg')


try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

warnings.filterwarnings('ignore', category=DeprecationWarning)

# ==========================================
# 1. TẢI DỮ LIỆU
# ==========================================
# Đảm bảo file CSV nằm cùng thư mục với file code
INPUT_FILE_PATH = Path(__file__).resolve().parent / 'water_potability.csv'
df = pd.read_csv(INPUT_FILE_PATH)

print("--- 5 DÒNG ĐẦU TIÊN CỦA DỮ LIỆU ---")
display(df.head())

# ==========================================
# 2. MÔ TẢ VÀ KHÁM PHÁ DỮ LIỆU (EDA)
# ==========================================
print("\n--- THÔNG TIN TỔNG QUAN ---")
df.info()

print("\n--- THỐNG KÊ MÔ TẢ ---")
display(df.describe())

# ==========================================
# 3. XỬ LÝ DỮ LIỆU THIẾU (MISSING VALUES)
# ==========================================
print("\n--- SỐ LƯỢNG DỮ LIỆU THIẾU TRƯỚC KHI XỬ LÝ ---")
print(df.isnull().sum())

# Điền các giá trị thiếu bằng giá trị trung vị (median) để tránh bị ảnh hưởng bởi ngoại lệ
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())

print("\n--- SỐ LƯỢNG DỮ LIỆU THIẾU SAU KHI XỬ LÝ ---")
print(df.isnull().sum())

# ==========================================
# 4. XỬ LÝ NGOẠI LỆ (OUTLIERS)
# ==========================================
# Sử dụng phương pháp Z-score (loại bỏ các dòng có Z-score > 3 hoặc < -3)
z_scores = np.abs(stats.zscore(df.to_numpy(dtype=float)))
df_clean = df[(z_scores < 3).all(axis=1)].copy()

print(f"\nKích thước dữ liệu ban đầu: {df.shape}")
print(f"Kích thước dữ liệu sau khi xóa ngoại lệ: {df_clean.shape}")

# ==========================================
# 5. XÁC ĐỊNH BIẾN CATEGORICAL & NUMERICAL
# ==========================================
# Dataset này toàn bộ là biến số (Numerical), cột Potability là target (0 hoặc 1)
numerical_cols = [col for col in df_clean.columns if col != 'Potability']
categorical_cols = [] # Không có biến phân loại cần encode

print(f"\nCác biến dạng số cần chuẩn hóa: {numerical_cols}")

# ==========================================
# 6. TRỰC QUAN HÓA DỮ LIỆU (VISUALIZATION)
# ==========================================
# Vẽ biểu đồ phân phối (Histogram) cho các biến số
plt.figure(figsize=(15, 12))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_clean[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Phân phối của {col}')
    plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'histograms.png', dpi=150, bbox_inches='tight')
plt.close()

# Vẽ ma trận tương quan (Correlation Matrix)
plt.figure(figsize=(10, 8))
sns.heatmap(df_clean.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Ma trận tương quan giữa các đặc trưng')
plt.savefig(Path(__file__).resolve().parent / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 7. CHUẨN HÓA DỮ LIỆU (FEATURE TRANSFORMATION)
# ==========================================
# Scale các cột dạng số (trừ cột target Potability) bằng StandardScaler
scaler = StandardScaler()
df_clean[numerical_cols] = scaler.fit_transform(df_clean[numerical_cols])

print("\n--- 5 DÒNG ĐẦU SAU KHI CHUẨN HÓA (SCALE) ---")
display(df_clean.head())