import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.special import boxcox1p
from pathlib import Path

# Scikit-Learn modules
from sklearn.model_selection import KFold, cross_val_predict, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNetCV
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings('ignore')

matplotlib.use('Agg')

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)

# ==========================================
# 1. TẢI VÀ MÔ TẢ DỮ LIỆU
# ==========================================
df = pd.read_csv(Path(__file__).resolve().parent / 'Data' / 'insurance.csv')

print("--- 5 DÒNG ĐẦU TIÊN CỦA DỮ LIỆU ---")
display(df.head())

print("\n--- THỐNG KÊ MÔ TẢ ---")
display(df.describe(include='all'))

# ==========================================
# 2. CHUYỂN ĐỔI BIẾN CATEGORICAL -> NUMERICAL
# ==========================================
# Sử dụng pd.get_dummies để One-Hot Encode các cột phân loại (sex, smoker, region)
# drop_first=True để tránh bẫy đa cộng tuyến (dummy variable trap) trong hồi quy tuyến tính
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

print("\n--- DỮ LIỆU SAU KHI ENCODE ---")
display(df_encoded.head())

# ==========================================
# 3. NGHIÊN CỨU TƯƠNG QUAN (CORRELATION)
# ==========================================
plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Ma trận tương quan (Correlation Matrix)")
plt.savefig(Path(__file__).resolve().parent / 'correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()

# ==========================================
# 4. KIỂM TRA & CHUẨN HÓA PHÂN PHỐI CỦA TARGET (CHARGES)
# ==========================================
# Biến mục tiêu 'charges' thường bị lệch phải (right-skewed)
fig, ax = plt.subplots(1, 4, figsize=(20, 5))

# 4.1. Phân phối gốc
sns.histplot(data=df, x='charges', kde=True, ax=ax[0])
ax[0].set_title('Gốc (Original)')

# 4.2. Log Transformation
log_charges = np.log1p(df['charges'])
sns.histplot(log_charges.to_numpy(), kde=True, ax=ax[1], color='green')
ax[1].set_title('Log Transformation')

# 4.3. Square Root Transformation
sqrt_charges = np.sqrt(df['charges'])
sns.histplot(sqrt_charges.to_numpy(), kde=True, ax=ax[2], color='orange')
ax[2].set_title('Square Root')

# 4.4. Box-Cox Transformation
# Giá trị target phải dương để dùng Box-Cox
boxcox_charges = stats.boxcox(df['charges'].to_numpy())
sns.histplot(boxcox_charges, kde=True, ax=ax[3], color='red')
ax[3].set_title('Box-Cox')

plt.tight_layout()
plt.savefig(Path(__file__).resolve().parent / 'charges_transformations.png', dpi=150, bbox_inches='tight')
plt.close()

# Quyết định chọn Log Transformation vì nó dễ đưa ngược về giá trị gốc và làm phân phối khá chuẩn
y = np.log1p(df_encoded['charges'])
X = df_encoded.drop('charges', axis=1)

# ==========================================
# 5. XÂY DỰNG MÔ HÌNH VỚI PIPELINE & CROSS VALIDATION
# ==========================================
# Khởi tạo K-Folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Khởi tạo các Pipeline (Gộp bước Scale và Model lại với nhau)
pipelines = {
    'Vanilla LR': Pipeline([('scaler', StandardScaler()), ('model', LinearRegression())]),
    'Lasso': Pipeline([('scaler', StandardScaler()), ('model', Lasso(random_state=42))]),
    'Ridge': Pipeline([('scaler', StandardScaler()), ('model', Ridge(random_state=42))]),
    'ElasticNetCV': Pipeline([('scaler', StandardScaler()), ('model', ElasticNetCV(cv=kf, random_state=42))])
}

# Tham số để GridSearch cho Lasso và Ridge
param_grids = {
    'Vanilla LR': {},
    'Lasso': {'model__alpha': [0.001, 0.01, 0.1, 1, 10]},
    'Ridge': {'model__alpha': [0.1, 1, 10, 100]},
    'ElasticNetCV': {} # ElasticNetCV tự động tìm params tốt nhất
}

results = {}

for name in pipelines.keys():
    # Grid Search kết hợp Cross Validation
    grid = GridSearchCV(pipelines[name], param_grids[name], cv=kf, scoring='r2', n_jobs=-1)
    grid.fit(X, y)
    
    best_model = grid.best_estimator_
    
    # Dự đoán bằng Cross Validation
    y_pred = cross_val_predict(best_model, X, y, cv=kf)
    
    # Tính metrics (Đang tính trên thang đo Log)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    results[name] = {'Best Params': grid.best_params_, 'R2': r2, 'RMSE (Log)': rmse}

print("\n--- KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH ---")
results_df = pd.DataFrame(results).T
display(results_df)

# Kiểm tra trọng số (Coefficients) của mô hình Ridge tốt nhất để tìm Insight
best_ridge = GridSearchCV(pipelines['Ridge'], param_grids['Ridge'], cv=kf).fit(X, y).best_estimator_
coefs = pd.Series(best_ridge.named_steps['model'].coef_, index=X.columns).sort_values(ascending=False)
print("\n--- TRỌNG SỐ (COEFFICIENTS) CỦA RIDGE REGRESSION ---")
print(coefs)