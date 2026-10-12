import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. 讀取資料 ---
CSV_PATH = 'svmFull.csv'
print(f"正在讀取資料: {CSV_PATH} ...")
df = pd.read_csv(CSV_PATH)

# 假設 'label' 是你要預測的目標欄位，其餘為特徵
X = df.drop('label', axis=1)
y = df['label']

# 切分訓練集 (80%) 與測試集 (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. 特徵標準化 (SVM 必做步驟) ---
print("正在進行特徵標準化...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. 訓練 SVM 模型 ---
print("正在訓練 SVM 模型 (這可能需要一點時間)...")
# kernel 可以選擇 'linear' (線性), 'rbf' (非線性), 'poly' 等。這裡先用最泛用的 'rbf'
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train_scaled, y_train)

# --- 4. 評估模型準確度 ---
y_pred = svm_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ 模型訓練完成！測試集準確度: {accuracy * 100:.2f}%\n")
print("詳細分類報告:")
print(classification_report(y_test, y_pred))

# --- 5. 儲存權重檔與 Scaler ---
MODEL_OUTPUT = 'svm_model.pkl'
SCALER_OUTPUT = 'svm_scaler.pkl'

joblib.dump(svm_model, MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)

print(f"💾 SVM 模型已成功儲存至: {MODEL_OUTPUT}")
print(f"💾 Scaler 已成功儲存至: {SCALER_OUTPUT}")