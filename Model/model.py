# ==========================================
# Concrete Strength Prediction using XGBoost
# ==========================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

# -------------------------------
# 1️⃣ Load Dataset
# -------------------------------
file_path = r"D:\Concrete Strength Prediction\Model\Concrete_Data.csv"

if not os.path.exists(file_path):
    print(f"[ERROR] File not found at: {file_path}")
    print("Please check the dataset path and try again.")
    exit()

data = pd.read_csv(file_path)
data.columns = data.columns.str.strip()  # remove unwanted spaces

print("[INFO] Data loaded successfully.")
print("Columns:", data.columns.tolist())
print(data.head(), "\n")

# -------------------------------
# 2️⃣ Define Features and Target
# -------------------------------
X = data[['cement', 'blast_furnace_slag', 'fly_ash', 'water',
          'superplasticizer', 'coarse_aggregate', 'fine_aggregate', 'age']]
y = data['concrete_compressive_strength']

# -------------------------------
# 3️⃣ Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4️⃣ Train XGBoost Model
# -------------------------------
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 5️⃣ Evaluate Model
# -------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # compatible with all versions

print("Model Evaluation Results:")
print(f"R² Score       : {r2:.3f}")
print(f"MSE (Error)    : {mse:.3f}")
print(f"RMSE (Root MSE): {rmse:.3f}\n")

# -------------------------------
# 6️⃣ Visualization
# -------------------------------
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6, label="Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', lw=2, label="Perfect Prediction Line")
plt.xlabel("Actual Strength (MPa)")
plt.ylabel("Predicted Strength (MPa)")
plt.title("Concrete Strength Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -------------------------------
# 7️⃣ Save Model
# -------------------------------
save_path = r"D:\Concrete Strength Prediction\Model\concrete_strength_xgb_model.pkl"
joblib.dump(model, save_path)
print(f"[INFO] Model saved successfully at: {save_path}")
