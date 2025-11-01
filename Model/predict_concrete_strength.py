# ====================================================
# Concrete Strength Prediction - User Input Interface
# ====================================================

import joblib
import numpy as np
import os

# -------------------------------
# 1️⃣ Load the trained model
# -------------------------------
model_path = r"D:\Concrete Strength Prediction\Model\concrete_strength_xgb_model.pkl"

if not os.path.exists(model_path):
    print(f"[ERROR] Model file not found at: {model_path}")
    print("Please train the model first using model.py")
    exit()

model = joblib.load(model_path)
print("\n Model loaded successfully!\n")

# -------------------------------
# 2️⃣ Take user input
# -------------------------------
print("Enter the following details for Concrete Mix Design:")
print("----------------------------------------------------")

try:
    cement = float(input("Cement (kg/m³): "))
    blast_furnace_slag = float(input("Blast Furnace Slag (kg/m³): "))
    fly_ash = float(input("Fly Ash (kg/m³): "))
    water = float(input("Water (kg/m³): "))
    superplasticizer = float(input("Superplasticizer (kg/m³): "))
    coarse_aggregate = float(input("Coarse Aggregate (kg/m³): "))
    fine_aggregate = float(input("Fine Aggregate (kg/m³): "))
    age = float(input("Age of Concrete (days): "))

except ValueError:
    print("\n Invalid input detected. Please enter only numeric values.")
    exit()

# -------------------------------
# 3️⃣ Prepare data for prediction
# -------------------------------
input_features = np.array([[cement, blast_furnace_slag, fly_ash, water,
                            superplasticizer, coarse_aggregate, fine_aggregate, age]])

# -------------------------------
# 4️⃣ Make prediction
# -------------------------------
predicted_strength = model.predict(input_features)[0]

# -------------------------------
# 5️⃣ Display result
# -------------------------------
print("\n----------------------------------------------------")
print("🔹 Predicted Concrete Compressive Strength:")
print(f"   ➤ {predicted_strength:.2f} MPa")
print("----------------------------------------------------")
print("\n✅ Prediction completed successfully.")
