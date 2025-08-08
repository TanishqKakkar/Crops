import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# === 📄 Step 1: Load your CSV (update this path)
csv_path = r"D:\crops\csv with images\tomato_disease_with_image_path.csv"
df = pd.read_csv(csv_path)

# === 🏷️ Step 2: Generate LabelEncoder from 'predicted_disease'
label_encoder = LabelEncoder()
label_encoder.fit(df['predicted_disease'])

# === 🌡️ Step 3: Generate Scaler from 'temp' and 'humidity'
scaler = StandardScaler()
scaler.fit(df[['temp', 'humidity']])

# === 💾 Step 4: Save both objects
with open("D:/crops/PKL/label encoder/tomato_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("D:/crops/PKL/feature scale/tomato_feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Label encoder and scaler saved successfully.")
