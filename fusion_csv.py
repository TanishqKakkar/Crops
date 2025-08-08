import pandas as pd
import os

# === 🔧 Paths ===
env_csv_path = r"D:\crops\csv\mlp csv\tomato_disease_predicted.csv"  # 🔁 Your climate data
image_csv_path = r"D:\crops\csv\fusion csv\tomato_images.csv"  # 🔁 Image path CSV
output_path = r"D:\crops\csv\fusion csv\tomato_disease_with_image_path.csv"

# === 📄 Load CSVs
df_env = pd.read_csv(env_csv_path)
df_img = pd.read_csv(image_csv_path)

# === 🔁 Merge based on matching disease name
# Normalize labels for better matching (case/space insensitive)
df_env['predicted_disease_norm'] = df_env['predicted_disease'].str.strip().str.lower().str.replace(" ", "_")
df_img['label_norm'] = df_img['label'].str.strip().str.lower().str.replace(" ", "_")

# === 🔁 Left join based on disease name
df_merged = pd.merge(df_env, df_img, left_on='predicted_disease_norm', right_on='label_norm', how='left')

# === 🧹 Clean up
df_merged.drop(columns=['predicted_disease_norm', 'label_norm', 'label'], inplace=True)

# === 💾 Save merged dataset
df_merged.to_csv(output_path, index=False)
print(f"✅ Merged CSV saved to: {output_path}")
