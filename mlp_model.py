# === 📦 Imports ===
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import pickle

# === 📄 Load Your CSV ===
df = pd.read_csv(r"D:\crops\csv\mlp csv\cotton_disease_predicted.csv")  # change path as needed

# === ✅ Preprocess Labels ===
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['predicted_disease'])
y_cat = to_categorical(y_encoded)

# === 🌡️ Preprocess Climate Features ===
X = df[['temp', 'humidity']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === ✂️ Split Dataset ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_cat, test_size=0.2, random_state=42)

# === 🧠 MLP Architecture ===
input_layer = Input(shape=(2,))
x = Dense(64, activation='relu')(input_layer)
x = Dropout(0.2)(x)
x = Dense(32, activation='relu')(x)
output_layer = Dense(y_cat.shape[1], activation='softmax')(x)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 🚀 Train Model ===
model.fit(X_train, y_train, validation_split=0.1, epochs=25, batch_size=16)

# === ✅ Evaluate Model ===
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ MLP Model Accuracy: {acc:.4f}")

# === 💾 Save Model Weights, Scaler, LabelEncoder ===
model.save("D:/crops/models/mlp/cotton_mlp_model.h5")

with open("D:/crops/PKL/feature scale/cotton_feature_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("D:/crops/PKL/label encoder/cotton_mlp_label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("\n✅ Model weights, scaler, and label encoder saved.")
