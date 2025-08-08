import numpy as np
import pandas as pd
import os
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# === 📄 Load Dataset ===
df = pd.read_csv(r"D:\crops\csv\fusion csv\potato_disease_with_image_path.csv")  # columns: image_path, temp, humidity, predicted_disease

# === ♻️ Reuse from MLP Phase ===
with open(r"D:/crops/PKL/feature scale/potato_feature_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open(r"D:/crops/PKL/label encoder/potato_label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# === 🏷️ Encode Labels ===
df['label_encoded'] = label_encoder.transform(df['predicted_disease'])
y = to_categorical(df['label_encoded'])

# === 🌡️ Scale Climate Data ===
X_climate = scaler.transform(df[['temp', 'humidity']])

# === 🖼️ Preprocess Images ===
def preprocess_image(path, size=(224, 224)):
    img = load_img(path, target_size=size)
    return img_to_array(img) / 255.0

# === 🧼 Clean Image Paths and Load Images ===
df = df[df['image_path'].apply(os.path.exists)].reset_index(drop=True)
X_images = np.array([preprocess_image(p) for p in df['image_path']])
X_climate = X_climate[:len(X_images)]
y = y[:len(X_images)]

# === ✂️ Train-Test Split ===
from sklearn.model_selection import train_test_split
X_img_train, X_img_test, X_cli_train, X_cli_test, y_train, y_test = train_test_split(
    X_images, X_climate, y, test_size=0.2, random_state=42
)

# === 🧠 Load CNN & MLP Models ===
cnn_model = load_model(r"D:\crops\models\climate dependent cnn\potato_disease_model3.h5")
cnn_feature_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)

mlp_model = load_model(r"D:\crops\models\mlp\apple_mlp_model.h5")
mlp_feature_model = Model(inputs=mlp_model.input, outputs=mlp_model.layers[-2].output)

# === 🔍 Extract Features ===
X_img_feat_train = cnn_feature_model.predict(X_img_train, verbose=0)
X_img_feat_test = cnn_feature_model.predict(X_img_test, verbose=0)
X_cli_feat_train = mlp_feature_model.predict(X_cli_train, verbose=0)
X_cli_feat_test = mlp_feature_model.predict(X_cli_test, verbose=0)

# === 🔗 Fusion Model Definition ===
img_input = Input(shape=(X_img_feat_train.shape[1],))
cli_input = Input(shape=(X_cli_feat_train.shape[1],))
combined = Concatenate()([img_input, cli_input])
x = Dense(64, activation='relu')(combined)
x = Dropout(0.3)(x)
output = Dense(y.shape[1], activation='softmax')(x)

fusion_model = Model(inputs=[img_input, cli_input], outputs=output)
fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# === 🚀 Train Model ===
fusion_model.fit(
    [X_img_feat_train, X_cli_feat_train],
    y_train,
    validation_split=0.1,
    epochs=25,
    batch_size=32
)

# === 📊 Evaluate Model ===
loss, acc = fusion_model.evaluate([X_img_feat_test, X_cli_feat_test], y_test)
print(f"\n✅ Fusion Model Accuracy: {acc:.4f}")

# === 💾 Save Model ===
fusion_model.save("D:/crops/models/fusion/apple_fusion_model.h5")
print("✅ Fusion model saved.")
