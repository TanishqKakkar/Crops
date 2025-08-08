import os
import numpy as np
import pickle
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# === 📌 Paths (change as per your crop) ===
cnn_model_path = r"D:\crops\models\cnn\cotton_disease_model2.h5"
mlp_weights_path = r"D:\crops\models\mlp\cotton_climate_mlp_model.h5"
fusion_model_path = r"D:\crops\models\fusion\cotton_fusion_model.h5"
scaler_path = r"D:\crops\PKL\feature scale\cotton_feature_scaler.pkl"
encoder_path = r"D:\crops\PKL\label encoder\cotton_label_encoder.pkl"

# === 📁 Folder to Test ===
test_folder = r"D:\crops\test\cotton\Powdery mildew"  # 📝 put your folder path here

# === 🌡️ Manual Inputs for All Images ===
temperature = 32
humidity = 88

# === 📦 Load Models ===
cnn_model = load_model(cnn_model_path)
fusion_model = load_model(fusion_model_path)

# === 🧠 Rebuild MLP ===
mlp_input = Input(shape=(2,))
x = Dense(64, activation='relu')(mlp_input)
x = Dense(32, activation='relu')(x)
mlp_output = Dense(2, activation='softmax')(x)  # Adjust number of classes here
mlp_model = Model(inputs=mlp_input, outputs=mlp_output)
mlp_model.load_weights(mlp_weights_path)

# === 🔗 Load Scaler & LabelEncoder ===
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)
with open(encoder_path, "rb") as f:
    label_encoder = pickle.load(f)
class_names = label_encoder.classes_

# === 🔁 Helpers ===
def normalize_label(label):
    return label.strip().lower().replace(" ", "_")

def preprocess_image(path, size=(224, 224)):
    img = load_img(path, target_size=size)
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def preprocess_climate(temp, humidity):
    return scaler.transform([[temp, humidity]])

# === 🔍 Feature Extractors ===
cnn_feat_model = Model(inputs=cnn_model.input, outputs=cnn_model.layers[-2].output)
mlp_feat_model = Model(inputs=mlp_model.input, outputs=mlp_model.layers[-2].output)

# === 🔬 Evaluation ===
true_labels, cnn_preds, mlp_preds, fusion_preds = [], [], [], []
actual_label = normalize_label(os.path.basename(test_folder))

print(f"\n🔍 Testing folder: {test_folder}")
print(f"📌 Actual label: {actual_label}\n")

for img_file in os.listdir(test_folder):
    img_path = os.path.join(test_folder, img_file)

    try:
        # --- Inputs ---
        img_input = preprocess_image(img_path)
        climate_input = preprocess_climate(temperature, humidity)

        # --- CNN ---
        cnn_pred = cnn_model.predict(img_input, verbose=0)
        cnn_label = label_encoder.inverse_transform([np.argmax(cnn_pred)])[0]

        # --- MLP ---
        mlp_pred = mlp_model.predict(climate_input, verbose=0)
        mlp_label = label_encoder.inverse_transform([np.argmax(mlp_pred)])[0]

        # --- Fusion ---
        cnn_feat = cnn_feat_model.predict(img_input, verbose=0)
        mlp_feat = mlp_feat_model.predict(climate_input, verbose=0)
        fusion_pred = fusion_model.predict([cnn_feat, mlp_feat], verbose=0)
        fusion_label = label_encoder.inverse_transform([np.argmax(fusion_pred)])[0]

        # Append
        true_labels.append(actual_label)
        cnn_preds.append(cnn_label)
        mlp_preds.append(mlp_label)
        fusion_preds.append(fusion_label)

        print(f"✅ {img_file} → CNN: {cnn_label}, MLP: {mlp_label}, Fusion: {fusion_label}")

    except Exception as e:
        print(f"⚠️ Failed on {img_file}: {e}")

# === 📊 Evaluation Report ===
print("\n--- 🧪 Evaluation Summary ---")
if true_labels:
    print(f"\n🔷 CNN Accuracy: {accuracy_score(true_labels, cnn_preds):.2f}")
    print(f"🔷 MLP Accuracy: {accuracy_score(true_labels, mlp_preds):.2f}")
    print(f"🔷 Fusion Accuracy: {accuracy_score(true_labels, fusion_preds):.2f}")

    print("\n📄 Fusion Classification Report:")
    print(classification_report(true_labels, fusion_preds, labels=class_names, zero_division=0))
else:
    print("⚠️ No valid images processed.")
