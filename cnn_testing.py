import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# === 🔧 Config ===
model_dir = r"D:\crops\models\custom"
test_dir = r"D:\crops\test"
img_size = (224, 224)

# === 🧪 Crop: Tomato Only ===
crop = "potato"
crop_path = os.path.join(test_dir, crop)

model_path = os.path.join(model_dir, "potato_custom_cnn.h5")  # 🛠️ removed extra space
if not os.path.exists(model_path):
    print(f"⚠️ No model found at {model_path}, aborting.")
    exit()

model = load_model(model_path)

# === 🔁 Auto-generate class_names from folder structure ===
class_names = []
for folder in sorted(os.listdir(crop_path)):
    if os.path.isdir(os.path.join(crop_path, folder)):
        if "___" in folder:
            label = folder.split("___")[1]
        else:
            label = folder
        label = label.lower().strip().replace(" ", "_")
        class_names.append(label)

print(f"✅ Detected classes for Tomato: {class_names}")

true_labels = []
pred_labels = []

# === 🔁 Evaluate All Images
for folder in os.listdir(crop_path):
    folder_path = os.path.join(crop_path, folder)
    if not os.path.isdir(folder_path):
        continue

    if "___" in folder:
        true_label = folder.split("___")[1]
    else:
        true_label = folder
    true_label = true_label.lower().strip().replace(" ", "_")

    for img_file in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_file)

        try:
            img = load_img(img_path, target_size=img_size)
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            prediction = model.predict(img_array, verbose=0)
            pred_index = np.argmax(prediction)
            pred_label = class_names[pred_index]

            true_labels.append(true_label)
            pred_labels.append(pred_label)

        except Exception as e:
            print(f"⚠️ Failed on {img_path}: {e}")

# === ✅ Evaluate Results
if len(true_labels) == 0:
    print(f"⚠️ No valid images processed for crop: {crop}.")
else:
    cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='YlGnBu')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {crop}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    print(f"\n📊 Classification Report for {crop}:")
    print(classification_report(true_labels, pred_labels, labels=class_names, target_names=class_names, zero_division=0))
