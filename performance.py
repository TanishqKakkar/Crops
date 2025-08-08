import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# === Parameters ===
IMG_SIZE = 224  # or 128 depending on your model
BATCH_SIZE = 32
DATA_DIR = r'D:\crops\test\tomato'  # Replace with your dataset path

# === Load the trained model ===
model = load_model(r"D:\crops\models\cnn\tomato_disease_model2.h5")  # Adjust path if needed

# === Validation Data Generator ===
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# === Predictions ===
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

# === Confusion Matrix ===
cm = confusion_matrix(y_true, y_pred)

# === Plot Confusion Matrix ===
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.show()

# === Classification Report ===
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_labels, digits=3))