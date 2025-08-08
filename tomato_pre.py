import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

# Parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_INITIAL = 10
EPOCHS_FINE_TUNE = 5
DATA_DIR = r"D:\crops\dataset climatic dependent\cotton"

# Step 1: Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    DATA_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Step 2: Base Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False  # Freeze during initial training

# Step 3: Custom Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Step 4: Compile & Initial Training
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🟢 Training model (frozen base)...")
history1 = model.fit(
    train_generator,
    epochs=EPOCHS_INITIAL,
    validation_data=val_generator
)

# Step 5: Unfreeze and Fine-tune
base_model.trainable = True

# Lower learning rate for fine-tuning
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n🟡 Fine-tuning model (unfrozen base)...")
history2 = model.fit(
    train_generator,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=val_generator
)

# Step 6: Combine Training History
def combine_histories(h1, h2):
    history = {}
    for key in h1.history:
        history[key] = h1.history[key] + h2.history[key]
    return history

history = combine_histories(history1, history2)

# Step 7: Plot Accuracy & Loss
plt.figure(figsize=(8, 5))
plt.plot(history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(history['loss'], label='Train Loss', marker='o')
plt.plot(history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Step 8: Final Evaluation
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"\n✅ Final Validation Accuracy: {val_accuracy:.4f}")
print(f"✅ Final Validation Loss: {val_loss:.4f}")

# Step 9: Save the model to .h5 file
model.save("cotton_disease_model3.h5")
print("\n💾 Model saved as 'cotton_disease_model.h5'")