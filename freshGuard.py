# -*- coding: utf-8 -*-
"""FreshGuard-final"""

# -------------------------------
# 1. Imports and GPU setup
# -------------------------------
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# -------------------------------
# 1a. Check GPU
# -------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is available and memory growth enabled")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")
print("Num GPUs Available: ", len(gpus))

# -------------------------------
# 2. Dataset paths & parameters
# -------------------------------
DATASET_DIR = "/kaggle/input/freshguard-dataset-2-0/data_files"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 20
FINE_TUNE_LAYERS = 50

# -------------------------------
# 3. Prepare file paths and labels
# -------------------------------
all_image_paths = []
all_labels = []

for class_name in os.listdir(DATASET_DIR):
    class_dir = os.path.join(DATASET_DIR, class_name)
    if os.path.isdir(class_dir):
        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                all_image_paths.append(os.path.join(class_dir, img_name))
                all_labels.append(class_name)

unique_labels = sorted(list(set(all_labels)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}
int_to_label = {i: label for i, label in enumerate(unique_labels)}
all_int_labels = [label_to_int[label] for label in all_labels]

X_train, X_val, y_train, y_val = train_test_split(
    all_image_paths, all_int_labels, test_size=0.2, random_state=42, stratify=all_int_labels
)

# -------------------------------
# 4. TF.data pipeline with augmentation
# -------------------------------
def preprocess_image(img_path, label, training=True):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0
    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_brightness(img, 0.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
    return img, tf.one_hot(label, len(unique_labels))

def create_dataset(image_paths, labels, training=True):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=1000)
    ds = ds.map(lambda x, y: preprocess_image(x, y, training), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = create_dataset(X_train, y_train, training=True)
val_ds = create_dataset(X_val, y_val, training=False)

# -------------------------------
# 5. Compute class weights
# -------------------------------
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

# -------------------------------
# 6. Build MobileNetV2 model
# -------------------------------
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = True

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(len(unique_labels), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()

# -------------------------------
# 7. Train the model
# -------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights
)

# -------------------------------
# 8. Prediction function
# -------------------------------
def predict_image(img_path, show_image=False):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img_array = tf.expand_dims(img, axis=0) / 255.0

    predictions = model.predict(img_array, verbose=0)
    pred_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    if show_image:
        plt.imshow(img.numpy().astype("uint8"))
        plt.axis("off")
        plt.title(f"Prediction: {int_to_label[pred_class]} ({confidence*100:.2f}%)")
        plt.show()

    return int_to_label[pred_class], confidence

# -------------------------------
# 9. Classification Report
# -------------------------------
y_true, y_pred = [], []

for imgs, labels in val_ds:
    preds = model.predict(imgs, verbose=0)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels.numpy(), axis=1))



print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=unique_labels))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

# -------------------------------
# 10. Test example predictions
# -------------------------------
example_paths = [
    "/kaggle/input/test-banana-spoiled-1/IMG_2115.jpg",
    "/kaggle/input/test-foodfresh/IMG_0244.jpg",
    "/kaggle/input/test-foodfresh/IMG_0382.jpg",
    "/kaggle/input/test-foodfresh/IMG_2247.jpg",
    "/kaggle/input/test-foodfresh/IMG_5188.jpg"
]

for path in example_paths:
    label, conf = predict_image(path, show_image=True)
    print(f"{path} => Prediction: {label}, Confidence: {conf*100:.2f}%")

import os
import pickle

# Kaggle working directory
MODEL_DIR = "/kaggle/working/"

# Save in native Keras format
model_save_path = os.path.join(MODEL_DIR, "freshguard_model.keras")
model.save(model_save_path)  # ⚡ no save_format needed
print(f"✅ Model saved at: {model_save_path}")

# Save metadata
metadata = {
    "label_to_int": label_to_int,
    "int_to_label": int_to_label,
    "history": history.history
}
metadata_save_path = os.path.join(MODEL_DIR, "freshguard_metadata.pkl")
with open(metadata_save_path, "wb") as f:
    pickle.dump(metadata, f)
print(f"✅ Metadata saved at: {metadata_save_path}")
