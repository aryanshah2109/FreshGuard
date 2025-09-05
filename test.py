from tensorflow.keras.models import load_model
import pickle
import numpy as np
from PIL import Image

model = load_model(r"D:\Projects\FreshGuard\models\freshguard_model.keras")

with open(r"D:\Projects\FreshGuard\models\freshguard_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

int_to_label = metadata["int_to_label"]

img_path = r"D:\Projects\FreshGuard\data\Banana_Spoiled\IMG_2108.jpg"
img = Image.open(img_path).convert("RGB").resize((224, 224))
img_array = np.expand_dims(np.array(img)/255.0, axis=0)

preds = model.predict(img_array)
pred_class = np.argmax(preds[0])
confidence = np.max(preds[0])

print(f"Prediction: {int_to_label[pred_class]}, Confidence: {confidence*100:.2f}%")
