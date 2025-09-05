from keras.models import load_model

# 1️⃣ Load the old model (compile=False avoids InputLayer issues temporarily)
old_model_path = r"D:\Projects\FreshGuard\models\freshguard_model.h5"
model = load_model(old_model_path, compile=False)

# 2️⃣ Re-save in TF 2.12 format
new_model_path = r"D:\Projects\FreshGuard\models\freshguard_model_v2.h5"
model.save(new_model_path)

print(f"✅ Model successfully re-saved as {new_model_path}")
