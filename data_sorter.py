import os
import shutil
import pandas as pd

# Set your dataset root folder here
DATASET_DIR = "D:\Projects\FreshGuard\data"   # change to your dataset path

log = []

for fruit in os.listdir(DATASET_DIR):
    fruit_path = os.path.join(DATASET_DIR, fruit)
    if not os.path.isdir(fruit_path):
        continue

    # Loop over categories: Fresh / SlightlySpoiled / Spoiled
    for category in os.listdir(fruit_path):
        category_path = os.path.join(fruit_path, category)
        if not os.path.isdir(category_path):
            continue

        # Walk through all subdirectories (Day folders)
        for root, dirs, files in os.walk(category_path):
            for file in files:
                old_path = os.path.join(root, file)
                new_name = file

                # If file already exists in target, rename to avoid conflict
                target_path = os.path.join(category_path, new_name)
                counter = 1
                while os.path.exists(target_path):
                    name, ext = os.path.splitext(file)
                    new_name = f"{name}_{counter}{ext}"
                    target_path = os.path.join(category_path, new_name)
                    counter += 1

                # Move file to category root
                shutil.move(old_path, target_path)

                # Save log
                log.append({"fruit": fruit,
                            "category": category,
                            "original_path": old_path,
                            "new_path": target_path})

        # After flattening, remove empty Day folders
        for root, dirs, files in os.walk(category_path, topdown=False):
            for d in dirs:
                day_folder = os.path.join(root, d)
                if not os.listdir(day_folder):  # delete if empty
                    os.rmdir(day_folder)

# Save mapping log
df = pd.DataFrame(log)
df.to_csv("dataset_log.csv", index=False)

print("✅ Dataset flattened successfully! Log saved as dataset_log.csv")
