import os
import shutil

# Input: Your current dataset root folder
SOURCE_DIR = "D:\Projects\FreshGuard\data"
# Output: Flattened dataset folder
DEST_DIR = "D:\Projects\FreshGuard\data_files"

os.makedirs(DEST_DIR, exist_ok=True)

# Walk through each fruit
for fruit in os.listdir(SOURCE_DIR):
    fruit_path = os.path.join(SOURCE_DIR, fruit)
    if not os.path.isdir(fruit_path):
        continue

    # Walk through each freshness condition
    for condition in os.listdir(fruit_path):
        condition_path = os.path.join(fruit_path, condition)
        if not os.path.isdir(condition_path):
            continue

        # Create flattened class folder (Fruit_Condition)
        class_name = f"{fruit}_{condition.replace(' ', '')}"  # remove spaces in names
        dest_class_dir = os.path.join(DEST_DIR, class_name)
        os.makedirs(dest_class_dir, exist_ok=True)

        # Walk through all subfolders (Day 1, Day 2, ...)
        for day in os.listdir(condition_path):
            day_path = os.path.join(condition_path, day)
            if not os.path.isdir(day_path):
                continue

            # Copy all images into the flattened class folder
            for img_file in os.listdir(day_path):
                src_img_path = os.path.join(day_path, img_file)

                # Only copy valid image files
                if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                dest_img_path = os.path.join(dest_class_dir, img_file)

                # Ensure unique filenames (avoid overwriting)
                base, ext = os.path.splitext(img_file)
                counter = 1
                while os.path.exists(dest_img_path):
                    dest_img_path = os.path.join(dest_class_dir, f"{base}_{counter}{ext}")
                    counter += 1

                shutil.copy(src_img_path, dest_img_path)

print("✅ Dataset converted successfully into", DEST_DIR)
