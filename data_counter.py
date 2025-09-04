import os

# Path to your dataset
dataset_dir = r'D:\Projects\FreshGuard\data_files'

# Initialize a dictionary to store the count of images per class
image_counts = {}

# Iterate over each class folder
for class_name in os.listdir(dataset_dir):
    class_path = os.path.join(dataset_dir, class_name)
    
    # Ensure it's a directory
    if os.path.isdir(class_path):
        # Count the number of image files in the class folder
        image_count = len([f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        image_counts[class_name] = image_count

# Display the image counts per class
for class_name, count in image_counts.items():
    print(f"Class: {class_name}, Images: {count}")
