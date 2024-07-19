import os
import shutil
import random

def split_dataset(source_folder, destination_folder, train_ratio=0.9, val_ratio=0.1):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Prepare directories for train, validation, and test datasets
    train_dir = os.path.join(destination_folder, 'train')
    val_dir = os.path.join(destination_folder, 'val')
    test_dir = os.path.join(destination_folder, 'test')

    # Create image and label subdirectories in train, val, test directories
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Get all image files
    image_files = [f for f in os.listdir(source_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(image_files)

    # Calculate split indices
    total_images = len(image_files)
    train_end = int(total_images * train_ratio)
    val_end = train_end + int(total_images * val_ratio)

    # Distribute files into train, val, and test sets
    train_images = image_files[:train_end]
    val_images = image_files[train_end:val_end]
    test_images = image_files[val_end:]

    # Function to copy paired image and label files
    def copy_files(files, dir_path):
        for img_file in files:
            base_name = os.path.splitext(img_file)[0]

            # Define source paths for image and JSON file
            img_src_path = os.path.join(source_folder, img_file)
            json_src_path = os.path.join(source_folder, base_name + '.json')

            # Define destination paths for image and JSON file
            img_dst_path = os.path.join(dir_path, 'images', img_file)
            json_dst_path = os.path.join(dir_path, 'labels', base_name + '.json')

            # Copy image file
            shutil.copy2(img_src_path, img_dst_path)
            
            # Check if JSON file exists and copy
            if os.path.exists(json_src_path):
                shutil.copy2(json_src_path, json_dst_path)

    # Copy files to their respective directories
    copy_files(train_images, train_dir)
    copy_files(val_images, val_dir)
    copy_files(test_images, test_dir)

    print(f"Files have been successfully split into train, validation, and test datasets in {destination_folder}.")

# Usage
source_folder = 'yolo'

for i in range(1, 6):  # Repeat splitting for 5 different datasets
    destination_folder = f'datasets{i}'
    random.seed(i)  # Set a different seed for each split
    split_dataset(source_folder, destination_folder)
