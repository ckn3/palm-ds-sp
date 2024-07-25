import ultralytics
ultralytics.checks()

import numpy as np
from ultralytics import YOLO, RTDETR
from ultralytics import YOLOv10
import cv2
import time
import os
import csv
import rasterio
from rasterio.windows import Window
from PIL import Image
from tqdm import tqdm
import pandas as pd
from shapely.geometry import box
import glob
import shutil

def inference_on_image(image_path):
    img = Image.open(image_path)
    results = model.predict(img, save=False, imgsz=800, conf=0.25)
    return results

# Function to save bounding box information to CSV and YOLO format txt file
def save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y):
    
    with open(predictions_csv, mode='a', newline='') as csv_file:  # Change mode to 'a' for appending
        csv_writer = csv.writer(csv_file)

        with open(local_txt_path, 'w') as txt_file:
            for result in results:
                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    for i, box in enumerate(result.boxes.xyxy):
                        # box coordinates
                        x1, y1, x2, y2 = map(float, box.tolist()[:4])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        lon, lat = rasterio.transform.xy(transform, center_y, center_x, offset='center')
                        width = x2 - x1
                        height = y2 - y1
                        confidence = result.boxes.conf[i].item()

                        # Normalized coordinates for YOLO format
                        norm_x_center = center_x / crop_size
                        norm_y_center = center_y / crop_size
                        norm_width = width / crop_size
                        norm_height = height / crop_size

                        # class information
                        class_id = int(result.boxes.cls[i].item())
                        class_name = result.names[class_id]

                        # Append to predictions CSV
                        csv_writer.writerow([lon, lat, width, height, class_name, x, y, center_x, center_y, confidence])

                        # Write YOLO format to txt file
                        txt_file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

def remove_redundant_predictions(csv_path, output_csv_path, transform, stride):
    df = pd.read_csv(csv_path)
    
    # Convert lon/lat to pixel coordinates and calculate bounding boxes
    df['y_center'], df['x_center'] = zip(*df.apply(lambda row: rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude']), axis=1))
    df['x1'] = df['x_center'] - (df['Width'] / 2)
    df['y1'] = df['y_center'] - (df['Height'] / 2)
    df['x2'] = df['x_center'] + (df['Width'] / 2)
    df['y2'] = df['y_center'] + (df['Height'] / 2)
    df['box'] = df.apply(lambda row: box(row['x1'], row['y1'], row['x2'], row['y2']), axis=1)
    df['area'] = df.apply(lambda row: row['box'].area, axis=1)
    
    # Sort boxes by area in descending order to prioritize larger boxes
    df_sorted = df.sort_values(by='area', ascending=False)
    
    keep_boxes = []  # List to keep boxes that are not redundant
    box_info = []  # To store box and their x, y indices

    for _, current_row in df_sorted.iterrows():
        current_box = current_row['box']
        current_x = current_row['X']
        current_y = current_row['Y']

        # Limit the search to nearby entries based on the col and row
        for kept_box, kept_x, kept_y in box_info:
            if abs(kept_x - current_x) <= crop_size and abs(kept_y - current_y) <= crop_size:
                intersection = current_box.intersection(kept_box)
                intersection_area = intersection.area
                if intersection_area / kept_box.area > 0.9:
                    keep_boxes.remove(kept_box)
                    box_info = [(box, x, y) for box, x, y in box_info if box != kept_box]
                    print(f"Removed overlapping box due to proximity: {intersection_area / kept_box.area}")

        # Add the current box if it is not already included in the kept boxes
        if current_box not in keep_boxes:
            keep_boxes.append(current_box)
            box_info.append((current_box, current_x, current_y))

    # Filter the dataframe to include only boxes that are kept
    df_filtered = df[df['box'].isin(keep_boxes)]
    df_filtered[['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class', 'X', 'Y', 'Xc', 'Yc', 'Conf']].to_csv(output_csv_path, index=False)
    print(f"Filtered predictions saved to {output_csv_path}")

# Get the model name and path
model_name = input("Enter the model name: ")
model_path = input("Enter the model path: ")
model = YOLOv10(model_path)

# Define parameters
crop_size = 800
stride = 400
site_number = input("Enter the site number: ")
site_directory = f"images/site{site_number}"
tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
if not tif_files:
    print(f"No TIFF files found in {site_directory}.")
    exit()
tif_file = tif_files[0]
input_path = os.path.join(site_directory, tif_file)
base_name = os.path.splitext(os.path.basename(input_path))[0]
save_dir = os.path.join(site_directory, f'cropped_{base_name}')
predictions_csv = os.path.join(site_directory, f'predictions_{base_name}_{model_name}.csv')

# Make directories and CSV file
os.makedirs(save_dir, exist_ok=True)
with open(predictions_csv, 'w', newline='') as file:  # Overwrite mode
    writer = csv.writer(file)
    writer.writerow(['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class', 'X', 'Y', 'Xc', 'Yc', 'Conf'])

# Open and process the full landscape image

start_time1 = time.time()

with rasterio.open(input_path) as src:
    image_width, image_height = src.width, src.height
    for y in tqdm(range(0, src.height - crop_size + 1, stride), desc="Processing"):
        for x in range(0, src.width - crop_size + 1, stride):
            window = Window(x, y, crop_size, crop_size)
            transform = src.window_transform(window)
            crop_image_path = os.path.join(save_dir, f"crop_{x}_{y}.tif")
            local_txt_path = os.path.splitext(crop_image_path)[0] + '.txt'
            with rasterio.open(crop_image_path, 'w', driver='GTiff', height=crop_size, width=crop_size, count=src.count, dtype=src.dtypes[0], transform=transform) as dst:
                dst.write(src.read(window=window))

            # Perform inference and save results
            results = inference_on_image(crop_image_path)
            save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size, x, y)


end_time1 = time.time()
print("Inference and result saving complete.")

# Remove redundant predictions
csv_path = os.path.join(site_directory, f'predictions_{base_name}_{model_name}.csv')
output_csv_path = os.path.join(site_directory, f'filtered_{base_name}_{model_name}.csv')

# Read the TIFF to get the transform
with rasterio.open(input_path) as src:
    transform = src.transform

print("Cleaning and result saving complete.")
print(f"Total inference time: {end_time1 - start_time1} seconds")

# Delete the CSV file at csv_path
# if os.path.exists(csv_path):
#     os.remove(csv_path)
#     print(f"Deleted CSV file at {csv_path}")
# else:
#     print(f"CSV file at {csv_path} does not exist.")

# Delete the folder that saved txt and tif images
if os.path.exists(save_dir):
    # Delete all files and subdirectories in the directory
    for filename in os.listdir(save_dir):
        file_path = os.path.join(save_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')
    
    # Delete the directory itself
    shutil.rmtree(save_dir)
    print(f"Deleted folder at {save_dir}")
else:
    print(f"Folder at {save_dir} does not exist.")