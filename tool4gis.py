import shutil
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import Polygon, box, Point
import os
import pandas as pd
from PIL import Image
from ultralytics import YOLO, SAM
import csv
import torch
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def draw_polygon(image):
    points = []
    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x * 10, y * 10))  # Scale up the points because the image was downscaled
            cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
            if len(points) > 1:
                cv2.line(image, (points[-2][0] // 10, points[-2][1] // 10), (points[-1][0] // 10, points[-1][1] // 10), (0, 255, 0), 2)
            cv2.imshow("image", image)
    cv2.imshow("image", image)
    cv2.setMouseCallback("image", click_event)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()
    if key == 13 and len(points) > 2:  # Enter key
        return Polygon(points)
    return None

def load_yolo_model(model_path):
    return YOLO(model_path)

def inference_on_image(model, image_path):
    img = Image.open(image_path)
    results = model.predict(img, save=False, imgsz=800, conf=0.5)
    return results

def save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size):
    with open(predictions_csv, mode='a', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        if os.stat(predictions_csv).st_size == 0:
            csv_writer.writerow(['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class'])  # Header
        with open(local_txt_path, 'w') as txt_file:
            for result in results:
                if result.boxes is not None and len(result.boxes.xyxy) > 0:
                    for i, box in enumerate(result.boxes.xyxy):
                        x1, y1, x2, y2 = map(float, box.tolist()[:4])
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        lon, lat = rasterio.transform.xy(transform, center_y, center_x, offset='center')
                        width = x2 - x1
                        height = y2 - y1
                        norm_x_center = center_x / crop_size
                        norm_y_center = center_y / crop_size
                        norm_width = width / crop_size
                        norm_height = height / crop_size
                        class_id = int(result.boxes.cls[i].item())
                        class_name = result.names[class_id]
                        csv_writer.writerow([lon, lat, width, height, class_name])
                        txt_file.write(f"{class_id} {norm_x_center} {norm_y_center} {norm_width} {norm_height}\n")

def remove_redundant_predictions(csv_path, output_csv_path, transform):
    df = pd.read_csv(csv_path)
    df['y_center'], df['x_center'] = zip(*df.apply(lambda row: rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude']), axis=1))
    df['x1'] = df['x_center'] - (df['Width'] / 2)
    df['y1'] = df['y_center'] - (df['Height'] / 2)
    df['x2'] = df['x_center'] + (df['Width'] / 2)
    df['y2'] = df['y_center'] + (df['Height'] / 2)
    df['box'] = df.apply(lambda row: box(row['x1'], row['y1'], row['x2'], row['y2']), axis=1)
    df['area'] = df.apply(lambda row: row['box'].area, axis=1)
    
    df_sorted = df.sort_values(by='area', ascending=False)
    keep_boxes = []
    for _, current_row in df_sorted.iterrows():
        current_box = current_row['box']
        current_area = current_box.area
        width_to_height_ratio = current_row['Width'] / current_row['Height']
        
        if not (0.4 <= width_to_height_ratio <= 2.5) or current_area < 5000:
            continue  # Skip due to unacceptable width-to-height ratio or tiny area

        remove = False
        for kept_box in keep_boxes[:]:
            intersection = current_box.intersection(kept_box)
            intersection_area = intersection.area
            
            if intersection_area / kept_box.area > 0.9:  # A refined box is found, with 90%+ overlap, before the box is detected as redundant
                keep_boxes.remove(kept_box)
            elif intersection_area / current_area > 0.85:  # The current box is redundant, especially the nested ones
                remove = True
                break
            elif intersection_area / kept_box.area > 0.8:  # A refined box is found, with 80%+ overlap
                keep_boxes.remove(kept_box)

        if not remove:
            keep_boxes.append(current_box)

    df_filtered = df[df['box'].isin(keep_boxes)]
    df_filtered[['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class']].to_csv(output_csv_path, index=False)
    if os.path.exists(csv_path):
        os.remove(csv_path)

def plot_bounding_boxes(image, csv_path, transform):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        center_y, center_x = rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude'])
        width, height = row['Width'], row['Height']
        class_id = row['Predicted Class']
        color = 'blue' if class_id == 'Fan unk.' else 'red'
        x1, y1 = center_x - width / 2, center_y - height / 2
        x2, y2 = center_x + width / 2, center_y + height / 2
        if class_id == 'Fan unk.':
            color = (0, 0, 255)  # Numeric value for blue
        else:
            color = (255, 0, 0)  # Numeric value for red
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image

def apply_segmentation_to_landscape(site_directory, cropped_tif_path, sam_model_path, csv_path, polygon, minx, miny):
    input_tif_path = cropped_tif_path
    with rasterio.open(input_tif_path) as src:
        data = src.read([1, 2, 3])
        transform = src.transform
        mask_global = np.zeros((src.height, src.width), dtype=np.uint8)

    df = pd.read_csv(csv_path)
    df['y_center'], df['x_center'] = zip(*df.apply(lambda row: rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude']), axis=1))
    df['class_id'] = df['Predicted Class'].map({'Fan unk.': 1, 'Bottlebrush unk.': 2})

    model = SAM(sam_model_path)

    fan_count = 0
    bottlebrush_count = 0

    for _, row in df.iterrows():
        x, y, width, height = int(row['x_center']), int(row['y_center']), int(row['Width']), int(row['Height'])
        if not polygon.contains(Point(x+minx, y+miny)):
            continue 

        if row['Predicted Class'] == 'Fan unk.':
            fan_count += 1
        elif row['Predicted Class'] == 'Bottlebrush unk.':
            bottlebrush_count += 1
        x1, y1 = max(x - 400, 0), max(y - 400, 0)
        x2, y2 = min(x1 + 800, src.width), min(y1 + 800, src.height)
        xc = x - x1
        yc = y - y1

        cropped_image = data[:, y1:y2, x1:x2]
        if cropped_image.shape[1] < 800 or cropped_image.shape[2] < 800:
            pad_width = ((0, 0),  # No padding for bands
                         (0, max(0, 800 - cropped_image.shape[1])),  # Padding for rows
                         (0, max(0, 800 - cropped_image.shape[2])))  # Padding for columns
            cropped_image = np.pad(cropped_image, pad_width, mode='constant', constant_values=0)

        cropped_image = np.moveaxis(cropped_image, 0, -1)
        image_pil = Image.fromarray(np.uint8(cropped_image * 255))

        results = model(image_pil, bboxes=[[xc - width // 2, yc - height // 2, xc + width // 2, yc + height // 2 ]])

        for r in results:
            masks = r.masks.data
            mask_combined = torch.any(masks, dim=0).int() * row['class_id']
            mask_combined = mask_combined.cpu().numpy()
            print(x1, x2, y1, y2, x2-x1, y2-y1)
            print(mask_global[y1:y2, x1:x2].shape, mask_combined[0:y2-y1, 0:x2-x1].shape)
            # if x1 < x2 and y1 < y2:
            mask_global[y1:y2, x1:x2] = np.maximum(mask_global[y1:y2, x1:x2], mask_combined[0:y2-y1, 0:x2-x1])

    total_palms = fan_count + bottlebrush_count
    print(f"{total_palms} palm trees found within the polygon, {fan_count} are Fan unk., {bottlebrush_count} are Bottlebrush unk.")

    output_image_path = os.path.join(site_directory, f'Annotated.png')
    colors = ['black', 'blue', 'red']
    cmap = ListedColormap(colors)

    polygon_points = np.array([(int(x) - minx, int(y) - miny) for x, y in polygon.exterior.coords], dtype=np.int32).reshape((-1, 1, 2))

    # Draw the bounding boxes on the image
    img = np.moveaxis(data, 0, -1).copy()
    img = plot_bounding_boxes(img, csv_path, transform)

    cv2.polylines(img, [polygon_points], True, (0, 255, 0), 5)
    plt.figure(figsize=(img.shape[1] / 100, img.shape[0] / 100), dpi=100)
    plt.imshow(img)
    plt.imshow(mask_global, cmap=cmap, alpha=0.5)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Segmentation overlay saved to {output_image_path}")

def main():
    site_number = input("Enter the site number: ")
    model_path = input("Enter the path to the YOLO model (.pt file) [default: best.pt]: ") or "best.pt"
    model = load_yolo_model(model_path)
    sam_model_path = input("Enter the path to the SAM model (.pt file) [default: mobile_sam.pt]: ") or "mobile_sam.pt"

    site_directory = f"images/site{site_number}"
    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"No TIFF files found in {site_directory}.")
        return
    tif_file = tif_files[0]
    tif_path = os.path.join(site_directory, tif_file)
    with rasterio.open(tif_path) as src:
        img_array = src.read([1, 2, 3], out_shape=(3, src.height // 10, src.width // 10))
        img_array = np.moveaxis(img_array, 0, -1)
        img_for_polygon = np.uint8((img_array / img_array.max()) * 255)
        cv_image = cv2.cvtColor(img_for_polygon, cv2.COLOR_RGB2BGR)
        polygon = draw_polygon(cv_image)

    if not polygon:
        print("No valid polygon drawn.")
        return

    minx, miny, maxx, maxy = map(int, polygon.bounds)
    crop_size = 800
    stride = 400

    predictions_csv = f"images/site{site_number}_predictions.csv"
    os.makedirs(f"images/site{site_number}_crops", exist_ok=True)

    with open(predictions_csv, 'w', newline='') as file:  # Open CSV file once to write header
        csv_writer = csv.writer(file)
        csv_writer.writerow(['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class'])

    with rasterio.open(tif_path) as src:
        for y in range(miny, maxy, stride):
            for x in range(minx, maxx, stride):
                if x + crop_size > src.width or y + crop_size > src.height:
                    continue
                window = Window(x, y, crop_size, crop_size)
                transform = src.window_transform(window)
                crop_image = src.read(window=window)
                crop_image_path = f"images/site{site_number}_crops/crop_{x}_{y}.tif"
                local_txt_path = crop_image_path.replace('.tif', '.txt')
                with rasterio.open(crop_image_path, 'w', driver='GTiff', height=crop_size, width=crop_size, count=src.count, dtype=src.dtypes[0], transform=transform) as dst:
                    dst.write(crop_image)
                results = inference_on_image(model, crop_image_path)
                save_bounding_box_info_to_csv_and_yolo_txt(results, predictions_csv, local_txt_path, transform, crop_size)

    if pd.read_csv(predictions_csv).empty:
        print("0 palms found. Skipping cleaning and segmentation.")
        return
    
    cleaned_csv_path = f"images/site{site_number}_cleaned_predictions.csv"
    remove_redundant_predictions(predictions_csv, cleaned_csv_path, transform)

    # Crop the TIFF to the polygon area
    with rasterio.open(tif_path) as src:
        data = src.read(window=Window.from_slices((miny, maxy), (minx, maxx)))
        cropped_transform = src.window_transform(Window.from_slices((miny, maxy), (minx, maxx)))
        cropped_tif_path = os.path.join(site_directory, f"cropped.tif")
        with rasterio.open(cropped_tif_path, 'w', driver='GTiff', height=maxy-miny, width=maxx-minx, count=src.count, dtype=src.dtypes[0], transform=cropped_transform) as dst:
            dst.write(data)

    # Apply segmentation and visualization
    print(polygon.bounds)
    print(minx, miny)
    apply_segmentation_to_landscape(site_directory, cropped_tif_path, sam_model_path, cleaned_csv_path, polygon, minx, miny)

    if os.path.exists(cropped_tif_path):
        os.remove(cropped_tif_path)
    
    shutil.rmtree(f"images/site{site_number}_crops")
    print(f"Cleaned up temporary files.")

if __name__ == "__main__":
    main()
