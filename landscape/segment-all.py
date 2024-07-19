import os
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch
import time

from ultralytics import SAM

def apply_segmentation_to_landscape(site_number):
    site_directory = f"images/site{site_number}"
    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"No TIFF files found in {site_directory}.")
        return
    tif_file = tif_files[0]
    input_tif_path = os.path.join(site_directory, tif_file)
    base_name = os.path.splitext(tif_file)[0]
    csv_path = os.path.join(site_directory, f'combined_predictions.csv')

    with rasterio.open(input_tif_path) as src:
        data = src.read([1, 2, 3])
        transform = src.transform
        mask_global = np.zeros((src.height, src.width), dtype=np.uint8)

    df = pd.read_csv(csv_path)
    df['y_center'], df['x_center'] = zip(*df.apply(lambda row: rasterio.transform.rowcol(transform, row['Longitude'], row['Latitude']), axis=1))
    df['x1'] = df['x_center'] - df['Width'] / 2
    df['y1'] = df['y_center'] - df['Height'] / 2
    df['x2'] = df['x_center'] + df['Width'] / 2
    df['y2'] = df['y_center'] + df['Height'] / 2
    df['class_id'] = df['Predicted Class'].map({'Fan unk.': 1, 'Bottlebrush unk.': 2})

    model = SAM('mobile_sam.pt')

    for _, row in df.iterrows():
        x, y, width, height = int(row['x_center']), int(row['y_center']), int(row['Width']), int(row['Height'])
        x1, y1 = max(x - 400, 0), max(y - 400, 0)
        x2, y2 = min(x1 + 800, src.width), min(y1 + 800, src.height)
        x1, y1 = x2 - 800, y2 - 800

        cropped_image = data[:, y1:y2, x1:x2]
        cropped_image = np.moveaxis(cropped_image, 0, -1)
        image_pil = Image.fromarray(np.uint8(cropped_image * 255))

        results = model(image_pil, bboxes=[[400 - width // 2, 400 - height // 2, width // 2 + 400, height // 2 + 400]])

        for r in results:
            masks = r.masks.data
            mask_combined = torch.any(masks, dim=0).int() * row['class_id']
            mask_combined = mask_combined.cpu().numpy()
            mask_global[y1:y2, x1:x2] = np.maximum(mask_global[y1:y2, x1:x2], mask_combined)

            # Save intermediate result for each processed patch
            # mask_rgba = np.zeros((800, 800, 4))  # Creating RGBA mask for visualization
            # mask_rgba[..., 0] = (mask_combined == 1)  # Red channel for class 1
            # mask_rgba[..., 2] = (mask_combined == 2)  # Blue channel for class 2
            # mask_rgba[..., 3] = (mask_combined > 0) * 0.5  # Transparency for non-zero mask values
            # overlay = (1 - mask_rgba[..., 3, np.newaxis]) * (cropped_image / 255) + mask_rgba[..., :3]
            # overlay = np.clip(overlay, 0, 1)
            # print(np.max(overlay))
            # plt.imsave(os.path.join('intermediate', f'patch_{base_name}_{x}_{y}.png'), overlay)

    output_image_path = os.path.join(site_directory, f'Annotated_{base_name}.png')
    colors = ['black', 'blue', 'red']  # Custom colormap: background, class 1, class 2
    cmap = ListedColormap(colors)

    plt.figure(figsize=(data.shape[2] / 100, data.shape[1] / 100), dpi=100)
    plt.imshow(np.moveaxis(data, 0, -1))
    plt.imshow(mask_global, cmap=cmap, alpha=0.5)
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Segmentation overlay saved to {output_image_path}")

site_number = input("Enter the site number: ")

start_time = time.time()
apply_segmentation_to_landscape(site_number)
end_time = time.time()
print(f"Time taken: {end_time - start_time:.2f} seconds.")