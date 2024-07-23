import os
import pandas as pd
import torch
import torchvision
import time

def apply_nms_to_sites(site_numbers, model_name, iou_threshold=0.5):
    results = {}
    for site_number in site_numbers:
        # Define the directory and path for CSV based on site and model name
        site_directory = f"images/site{site_number}"
        tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
        if len(tif_files) != 1:
            raise ValueError(f"There should be exactly one TIFF file in the directory for site {site_number}.")
        tif_name = tif_files[0]

        # Construct the path to the CSV
        csv_path = os.path.join(site_directory, f'predictions_{tif_name[:-4]}_{model_name}.csv')
        output_csv_path = os.path.join(site_directory, f'filtered_{tif_name[:-4]}_{model_name}.csv')

        start_time = time.time()

        # Read the CSV
        df = pd.read_csv(csv_path)

        # Adjust center coordinates by adding X and Y shifts
        df['Adjusted Longitude'] = df['Xc'] + df['X']
        df['Adjusted Latitude'] = df['Yc'] + df['Y']

        # Convert center coordinates to corners (x1, y1, x2, y2) for NMS
        df['x1'] = df['Adjusted Longitude'] - df['Width'] / 2
        df['y1'] = df['Adjusted Latitude'] - df['Height'] / 2
        df['x2'] = df['Adjusted Longitude'] + df['Width'] / 2
        df['y2'] = df['Adjusted Latitude'] + df['Height'] / 2

        # Convert DataFrame columns to tensors for NMS
        boxes = torch.tensor(df[['x1', 'y1', 'x2', 'y2']].values, dtype=torch.float32)
        scores = torch.tensor(df['Conf'].values, dtype=torch.float32)

        # Apply Non-Maximum Suppression
        keep_indices = torchvision.ops.nms(boxes, scores, iou_threshold)

        # Filter the DataFrame to keep only the rows that were not suppressed
        df_filtered = df.iloc[keep_indices.numpy()]

        # Filter out extreme aspect ratios
        aspect_ratio = df_filtered['Height'] / df_filtered['Width']
        df_filtered = df_filtered[(aspect_ratio <= 3) & (aspect_ratio >= 0.33)]

        # Select only the necessary columns to save, including all original columns
        columns_to_keep = ['Longitude', 'Latitude', 'Width', 'Height', 'Predicted Class', 'X', 'Y', 'Xc', 'Yc', 'Conf']
        df_final = df_filtered[columns_to_keep]
        
        # Save the filtered DataFrame to a new CSV file
        df_final.to_csv(output_csv_path, index=False)

        end_time = time.time()

        results[site_number] = df_final
        print(f"Filtered predictions saved to {output_csv_path}")
        print(f"Total cleaning time: {end_time - start_time} seconds")
    
    return results

# Example usage for multiple sites with model 'v10'
site_numbers = [1, 2, 12, 15, 16]
model_name = 'v9'
iou_threshold = 0.5
results = apply_nms_to_sites(site_numbers, model_name, iou_threshold)
