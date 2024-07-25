import os
import pandas as pd
import rasterio
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_degree_bounds(lon, lat, meters, dataset):
    pixelSizeX, pixelSizeY = dataset.transform[0], abs(dataset.transform[4])
    if pixelSizeX < 0.0001:
        meters_per_degree = 111300
        degreesX = meters / meters_per_degree
        degreesY = meters / (meters_per_degree * math.cos(math.radians(lat)))
    else:
        degreesX = meters * pixelSizeX
        degreesY = meters * pixelSizeY
    return lon - degreesX, lon + degreesX, lat - degreesY, lat + degreesY

def compare_coordinates(site_numbers, model_name, distance_threshold=10):
    plt.figure(figsize=(7, 4))  # Adjusted for a two-column paper
    site_colors = sns.color_palette("hsv", len(site_numbers))
    
    # Dictionary to map file basenames to desired legend labels
    legend_labels = {
        'FCAT6': 'FCAT 1',
        'FCAT10': 'FCAT 2',
        'FCAT11': 'FCAT 3',
        'JAMACOAQUE1': 'Jama-Coaque 1',
        'JAMACOAQUE2': 'Jama-Coaque 2'
    }

    site_labels = []

    for site_number, color in zip(site_numbers, site_colors):
        site_directory = f"images/site{site_number}"
        tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
        if len(tif_files) != 1:
            raise ValueError(f"There should be exactly one TIFF file in the directory for site {site_number}.")
        tif_name = tif_files[0]
        orthomosaic_file = os.path.join(site_directory, tif_name)

        filtered_csv = os.path.join(site_directory, f'filtered_{tif_name[:-4]}_{model_name}.csv')
        tif_csv = os.path.join(site_directory, f'{tif_name[:-4]}.csv')
        filtered_df = pd.read_csv(filtered_csv)
        tif_df = pd.read_csv(tif_csv)

        with rasterio.open(orthomosaic_file) as dataset:
            closest_records = []
            count_ratio = 0

            distances = []
            for index, tif_row in tif_df.iterrows():
                lon1, lat1 = tif_row['POINT_X'], tif_row['POINT_Y']
                lon_min, lon_max, lat_min, lat_max = get_degree_bounds(lon1, lat1, distance_threshold, dataset)

                close_points = filtered_df[(filtered_df['Longitude'] >= lon_min) & (filtered_df['Longitude'] <= lon_max) &
                                           (filtered_df['Latitude'] >= lat_min) & (filtered_df['Latitude'] <= lat_max)].copy()
                if not close_points.empty:
                    count_ratio += 1
                    close_points['Distance'] = ((close_points['Longitude'] - lon1)**2 + (close_points['Latitude'] - lat1)**2)**0.5
                    meters_per_degree = 111300
                    close_points['Distance'] *= meters_per_degree

                    closest_point = close_points.loc[close_points['Distance'].idxmin()]
                    closest_records.append({
                        'Detected_Longitude': lon1,
                        'Detected_Latitude': lat1,
                        'Human_Longitude': closest_point['Longitude'],
                        'Human_Latitude': closest_point['Latitude'],
                        'Distance': closest_point['Distance']
                    })
                    distances.append(closest_point['Distance'])

            if distances:
                sns.kdeplot(distances, bw_adjust=0.5, color=color, label=legend_labels.get(tif_name[:-4], tif_name[:-4]), clip=(0, None))
                site_labels.append(legend_labels.get(tif_name[:-4], tif_name[:-4]))

            closest_df = pd.DataFrame(closest_records)
            closest_df.to_csv(os.path.join(site_directory, f'closest_points_{tif_name[:-4]}.csv'), index=False)
            site_ratio = count_ratio / len(tif_df) if len(tif_df) > 0 else 0
            print(f"Site {site_number} matched ratio: {site_ratio:.4f}")

    # plt.title('Distribution of Distance Shifts Across Sites')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Density')
    plt.legend(title='Site Name', loc='upper right', labels=site_labels, fontsize='small')
    # plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/distribution_of_distances.png')
    plt.close()

# Example usage for sites 1 through 5
site_numbers = [1, 2, 12, 15, 16] 
compare_coordinates(site_numbers, 'rt', 5)
