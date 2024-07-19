import os
import pandas as pd
import rasterio
import math
import matplotlib.pyplot as plt

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

def compare_coordinates(site_numbers, distance_threshold=10):
    all_distances = []  # Collect distances from all sites
    total_detected = 0
    total_labelled = 0

    for site_number in site_numbers:
        site_directory = f"images/site{site_number}"
        tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
        if len(tif_files) != 1:
            raise ValueError(f"There should be exactly one TIFF file in the directory for site {site_number}.")
        tif_name = tif_files[0]
        orthomosaic_file = os.path.join(site_directory, tif_name)

        filtered_csv = os.path.join(site_directory, f'filtered_{tif_name[:-4]}_v10.csv')
        tif_csv = os.path.join(site_directory, f'{tif_name[:-4]}.csv')
        filtered_df = pd.read_csv(filtered_csv)
        tif_df = pd.read_csv(tif_csv)

        with rasterio.open(orthomosaic_file) as dataset:
            closest_records = []
            count_ratio = 0

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
                    all_distances.append(closest_point['Distance'])

            closest_df = pd.DataFrame(closest_records)
            closest_df.to_csv(os.path.join(site_directory, f'closest_points_{tif_name[:-4]}.csv'), index=False)
            site_ratio = count_ratio / len(tif_df) if len(tif_df) > 0 else 0
            print(f"Site {site_number} matched ratio: {site_ratio:.4f}")
            total_detected += count_ratio
            total_labelled += len(tif_df)

    overall_ratio = total_detected / total_labelled if total_labelled > 0 else 0
    print(f"Overall matched ratio across all sites: {overall_ratio:.4f}")

    # Plotting the histogram for all sites
    plt.figure(figsize=(10, 6))
    plt.hist(all_distances, bins=30, color='blue', edgecolor='black')
    plt.title('Histogram of Distances in Meters Across All Sites')
    plt.xlabel('Distance (meters)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('images/combined_distance_histogram.png')
    plt.close()

# Example usage for sites 1 through 5
site_numbers = range(1, 2)
compare_coordinates(site_numbers, 5)
