import numpy as np
import pandas as pd
import rasterio
import os
from scipy.spatial import cKDTree

def compute_g_function(coords, support, dataset):
    # Create a KDTree for the coordinates
    tree = cKDTree(coords)
    
    # Initialize an array to hold G(d) values
    g_values = np.zeros(len(support))
    
    # Convert distances to meters
    meters_per_degree = 111300
    
    for i, threshold in enumerate(support):
        count = 0
        for lon1, lat1 in coords:
            # Find distances to the closest neighbor
            distances, _ = tree.query([lon1, lat1], k=2)  # k=2 because the nearest neighbor will be the point itself
            min_distance = distances[1] * meters_per_degree  # Convert from degrees to meters
            
            if min_distance <= threshold:
                count += 1
        
        # Calculate the proportion of points with neighbors within the distance threshold
        g_values[i] = count / len(coords) if len(coords) > 0 else 0
    
    return g_values

def simulate_and_get_g_values(p, sigma, csv_path, tif_path, num_points=None):
    # Load the real data coordinates
    data = pd.read_csv(csv_path)
    real_coords = data[["Longitude", "Latitude"]].values

    if num_points is None:
        num_points = len(data)
    
    # Load TIFF to get bounds, extent, and create a mask
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        img = src.read(1)  # Read the first band
        height, width = img.shape
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]  # Define extent

    # Add a 1000-pixel buffer around the image dimensions
    buffer = 1000
    expanded_height = height + 2 * buffer
    expanded_width = width + 2 * buffer

    # Create a binary mask where white pixels (or near-white) are treated as background
    mask = np.where(img < 255, 1, 0)  # Non-white pixels are foreground (1), white is background (0)
    
    # Expand the mask with zeros (background)
    expanded_mask = np.zeros((expanded_height, expanded_width), dtype=mask.dtype)
    expanded_mask[buffer:buffer+height, buffer:buffer+width] = mask

    # Generate valid coordinates from the expanded mask (foreground pixels)
    valid_coords = np.column_stack(np.where(expanded_mask == 1))

    # Simulation function with p and sigma
    def simulate_points(num_points, domain_size, p=0.5, sigma=50):
        initial_palm = np.random.uniform(0, [domain_size[0], domain_size[1]], size=2)
        palms = [initial_palm]

        while len(palms) < num_points:
            parent_palm = palms[np.random.randint(len(palms))]

            if np.random.rand() < p:
                offspring_palm = parent_palm + np.random.normal(0, sigma, size=2)
                offspring_palm = np.clip(offspring_palm, [0, 0], [domain_size[0], domain_size[1]])
            else:
                offspring_palm = np.random.uniform(0, [domain_size[0], domain_size[1]], size=2)

            palms.append(offspring_palm)

        return np.array(palms)

    # Filter points that fall within the valid mask area
    def filter_points_within_mask(simulated_points, valid_coords):
        filtered_points = []
        for point in simulated_points:
            row, col = int(point[0]), int(point[1])
            if 0 <= row < expanded_mask.shape[0] and 0 <= col < expanded_mask.shape[1]:
                if expanded_mask[row, col] == 1:
                    filtered_points.append(point)
        return np.array(filtered_points)

    # Generate and filter points
    simulated_points = simulate_points(num_points, (expanded_height, expanded_width), p, sigma)
    filtered_points = filter_points_within_mask(simulated_points, valid_coords)

    # Crop the points back to the original image area (remove the buffer)
    def crop_points(filtered_points, buffer, height, width):
        cropped_points = []
        for point in filtered_points:
            if (buffer <= point[0] < buffer + height) and (buffer <= point[1] < buffer + width):
                cropped_points.append(point - [buffer, buffer])
        return np.array(cropped_points)

    cropped_points = crop_points(filtered_points, buffer, height, width)

    # If not enough points, simulate additional points
    while len(cropped_points) < num_points:
        remaining_points_needed = num_points - len(cropped_points)
        print(f"Generating {remaining_points_needed} more points to meet the required number...")
        additional_points = simulate_points(remaining_points_needed, (expanded_height, expanded_width), p, sigma)
        additional_filtered_points = filter_points_within_mask(additional_points, valid_coords)
        cropped_additional_points = crop_points(additional_filtered_points, buffer, height, width)
        if cropped_additional_points.size == 0:
            print("Warning: No additional points matched the mask. Try again...")
            continue
        cropped_points = np.vstack((cropped_points, cropped_additional_points))

    # Convert cropped points to the extent of the TIFF file
    def convert_to_extent(simulated_points, extent, domain_size):
        min_x, max_x = extent[0], extent[1]
        min_y, max_y = extent[2], extent[3]
        
        scale_x = (max_x - min_x) / domain_size[1]
        scale_y = (max_y - min_y) / domain_size[0]
        
        # Scale and shift points to match the extent
        converted_points = np.zeros_like(simulated_points, dtype=np.float64)
        converted_points[:, 0] = simulated_points[:, 1] * scale_x + min_x
        converted_points[:, 1] = (domain_size[0] - simulated_points[:, 0]) * scale_y + min_y
        
        return converted_points

    converted_points = convert_to_extent(cropped_points[:num_points], extent, (height, width))

    # Convert to DataFrame for easier handling
    simulated_df = pd.DataFrame(converted_points, columns=["Longitude", "Latitude"])

    # Calculate Ripley's G functions for real and simulated data
    support = np.arange(1, 41, 1)  # Distance thresholds for G function in meters
    with rasterio.open(tif_path) as src:
        g_real = compute_g_function(real_coords, support, src)
        g_sim = compute_g_function(simulated_df.values, support, src)

    return g_real, g_sim

def integrate_absolute_difference(g_real, g_sim):
    # Integration of the absolute difference
    return np.trapz(np.abs(g_real - g_sim))

def main():
    site_number = input("Enter site number: ")
    model_name = "rt-1"  # Adjust the model name if needed

    # Derive paths based on the site number
    site_directory = f"images/site{site_number}"
    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"No TIFF files found in {site_directory}.")
        return
    tif_file = tif_files[0]
    input_tif_path = os.path.join(site_directory, tif_file)
    base_name = os.path.splitext(tif_file)[0]
    csv_path = os.path.join(site_directory, f'filtered_predictions_{base_name}_{model_name}.csv')

    # Grid search over p and sigma
    # p_values = [0.1, 0.2]
    # sigma_values = [10, 20]
    p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    sigma_values = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
 
 
    results = []

    for p in p_values:
        for sigma in sigma_values:
            print(f"Processing p={p}, sigma={sigma}...")
            integral_values = []
            
            for _ in range(3):  # Perform 3 simulations for each pair
                g_real, g_sim = simulate_and_get_g_values(p, sigma, csv_path, input_tif_path)
                diff_integral = integrate_absolute_difference(g_real, g_sim)
                integral_values.append(diff_integral)
            
            # Average the results from the 5 simulations
            avg_integral = np.mean(integral_values)
            results.append([p, sigma, avg_integral])

    # Save results to a CSV file
    results_df = pd.DataFrame(results, columns=['p', 'sigma', 'integral'])
    results_df.to_csv('spatial-patterns/grid_search_results.csv', index=False)

if __name__ == "__main__":
    main()
