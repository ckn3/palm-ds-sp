import numpy as np
import pandas as pd
import rasterio
import os
from scipy.spatial import cKDTree

def compute_g_function(coords, support):
    tree = cKDTree(coords)
    g_values = np.zeros(len(support))
    meters_per_degree = 111300
    
    for i, threshold in enumerate(support):
        count = 0
        for lon1, lat1 in coords:
            distances, _ = tree.query([lon1, lat1], k=2)
            min_distance = distances[1] * meters_per_degree
            
            if min_distance <= threshold:
                count += 1
        
        g_values[i] = count / len(coords) if len(coords) > 0 else 0
    
    return g_values

def compute_f_function(coords, random_process_coords, support):
    tree = cKDTree(coords)
    f_values = np.zeros(len(support))
    meters_per_degree = 111300
    
    for i, threshold in enumerate(support):
        count = 0
        for x, y in random_process_coords:
            distance, _ = tree.query([x, y], k=1)
            distance_in_meters = distance * meters_per_degree
            
            if distance_in_meters <= threshold:
                count += 1
        
        f_values[i] = count / len(random_process_coords) if len(random_process_coords) > 0 else 0
    
    return f_values

def generate_uniform_points(num_points, height, width, mask):
    uniform_points_pixel = np.column_stack((
        np.random.randint(0, height, num_points),
        np.random.randint(0, width, num_points)
    ))
    
    filtered_points = filter_points_within_mask(uniform_points_pixel, mask)

    while len(filtered_points) < num_points:
        remaining_points_needed = num_points - len(filtered_points)
        print(f"Generating {remaining_points_needed} more uniform points to meet the required number...")
        additional_points_pixel = np.column_stack((
            np.random.randint(0, height, remaining_points_needed),
            np.random.randint(0, width, remaining_points_needed)
        ))
        additional_filtered_points = filter_points_within_mask(additional_points_pixel, mask)
        if additional_filtered_points.size == 0:
            print("Warning: No additional uniform points matched the mask. Try again...")
            continue
        filtered_points = np.vstack((filtered_points, additional_filtered_points))
    
    return filtered_points

def filter_points_within_mask(points_pixel, mask):
    filtered_points = []
    for point in points_pixel:
        row, col = int(point[0]), int(point[1])
        if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
            if mask[row, col] == 1:
                filtered_points.append(point)
    return np.array(filtered_points)

def simulate_points(num_points, domain_size, p=0.5, sigma=50):
    # initial_palm = np.random.uniform(0, [domain_size[0], domain_size[1]], size=2)
    # palms = [initial_palm]
    initial_palms = np.random.uniform(0, [domain_size[0], domain_size[1]], size=(10, 2))
    palms = list(initial_palms)

    while len(palms) < num_points:
        parent_palm = palms[np.random.randint(len(palms))]

        if np.random.rand() < p:
            offspring_palm = parent_palm + np.random.normal(0, sigma, size=2)
            offspring_palm = np.clip(offspring_palm, [0, 0], [domain_size[0], domain_size[1]])
        else:
            offspring_palm = np.random.uniform(0, [domain_size[0], domain_size[1]], size=2)

        palms.append(offspring_palm)

    return np.array(palms)

def crop_points(filtered_points, buffer, height, width):
    cropped_points = []
    for point in filtered_points:
        if (buffer <= point[0] < buffer + height) and (buffer <= point[1] < buffer + width):
            cropped_points.append(point - [buffer, buffer])
    return np.array(cropped_points)

def convert_to_extent(simulated_points, extent, domain_size):
    min_x, max_x = extent[0], extent[1]
    min_y, max_y = extent[2], extent[3]
    
    scale_x = (max_x - min_x) / domain_size[1]
    scale_y = (max_y - min_y) / domain_size[0]
    
    converted_points = np.zeros_like(simulated_points, dtype=np.float64)
    converted_points[:, 0] = simulated_points[:, 1] * scale_x + min_x
    converted_points[:, 1] = (domain_size[0] - simulated_points[:, 0]) * scale_y + min_y
    
    return converted_points

def simulate_and_get_gf_values(p, sigma, real_coords, mask, height, width, extent, buffer, num_points):
    expanded_height = height + 2 * buffer
    expanded_width = width + 2 * buffer

    simulated_points = simulate_points(num_points, (expanded_height, expanded_width), p, sigma)
    filtered_points = filter_points_within_mask(simulated_points, mask)

    cropped_points = crop_points(filtered_points, buffer, height, width)

    while len(cropped_points) < num_points:
        remaining_points_needed = num_points - len(cropped_points)
        print(f"Generating {remaining_points_needed} more points to meet the required number...")
        additional_points = simulate_points(remaining_points_needed, (expanded_height, expanded_width), p, sigma)
        additional_filtered_points = filter_points_within_mask(additional_points, mask)
        cropped_additional_points = crop_points(additional_filtered_points, buffer, height, width)
        if cropped_additional_points.size == 0:
            print("Warning: No additional points matched the mask. Try again...")
            continue
        cropped_points = np.vstack((cropped_points, cropped_additional_points))

    converted_points = convert_to_extent(cropped_points[:num_points], extent, (height, width))
    simulated_df = pd.DataFrame(converted_points, columns=["Longitude", "Latitude"])

    uniform_points_pixel = generate_uniform_points(len(real_coords), height, width, mask)
    uniform_points = convert_to_extent(uniform_points_pixel, extent, (height, width))

    g_support = np.arange(1, 41, 1)
    f_support = np.arange(2, 81, 2)
    
    g_real = compute_g_function(real_coords, g_support)
    g_sim = compute_g_function(simulated_df.values, g_support)

    f_real = compute_f_function(real_coords, uniform_points, f_support)
    f_sim = compute_f_function(simulated_df.values, uniform_points, f_support)

    return g_real, g_sim, f_real, f_sim

def integrate_absolute_difference(g_real, g_sim, f_real, f_sim):
    g_diff_integral = np.trapz(np.abs(g_real - g_sim))
    f_diff_integral = np.trapz(np.abs(f_real - f_sim))
    return g_diff_integral + f_diff_integral

def load_tiff_data(tif_path):
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        img = src.read(1)
        height, width = img.shape
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
    
    buffer = 1000
    expanded_height = height + 2 * buffer
    expanded_width = width + 2 * buffer

    mask = img < 255
    
    expanded_mask = np.zeros((expanded_height, expanded_width), dtype=mask.dtype)
    expanded_mask[buffer:buffer+height, buffer:buffer+width] = mask
    
    return mask, height, width, extent, buffer

def main():
    site_number = input("Enter site number: ")
    model_name = "rt-1"

    site_directory = f"images/site{site_number}"
    tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
    if not tif_files:
        print(f"No TIFF files found in {site_directory}.")
        return
    tif_file = tif_files[0]
    input_tif_path = os.path.join(site_directory, tif_file)
    base_name = os.path.splitext(tif_file)[0]
    csv_path = os.path.join(site_directory, f'filtered_predictions_{base_name}_{model_name}.csv')

    mask, height, width, extent, buffer = load_tiff_data(input_tif_path)
    
    data = pd.read_csv(csv_path)
    real_coords = data[["Longitude", "Latitude"]].values

    # p_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # sigma_values = [10,20,30,40,50,60,70,80,90,100,120,140,160,180,200,250,300,350,400,450,500]
    p_values = [0.35,0.36,0.37,0.38,0.39,0.4,0.41,0.42,0.43,0.44,0.45,0.46,0.47,0.48,0.49,0.5,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59,0.6]
    sigma_values = [30,40,50,60,70,80,90,100,110,120]

    results = []

    for p in p_values:
        for sigma in sigma_values:
            print(f"Processing p={p}, sigma={sigma}...")
            integral_values = []
            
            for _ in range(10):
                g_real, g_sim, f_real, f_sim = simulate_and_get_gf_values(
                    p, sigma, real_coords, mask, height, width, extent, buffer, len(real_coords)
                )
                diff_integral = integrate_absolute_difference(g_real, g_sim, f_real, f_sim)
                integral_values.append(diff_integral)
            
            avg_integral = np.mean(integral_values)
            results.append([p, sigma, avg_integral])
            print(f"Average integral for p={p}, sigma={sigma}: {avg_integral}")
    
    results_df = pd.DataFrame(results, columns=["p", "sigma", "integral"])
    output_csv_path = os.path.join(f"spatial-patterns/results_{base_name}_2.csv")
    results_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")

if __name__ == "__main__":
    main()
