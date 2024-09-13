import numpy as np
import pandas as pd
import rasterio
import os
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def format_func(value, tick_number):
    # Format the ticks to have 4 decimal places
    return f'{value:.4f}'

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
        for lon1, lat1 in random_process_coords:
            min_distance, _ = tree.query([lon1, lat1])
            min_distance *= meters_per_degree
            if min_distance <= threshold:
                count += 1
        f_values[i] = count / len(random_process_coords) if len(random_process_coords) > 0 else 0

    return f_values

def simulate_and_plot_ripley(p, sigma, csv_path, tif_path, output_path, include_basemap=True, num_points=None):
    # Load the real data coordinates
    data = pd.read_csv(csv_path)
    real_coords = data[["Longitude", "Latitude"]].values

    if num_points is None:
        num_points = len(data)

    # Load TIFF to get bounds, extent, and create a mask
    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        img = src.read(1)  # Read only the first band
        height, width = img.shape
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        print('TIFF loaded')
        mask = img < 255

    # Generate valid coordinates from the mask (foreground pixels)
    # valid_coords = np.column_stack(np.where(mask == 1))
    print('Mask generated')

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

    def filter_points_within_mask(points, mask):
        filtered_points = []
        for point in points:
            row, col = int(point[0]), int(point[1])
            if 0 <= row < mask.shape[0] and 0 <= col < mask.shape[1]:
                if mask[row, col] == 1:
                    filtered_points.append(point)
        return np.array(filtered_points)

    # Generate and filter simulated points
    print(f"Start to generate {num_points} simulated points...")
    simulated_points = simulate_points(num_points, (height, width), p, sigma)
    filtered_points = filter_points_within_mask(simulated_points, mask)

    # Ensure the number of points matches real data
    while len(filtered_points) < num_points:
        remaining_points_needed = num_points - len(filtered_points)
        print(f"Generating {remaining_points_needed} more simulated points to meet the required number...")
        additional_points = simulate_points(remaining_points_needed, (height, width), p, sigma)
        additional_filtered_points = filter_points_within_mask(additional_points, mask)
        if additional_filtered_points.size == 0:
            print("Warning: No additional simulated points matched the mask. Try again...")
            continue
        filtered_points = np.vstack((filtered_points, additional_filtered_points))

    # Generate uniform random points in pixel coordinates and ensure they match the number of real data points
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

    uniform_points_pixel = generate_uniform_points(num_points, height, width, mask)

    # Convert filtered points to the extent of the TIFF file
    def convert_to_extent(points, extent, domain_size):
        min_x, max_x = extent[0], extent[1]
        min_y, max_y = extent[2], extent[3]
        
        scale_x = (max_x - min_x) / domain_size[1]
        scale_y = (max_y - min_y) / domain_size[0]
        
        converted_points = np.zeros_like(points, dtype=np.float64)
        converted_points[:, 0] = points[:, 1] * scale_x + min_x
        converted_points[:, 1] = (domain_size[0] - points[:, 0]) * scale_y + min_y
        
        return converted_points

    converted_filtered_points = convert_to_extent(filtered_points[:num_points], extent, (height, width))
    converted_uniform_points = convert_to_extent(uniform_points_pixel[:num_points], extent, (height, width))

    simulated_df = pd.DataFrame(converted_filtered_points, columns=["Longitude", "Latitude"])
    uniform_df = pd.DataFrame(converted_uniform_points, columns=["Longitude", "Latitude"])

    print(f"Number of predicted points: {len(real_coords)}")
    print(f"Number of final simulated points: {len(simulated_df)}")
    print(f"Number of final uniform points: {len(uniform_df)}")

    # Calculate Ripley's G functions for real and simulated data
    g_support = np.arange(1, 41, 1)
    f_support = np.arange(2, 81, 2)
    g_real = compute_g_function(real_coords, g_support)
    g_sim = compute_g_function(simulated_df.values, g_support)

    # Generate random points for F function calculation
    random_process_coords = np.column_stack((
        np.random.uniform(bounds.left, bounds.right, num_points),
        np.random.uniform(bounds.bottom, bounds.top, num_points)
    ))

    f_real = compute_f_function(real_coords, random_process_coords, f_support)
    f_sim = compute_f_function(simulated_df.values, random_process_coords, f_support)

    # Create the subplots
    fig, axs = plt.subplots(1, 5, figsize=(30, 6))

    if include_basemap:
        axs[0].imshow(img, cmap='gray', extent=extent)
        axs[0].scatter(real_coords[:, 0], real_coords[:, 1], color='blue', s=1, label="Predicted Points")
        axs[0].set_title("Predicted Points by PalmDSNet")
        axs[0].set_xlabel("Longitude")
        axs[0].set_ylabel("Latitude")
        axs[0].legend(loc='upper right')

        axs[1].imshow(img, cmap='gray', extent=extent)
        axs[1].scatter(simulated_df["Longitude"], simulated_df["Latitude"], color='red', s=1, label="Simulated Points")
        axs[1].set_title(f"Simulated Points on Basemap with p={p} and sigma={sigma}")
        axs[1].set_xlabel("Longitude")
        axs[1].set_ylabel("Latitude")
        axs[1].legend(loc='upper right')

        axs[2].imshow(img, cmap='gray', extent=extent)
        axs[2].scatter(uniform_df["Longitude"], uniform_df["Latitude"], color='green', s=1, label="Uniform Points")
        axs[2].set_title("Uniform Points on Basemap")
        axs[2].set_xlabel("Longitude")
        axs[2].set_ylabel("Latitude")
        axs[2].legend(loc='upper right')
    else:
        axs[0].scatter(real_coords[:, 0], real_coords[:, 1], color='blue', s=1, label="Predicted Points")
        axs[0].set_title("Predicted Points")
        axs[0].set_xlabel("Longitude")
        axs[0].set_ylabel("Latitude")
        axs[0].legend(loc='upper right')

        axs[1].scatter(simulated_df["Longitude"], simulated_df["Latitude"], color='red', s=1, label="Simulated Points")
        axs[1].set_title(f"Simulated Points with p={p} and sigma={sigma}")
        axs[1].set_xlabel("Longitude")
        axs[1].set_ylabel("Latitude")
        axs[1].legend(loc='upper right')

        axs[2].scatter(uniform_df["Longitude"], uniform_df["Latitude"], color='green', s=1, label="Uniform Points")
        axs[2].set_title("Uniform Points")
        axs[2].set_xlabel("Longitude")
        axs[2].set_ylabel("Latitude")
        axs[2].legend(loc='upper right')

    # Set custom tick format for the first three subplots
    for i in range(3):
        axs[i].xaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        axs[i].yaxis.set_major_formatter(ticker.FuncFormatter(format_func))
        axs[i].xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))

    axs[3].plot(g_support, g_sim, color="k", label="Simulation")
    axs[3].plot(g_support, g_real, color="orangered", label="Observed")
    axs[3].set_xlabel("Distance (meters)")
    axs[3].set_ylabel("Cumulative Portion")
    axs[3].set_xlim(0, max(g_support))
    axs[3].set_ylim(0, 1.1)
    axs[3].set_title(r"Ripley's $G(d)$ function")
    axs[3].legend(loc='upper left')

    axs[4].plot(f_support, f_sim, color="k", label="Simulation")
    axs[4].plot(f_support, f_real, color="orangered", label="Observed")
    axs[4].set_xlabel("Distance (meters)")
    axs[4].set_ylabel("Cumulative Portion")
    axs[4].set_xlim(0, max(f_support))
    axs[4].set_ylim(0, 1.1)
    axs[4].set_title(r"Ripley's $F(d)$ function")
    axs[4].legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

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

    include_basemap = input("Include basemap? (Y/N): ").strip().upper() == "Y"

    p = 0.46
    sigma = 70

    output_file = f'spatial-patterns/p{p}_sigma{sigma}_{base_name}.png'
    simulate_and_plot_ripley(p, sigma, csv_path, input_tif_path, output_file, include_basemap)

if __name__ == "__main__":
    main()
