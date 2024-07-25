import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

def calculate_distances(df, meters_per_degree):
    """Calculate distances to top 1 and average of top 5 nearest neighbors in meters."""
    coordinates = df[['Longitude', 'Latitude']].values
    tree = cKDTree(coordinates)
    distances, _ = tree.query(coordinates, k=6)  # k=6 because it includes the point itself

    # Convert Euclidean distance in degrees to meters
    top1_distances = distances[:, 1] * meters_per_degree  # First column is distance to itself
    top5_distances = np.mean(distances[:, 1:6], axis=1) * meters_per_degree  # Average of columns 1-5

    return top1_distances, top5_distances

def plot_distributions(site_numbers, model_name):
    plt.figure(figsize=(7, 5))  # Adjusted size for two-column paper, one column wide
    site_colors = sns.color_palette("hsv", len(site_numbers))

    # Dictionary to map file basenames to desired legend labels
    legend_labels = {
        1: 'FCAT 1', 2: 'FCAT 2', 12: 'FCAT 3',
        15: 'Jama-Coaque 1', 16: 'Jama-Coaque 2'
    }

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 5), sharex=True)

    max_distance = 0  # To find the maximum distance for consistent x-axis range

    for site_number, color in zip(site_numbers, site_colors):
        site_directory = f"images/site{site_number}"
        csv_files = [f for f in os.listdir(site_directory) if f.startswith(f'filtered_') and f.endswith(f'_{model_name}.csv')]

        for csv_file in csv_files:
            csv_path = os.path.join(site_directory, csv_file)
            df = pd.read_csv(csv_path)
            # Compute distances and adjust from degrees to meters
            meters_per_degree = 111300  # Average conversion near the equator
            top1_distances, top5_distances = calculate_distances(df, meters_per_degree)

            # Track the maximum distance for consistent x-axis scaling
            max_distance = max(max_distance, max(top1_distances.max(), top5_distances.max()))

            # Plotting top 1 and top 5 distances
            sns.kdeplot(top1_distances, bw_adjust=0.5, color=color, label=legend_labels[site_number], ax=axes[0])
            sns.kdeplot(top5_distances, bw_adjust=0.5, color=color, label=legend_labels[site_number], ax=axes[1])

    # Set consistent x-axis limits based on maximum distance found
    axes[0].set_xlim(0, 50)
    axes[1].set_xlim(0, 50)

    # Configure plot aesthetics
    axes[0].set_title('Top 1 Nearest Neighbor Distance Distribution')
    axes[1].set_title('Average Top 5 Nearest Neighbor Distance Distribution')
    axes[1].set_xlabel('Distance (meters)')
    axes[0].set_ylabel('Density')
    axes[1].set_ylabel('Density')

    # Set legends for both subfigures
    axes[0].legend(title='Site Name')
    axes[1].legend(title='Site Name')

    plt.tight_layout()
    plt.savefig('images/neighbor_distance_distribution_vertical.png')
    plt.close()

# Example usage for sites 1 through 5
site_numbers = [15, 16, 1, 2, 12]
model_name = 'rt'
plot_distributions(site_numbers, model_name)
