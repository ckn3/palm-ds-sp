import rasterio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import os

# Define paths
site_number = input("Enter the site number: ")
model_name = input("Enter the model name: ")

site_directory = f"images/site{site_number}"
tif_files = [f for f in os.listdir(site_directory) if f.endswith('.tif')]
if not tif_files:
    print(f"No TIFF files found in {site_directory}.")
    exit()
tif_file = tif_files[0]
input_tif_path = os.path.join(site_directory, tif_file)
base_name = os.path.splitext(tif_file)[0]
print(base_name)
# csv_path = os.path.join(site_directory, f'filtered_{base_name}_v10s.csv')
csv_path = os.path.join(site_directory, f'filtered_{base_name}_{model_name}.csv')

output_image_paths = [os.path.join(site_directory, f'Bbox_{base_name}_{model_name}.png')]
    

# Printing paths to confirm
for path in output_image_paths:
    print(f"Output image path set to: {path}")

with rasterio.open(input_tif_path) as src:
    data = src.read([1, 2, 3])  # Assuming the first three bands are RGB
    transform = src.transform

    dpi = 100
    fig_width = src.width / dpi
    fig_height = src.height / dpi

for output_image_path in output_image_paths:
    fig, ax = plt.subplots(1, figsize=(fig_width, fig_height), dpi=dpi)
    ax.imshow(data.transpose(1, 2, 0), origin='upper')

    df = pd.read_csv(csv_path)

    def coords_to_pixel(lon, lat, transform):
        px, py = ~transform * (lon, lat)
        return int(px), int(py)

    colors = {'Bottlebrush unk.': 'blue', 'Fan unk.': 'red', 'Palm': 'red'}

    for index, row in df.iterrows():
        lon, lat = row['Longitude'], row['Latitude']
        width, height = row['Width'], row['Height']
        class_name = row['Predicted Class']
        x, y = coords_to_pixel(lon, lat, transform)
        marker_size_points = (50) ** 2 * width * height / 25000

        rect = patches.Rectangle((x - width/2, y - height/2), width, height, linewidth=2, edgecolor=colors.get(class_name, 'green'), facecolor='none')
        ax.add_patch(rect)

    handles, labels = plt.gca().get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
    plt.axis('off')
    plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
