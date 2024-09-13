## convert.py

The `convert.py` script is used to convert JSON label files into YOLO format labels. It processes JSON files found in a specified directory and generates corresponding `.txt` files, where bounding box information is reformatted to YOLO format for training object detection models.

### Key Functionality:
- **JSON to YOLO Conversion**: The script walks through the `source_folder`, detects `.json` files, extracts bounding box data, and converts the coordinates into YOLO format.
  - It calculates the bounding box center (`x_center`, `y_center`), width, and height normalized to the image size (800x800).
  - The labels are written to a `.txt` file, where each entry includes the class index (always set to "0" in this script), `x_center`, `y_center`, `width`, and `height`.

### How to Use:
1. Ensure your JSON files are stored in folders like `datasets1`, `datasets2`, etc.
2. Run the script, and for each folder (e.g., `datasets1`, `datasets2`), it will create YOLO-compatible `.txt` files in the same directory as the corresponding `.json` files.

## count.py

The `count.py` script is designed to count the total number of lines across all `.txt` files in a specified directory. This can be useful for tasks like checking the number of annotations in YOLO format or verifying data consistency.

### Key Functionality:
- **Line Counting in `.txt` Files**: The script recursively walks through the specified directory and its subdirectories, identifying all `.txt` files and summing up the number of lines in each file.

### How to Use:
1. Specify the target directory containing the `.txt` files (can include subdirectories).
2. The script will scan all the `.txt` files and output the total number of lines found across them.

## tif2yolo.py

The `tif2yolo` function crops the images in the `images/site{x}` folder into 800x800 patches and creates the corresponding `.txt` files for these images using the corresponding `.csv` file, which are formatted for YOLO training. All processed images and their corresponding `.txt` files are saved to the `datasets` folder. Note that the TIFF images under `images/site{x}` have the same basename as the CSV file to ensure alignment. The CSV file's columns `Especie`, `POINT_X`, and `POINT_Y` are used for creating and labeling the bounding boxes, with each bounding box having the same size of 10x10 meters.

The dataset is split into three subsets:
- `train`: 80%
- `val`: 10%
- `test`: 10%

Before you can use the `tif2yolo` functionality, you need to install the following packages:

[![rasterio - 1.3.10](https://img.shields.io/badge/rasterio-1.3.10-blue?logo=python)](https://rasterio.readthedocs.io/en/stable/)
[![shapely - 2.0.4](https://img.shields.io/badge/shapely-2.0.4-blue?logo=python)](https://shapely.readthedocs.io/en/stable/manual.html)
[![geopandas - 0.14.4](https://img.shields.io/badge/geopandas-0.14.4-blue?logo=python)](https://geopandas.org/en/stable/)
[![natsort - 8.4.0](https://img.shields.io/badge/natsort-8.4.0-blue?logo=python)](https://pypi.org/project/natsort/)
[![scikit-learn - 1.4.2](https://img.shields.io/badge/scikit--learn-1.4.2-blue?logo=python)](https://scikit-learn.org/stable/)

- `rasterio`: A library for reading and writing geospatial raster data.
- `shapely`: A library for manipulation and analysis of planar geometric objects.
- `geopandas`: An open source project to make working with geospatial data in Python easier.
- `natsort`: A natural sorting algorithm to sort lists of strings and numbers naturally.
- `scikit-learn`: A machine learning library for Python.

You can install these packages using pip. Run the following command:

```bash
pip install rasterio shapely geopandas natsort scikit-learn
```

Ensure that your input images are placed in the `images/site{x}` directory before running the `tif2yolo` script.

