# palm-detection

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

- `rasterio`: A library for reading and writing geospatial raster data. 
- `shapely`: A library for manipulation and analysis of planar geometric objects.
- `geopandas`: An open source project to make working with geospatial data in Python easier.
- `natsort`: A natural sorting algorithm to sort lists of strings and numbers naturally.

You can install these packages using pip. Run the following command:

```bash
pip install rasterio shapely geopandas natsort
```

Ensure that your input images are placed in the `images/site{x}` directory before running the `tif2yolo` script.
