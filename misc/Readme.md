# Usage Overview

This collection of scripts facilitates the preparation and organization of datasets for YOLO-based object detection tasks. If your data is in YOLO format, you can directly use it for splitting. However, if your data is in JSON format, you will first need to use `convert.py` to transform it into YOLO format. After converting your data, you have two options for splitting it into training, validation, and test sets:

1. **`split.py`**: This script splits the dataset into training, validation, and test sets once.
2. **`split5.py`**: This script performs the splitting process five times, each with a different random seed, to create multiple datasets for robust cross-validation or experimentation.

To count the number of instances in your dataset, use `count.py`, which will tally the number of annotations across all `.txt` files.

**Workflow**:
1. Place your dataset in a folder named `yolo`.
2. Use `convert.py` to convert JSON labels to YOLO format.
3. Run `split.py` or `split5.py` to divide the dataset into train, validation, and test subsets according to your needs.


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

## split.py

The `split.py` script splits a dataset of images and their corresponding `.json` labels into training, validation, and test sets. This script ensures that the images and labels are organized into separate folders, making them ready for training and validation in object detection tasks.

### Key Functionality:
- **Dataset Splitting**: The dataset is divided into training, validation, and test sets, with default ratios of 90% training and 10% validation. The user can modify these ratios as needed.
- **Paired Image and Label Copying**: For each image, its corresponding `.json` label file is also copied to the appropriate folder. If a `.json` file does not exist for an image, it will skip copying the label.
  
### How to Use:
1. Specify the source folder containing the images and `.json` label files.
2. Define the destination folder where the `train`, `val`, and `test` sets will be stored.
3. Optionally adjust the split ratios (default: 90% for training, 10% for validation).

The script will create the following directory structure within the destination folder:
```
datasets1/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```
The train, validation, and test sets will contain images and their corresponding `.json` label files (if available). The script will print a message confirming successful dataset splitting.

## split5.py

The `split5.py` script splits a dataset of images and their corresponding `.json` labels into training, validation, and test sets. The splitting process is repeated **five times** with different random seeds, creating five separate datasets. Each dataset can be used for cross-validation or other experimental setups.

### Key Functionality:
- **Repeated Dataset Splitting**: The script splits the dataset five times, each with a different random shuffle of the images.
- **Separate Folders for Each Split**: Each split creates its own `train`, `val`, and `test` directories under separate folders (e.g., `datasets1`, `datasets2`, etc.).
- **Paired Image and Label Copying**: For each image, the corresponding `.json` label is copied to the appropriate folder. If the `.json` file is missing, only the image is copied.

### How to Use:
1. Specify the source folder containing the images and `.json` label files.
2. The script will automatically create five datasets, named `datasets1`, `datasets2`, etc.
3. Each dataset contains train, validation, and test subdirectories for images and labels.

The script will create the following folder structure:

```
datasets1/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

datasets2/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

...

datasets5/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each folder will contain the train, val, and test splits, with the corresponding images and `.json` label files.
