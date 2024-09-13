# palm-detection-segmentation

## main.py

The `main.py` script provides functionality for training and evaluating object detection models using the Ultralytics YOLO and RTDETR frameworks. This script allows users to select a pre-trained model or provide their own model for training or evaluation. The main features include model selection, training for a specified number of epochs, and evaluation with performance metrics. Results and metrics are saved for further analysis.

### Key Functionalities:
- **Model Selection**: Users can choose from a set of YOLO and RTDETR models or provide their own model path for custom training or evaluation.
- **Training**: Trains the selected model using the specified dataset (`datasets1/data.yaml`) and saves the results, including weights, during the training process. It also supports multi-GPU training.
- **Evaluation**: Evaluates the performance of a trained model, providing key metrics such as mAP (mean Average Precision) across various thresholds (mAP50, mAP75, mAP50-95). The evaluation results are saved in a CSV file for future reference.
  
### How to Use:
1. **Training**: 
   - Run the script and select the "train" option.
   - Choose the model from a list (YOLOv8, YOLOv9, RTDETR) or input the path to a custom model.
   - Input the number of epochs for training.
   - The model will be trained using the dataset provided in `datasets1/data.yaml`, and the results (including weights) will be saved automatically.
  
2. **Evaluation**: 
   - Run the script and select the "evaluate" option.
   - Choose the model to evaluate (pre-trained or custom).
   - The script will calculate and display performance metrics, including mAP values.
   - Metrics are saved to a CSV file under the `runs` directory.

### Example of Installation and Requirements:

To use `main.py`, the following packages need to be installed:

[![numpy - 1.24.2](https://img.shields.io/badge/numpy-1.24.2-blue?logo=python)](https://numpy.org/)  
[![ultralytics - 8.0.117](https://img.shields.io/badge/ultralytics-8.0.117-blue?logo=python)](https://docs.ultralytics.com/)  
[![opencv - 4.7.0](https://img.shields.io/badge/opencv-4.7.0-blue?logo=python)](https://opencv.org/)  
[![torch - 2.0.1](https://img.shields.io/badge/torch-2.0.1-blue?logo=pytorch)](https://pytorch.org/)

- **NumPy**: Provides support for large, multi-dimensional arrays and matrices, along with a large collection of mathematical functions to operate on these arrays.
- **Ultralytics**: A library that includes the YOLO and RTDETR models, useful for object detection.
- **OpenCV**: An open-source computer vision and machine learning software library.
- **Torch**: A deep learning framework that offers efficient tensor computation and automatic differentiation.

You can install these packages using pip:

```bash
pip install numpy ultralytics opencv-python torch
```

Before running `main.py`, ensure that the required environment variable is set and models and data are correctly configured in your directories.
