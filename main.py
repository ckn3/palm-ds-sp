import os
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import numpy as np
import ultralytics
from ultralytics import YOLO, RTDETR
import cv2
import time
import torch
from collections import deque

ultralytics.checks()

choice = input("Do you want to train or evaluate? (Enter 'train' or 'evaluate'): ")

if choice == 'train':
    model_choice = input("Which model do you want to use? Enter the number of your choice:\n"
                         "1. yolov8l\n"
                         "2. yolov8x\n"
                         "3. yolov9c\n"
                         "4. yolov9e\n"
                         "5. rtdert-l\n"
                         "6. rtdetr-x\n"
                         "7. others\n")
    
    if model_choice == '1':
        model = YOLO('yolov8l.pt')
    elif model_choice == '2':
        model = YOLO('yolov8x.pt')
    elif model_choice == '3':
        model = YOLO('yolov9c.pt')
    elif model_choice == '4':
        model = YOLO('yolov9e.pt')
    elif model_choice == '5':
        model = RTDETR('rtdetr-l.pt')
    elif model_choice == '6':
        model = RTDETR('rtdetr-x.pt')
    elif model_choice == '7':
        model_path = input("Enter the model you want to train with (e.g., 'yolov8m.pt'): ")
        model = YOLO(model_path)
    else:
        print("Invalid model choice.")
        exit()
    
    epochs = int(input("Enter the number of epochs to train: "))
    results = model.train(data='datasets1/data.yaml', epochs=epochs, imgsz=800, single_cls=True, device=[0,1,2,3])
    
elif choice == 'evaluate':
    model_choice = input("Which model do you want to use? Enter the number of your choice:\n"
                         "1. yolov8l\n"
                         "2. yolov8x\n"
                         "3. yolov9c\n"
                         "4. yolov9e\n"
                         "5. rtdetr-l\n"
                         "6. rtdetr-x\n"
                         "7. others\n")
    
    if model_choice == '1':
        model = YOLO('runs/detect/train-v8l/weights/best.pt')
    elif model_choice == '2':
        model = YOLO('weights/epochs300/v8.pt')
    elif model_choice == '3':
        model = YOLO('runs/detect/train-v9c/weights/best.pt')
    elif model_choice == '4':
        model = YOLO('weights/epochs300/v9.pt')
    elif model_choice == '5':
        model = RTDETR('runs/detect/train-rtl/weights/best.pt')
    elif model_choice == '6':
        model = RTDETR('weights/epochs300/rt.pt')
    elif model_choice == '7':
        model_path = input("Enter the model you want to evaluate with (e.g., 'yolov8m.pt'): ")
        model = YOLO(model_path)
    else:
        print("Invalid model choice.")
        exit()
    
    metrics = model.val()
    print("map50-95:", metrics.box.map)
    print("map50:", metrics.box.map50)
    print("map75:", metrics.box.map75)
    print("map50-95 of each category:", metrics.box.maps)

    # Save performance metrics to a CSV file
    csv_path = os.path.join('runs', 'performance_metrics.csv')
    with open(csv_path, 'a') as f:
        f.write(f"Model: {model_choice}\n")
        f.write(f"map50-95: {metrics.box.map}\n")
        f.write(f"map50: {metrics.box.map50}\n")
        f.write(f"map75: {metrics.box.map75}\n")
        f.write(f"map50-95 of each category: {metrics.box.maps}\n")
        f.write("\n")
    print("Performance metrics saved to:", csv_path)
    
else:
    print("Invalid choice.")