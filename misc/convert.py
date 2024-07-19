import json
import os

def convert_json_to_yolo(source_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.json'):
                json_path = os.path.join(root, file)
                txt_path = json_path.replace('.json', '.txt')

                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)

                with open(txt_path, 'w') as txt_file:
                    for shape in data['shapes']:
                        label = shape['label']
                        points = shape['points']
                        x1, y1 = points[0]
                        x2, y2 = points[1]
                        # Calculate YOLO format values
                        x_center = ((x1 + x2) / 2) / 800
                        y_center = ((y1 + y2) / 2) / 800
                        width = (abs(x2 - x1)) / 800
                        height = (abs(y2 - y1)) / 800
                        # Write to txt file, class "0" for any label
                        txt_file.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

# Usage
for i in range(1, 6):
    source_folder = f'datasets{i}'
    convert_json_to_yolo(source_folder)
