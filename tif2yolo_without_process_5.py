
import os
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
import torch


def convert_coordinates_to_pixels(tif_path, csv_1_path, csv_2_path):
    # 打开 TIFF 文件
    with rasterio.open(tif_path) as src:
        transform = src.transform

    # 读取 CSV 文件
    df = pd.read_csv(csv_1_path)

    # 检查 CSV 文件是否包含 'Latitude' 和 'Longitude' 列
    if 'POINT_X' not in df.columns or 'POINT_Y' not in df.columns:
        print("CSV 文件中缺少 'POINT_X' 或 'POINT_Y' 列。")
        return

    # 定义一个函数，将经纬度转换为像素坐标
    def lonlat_to_pixel(row):
        row_idx, col_idx = rasterio.transform.rowcol(transform, row['POINT_X'], row['POINT_Y'])
        return pd.Series({'x_center': col_idx, 'y_center': row_idx})

    # 应用转换函数，并将结果添加到 DataFrame 中
    pixel_coords = df.apply(lonlat_to_pixel, axis=1)
    df = pd.concat([df, pixel_coords], axis=1)

    # 保存更新后的 CSV 文件
    df.to_csv(csv_2_path, index=False)
    print(f"Updated CSV saved to {csv_2_path}")

# %%
def create_bbox(orthomosaic_file, csv_2_path, object_length):
    import os, rasterio
    from PIL import Image
    import pandas as pd

    # prj_name = orthomosaic_file.split(".")[0].split("/")[-1]
    prj_name = os.path.splitext(os.path.basename(orthomosaic_file))[0]
    # Image.MAX_IMAGE_PIXELS = 100000000000
    csv_path = csv_2_path
    img = Image
    img_dict = {}

    # nodata value in your imagery, if one applies for your input dataset
    nodata_value = (255, 255, 255)

    df = pd.read_csv(csv_path)
    dataset = rasterio.open(orthomosaic_file)

    pixelSizeY = dataset.transform[0]
    # this code assumes we are working in UTM or WGS84
    # you can customize to your desired spatial reference
    # or, better yet, incoroporate some kind of interpretation
    # but I have not
    if pixelSizeY < 0.0001:
        pixelSizeY = pixelSizeY * 111300

    bbox_pixel_length = round((object_length / pixelSizeY) / 2)
    print("bbox pixel-level length(width):", bbox_pixel_length)

    df['X_L'] = df['x_center'] - bbox_pixel_length
    df['Y_L'] = df['y_center'] - bbox_pixel_length
    df['X_R'] = df['x_center'] + bbox_pixel_length
    df['Y_R'] = df['y_center'] + bbox_pixel_length

    df.to_csv(csv_path, index=False)
    print(f"Updated CSV saved to {csv_path}")
    return bbox_pixel_length




# %%
def create_big_box(csv_2_path, img_size):
    csv_2_path = csv_2_path
    df = pd.read_csv(csv_2_path)

    # 创建大框
    df['x1'] = df['x_center'] - img_size / 2
    df['y1'] = df['y_center'] - img_size / 2
    df['x2'] = df['x_center'] + img_size / 2
    df['y2'] = df['y_center'] + img_size / 2
    for clumn in ['x1', 'y1', 'x2', 'y2']:
        df[clumn] = df[clumn].clip(lower=0)

    df.to_csv(csv_2_path, index=False)
    print(f"Updated CSV saved to {csv_2_path}")




def screenshot(idx, false_bboxs, csv_2_path):
    import pandas as pd
    import rasterio
    from rasterio.windows import Window
    from PIL import Image
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_2_path)
    col_labels = ['x_center', 'y_center']
    x_center, y_center = df.loc[idx, col_labels]
    # print(x_center, y_center)

    # 读取TIFF图像
    with rasterio.open(tif_path) as dataset:
        x1 = int(x_center - ((bbox_pixel_length + img_size) / 2))
        y1 = int(y_center - ((bbox_pixel_length + img_size) / 2))
        x2 = int(x_center + ((bbox_pixel_length + img_size) / 2))
        y2 = int(y_center + ((bbox_pixel_length + img_size) / 2))

        img_width = dataset.width
        img_height = dataset.height
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, img_width)
        y2 = min(y2, img_height)
        # print(x1, y1, x2, y2)
        # 计算窗口
        window = Window.from_slices((y1, y2), (x1, x2))

        # 读取窗口区域的图像数据
        img_data = dataset.read(window=window)

        # Make white
        for index in false_bboxs:
            # 转换为像素坐标
            col_labels = ['x_center', 'y_center']
            sub_x_center, sub_y_center = df.loc[index, col_labels]
            sub_x_center -= x1
            sub_y_center -= y1
            sub_x1 = int(sub_x_center - bbox_pixel_length / 2)
            sub_y1 = int(sub_y_center - bbox_pixel_length / 2)
            sub_x2 = int(sub_x_center + bbox_pixel_length / 2)
            sub_y2 = int(sub_y_center + bbox_pixel_length / 2)

            # 将区域涂白
            img_data[:, sub_y1:sub_y2, sub_x1:sub_x2] = 255

        # 截取real_x1, real_y1, real_x2, real_y2围成的区域
        real_x1 = int(x_center - img_size / 2)
        real_y1 = int(y_center - img_size / 2)
        real_x2 = int(x_center + img_size / 2)
        real_y2 = int(y_center + img_size / 2)

        # real_window = Window.from_slices((real_y1 - y1, real_y2 - y1), (real_x1 - x1, real_x2 - x1))
        real_img_data = img_data[:, real_y1 - y1:real_y2 - y1, real_x1 - x1:real_x2 - x1]

        print("img shape:", real_img_data.shape)
        # 转换为PIL Image对象
        real_img = Image.fromarray(real_img_data.transpose(1, 2, 0))
        # print("image shape:",img_data.shape)

        # 可视化图像
        # plt.imshow(real_img)
        # plt.title(f"Screenshot at index {idx}")
        # plt.axis('off')  # 隐藏坐标轴
        # plt.show()

        return real_img




def get_tif_filename(tif_path):
    # 获取文件的基本名称（包含扩展名）
    base_name = os.path.basename(tif_path)

    # 分离文件名和扩展名
    file_name, file_extension = os.path.splitext(base_name)

    # 检查文件是否为 TIF 格式
    if file_extension.lower() == '.tif' or file_extension.lower() == '.tiff':
        return file_name
    else:
        raise ValueError(f"The file {tif_path} is not a TIF file.")



# %%

def make_it_white(idx, csv_2_path, tif_path):
    import pandas as pd
    import rasterio
    from rasterio.windows import Window

    # 读取CSV文件
    df = pd.read_csv(csv_2_path)

    # 获取中心坐标
    col_labels = ['x_center', 'y_center']
    sub_x_center, sub_y_center = df.loc[idx, col_labels]

    # 确保中心坐标是数值型
    sub_x_center = float(sub_x_center)
    sub_y_center = float(sub_y_center)

    # 计算矩形区域边界
    sub_x1 = int(sub_x_center - bbox_pixel_length / 2)
    sub_y1 = int(sub_y_center - bbox_pixel_length / 2)
    sub_x2 = int(sub_x_center + bbox_pixel_length / 2)
    sub_y2 = int(sub_y_center + bbox_pixel_length / 2)

    # 打印调试信息
    print(f"sub_x1: {sub_x1}, sub_x2: {sub_x2}, sub_y1: {sub_y1}, sub_y2: {sub_y2}")

    # 读取TIFF图像的指定窗口并进行涂白操作
    with rasterio.open(tif_path, 'r+') as dataset:
        # 定义窗口
        window = Window.from_slices((sub_y1, sub_y2), (sub_x1, sub_x2))

        # 读取子区域
        sub_image = dataset.read(window=window)

        # 将子区域涂白（所有波段）
        sub_image[:, :, :] = 255

        # 将修改后的子区域写回原图
        dataset.write(sub_image, window=window)


# %%
def check_bbox_is_valid(csv_2_path, tif_path, output_dir, bbox_pixel_length=113):
    import pandas as pd

    df = pd.read_csv(csv_2_path)

    tif_name = get_tif_filename(tif_path)
    print(tif_name)
    column_to_check = ['x_center', 'y_center']
    if bbox_pixel_length is None:
        raise ValueError("bbox_pixel_length cannot be None")
    bbox_pixel_length = float(bbox_pixel_length)
    width = int(bbox_pixel_length / 2)

    for index, row in df.iterrows():
        center_x = row[column_to_check[0]]
        center_y = row[column_to_check[1]]
        true_bboxs = []
        false_bboxs = []

        for sub_index, sub_row in df.iterrows():
            if sub_index < index:
                continue

            sub_center_x = sub_row[column_to_check[0]]
            sub_center_y = sub_row[column_to_check[1]]

            if sub_center_x > center_x + 600 and sub_center_y > center_y + 600:
                break
            # print(sub_index)
            if (sub_center_x >= center_x - img_size / 2 + width) and (
                    sub_center_x <= center_x + img_size / 2 - width) and (
                    sub_center_y >= center_y - img_size / 2 + width) and (
                    sub_center_y <= center_y + img_size / 2 - width):
                true_bboxs.append(sub_index)

            elif (sub_center_x >= center_x - img_size / 2 - width) and (
                    sub_center_x <= center_x + img_size / 2 + width) and (
                    sub_center_y >= center_y - img_size / 2 - width) and (
                    sub_center_y <= center_y + img_size / 2 + width):
                false_bboxs.append(sub_index)

        real_img = screenshot(index, false_bboxs, csv_2_path)

        img_name = f"{tif_name}_image_{index}.png"
        label_name = f"{tif_name}_image_{index}.txt"
        img_path = os.path.join(output_dir, img_name)
        label_path = os.path.join(output_dir, label_name)
        print(img_path)
        real_img.save(img_path)

        # for index1 in true_bboxs:
        #     temp = 0
        #     col_labels = ['X_L', 'Y_L', 'X_R', 'Y_R']
        #     x1_min, y1_min, x1_max, y1_max = df.loc[index1, col_labels]
        #     for index2 in false_bboxs:
        #         col_labels = ['X_L', 'Y_L', 'X_R', 'Y_R']
        #         x2_min, y2_min, x2_max, y2_max = df.loc[index2, col_labels]
        #         # 检查是否重叠
        #         if x1_max < x2_min or x2_max < x1_min:
        #             temp += 0
        #         elif y1_max < y2_min or y2_max < y1_min:
        #             temp += 0
        #         else:
        #             temp += 1
        #     if temp > 0:
        #         make_it_white(index1, csv_2_path, tif_path)

        # 读取 CSV 文件
        df = pd.read_csv(csv_2_path)

        x_min = int(center_x - img_size / 2)
        y_min = int(center_y - img_size / 2)
        x_max = int(center_x + img_size / 2)
        y_max = int(center_y + img_size / 2)
        image_width = image_height = x_max - x_min

        for index in true_bboxs:
            col_labels = ['Especie', 'x_center', 'y_center', 'X_L', 'Y_L', 'X_R', 'Y_R']
            classes, sub_center_x, sub_center_y, sub_xl, sub_yl, sub_xr, sub_yr = df.loc[index, col_labels]
            sub_center_x -= x_min
            sub_center_y -= y_min
            sub_xl -= x_min
            sub_yl -= y_min
            sub_xr -= x_min
            sub_yr -= y_min

            yolo_x_center = sub_center_x / image_width
            yolo_y_center = sub_center_y / image_height
            yolo_width = (sub_xr - sub_xl) / image_width
            yolo_height = (sub_yr - sub_yl) / image_height

            # with rasterio.open(tif_path) as dataset:
            #     image = dataset.read()
            #     # 检查中心点像素是否为白色（所有波段均为255）

            if classes == 'Bottlebrush unk.':
                classes = 0
                yolo_label_content = f"{classes} {yolo_x_center} {yolo_y_center} {yolo_width} {yolo_height}\n"
                with open(label_path, 'a') as label_file:
                    label_file.write(yolo_label_content)
            if classes == 'Fan unk.':
                classes = 1
                yolo_label_content = f"{classes} {yolo_x_center} {yolo_y_center} {yolo_width} {yolo_height}\n"
                with open(label_path, 'a') as label_file:
                    label_file.write(yolo_label_content)


            # if np.all(image[:, sub_center_y, sub_center_x] != 255):

    print("Complete!")




# %%
def split_dataset(dataset_path, test_size_of_whole=0.1, val_size_of_train=0.11):
    import os
    import shutil
    from sklearn.model_selection import train_test_split

    # 数据集目录
    dataset_dir = dataset_path
    # classes_file = os.path.join(dataset_dir, 'classes.txt')

    # 获取所有图片文件名（不包括扩展名）
    image_files = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.png')]

    # 划分数据集
    train_val_files, test_files = train_test_split(image_files, test_size=test_size_of_whole, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=val_size_of_train,
                                              random_state=42)  # 0.11 x 0.9 ≈ 0.1

    # 在原始数据集目录内创建train、val、test子文件夹
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dataset_dir, split), exist_ok=True)

    # 定义一个函数来移动图片和标注文件
    def move_files(files, split):
        for filename in files:
            for ext in ['.png', '.txt']:
                src_file = os.path.join(dataset_dir, filename + ext)
                dst_file = os.path.join(dataset_dir, split, filename + ext)
                if os.path.exists(src_file):
                    shutil.move(src_file, dst_file)

    # 移动文件到相应的子文件夹中
    move_files(train_files, 'train')
    move_files(val_files, 'val')
    move_files(test_files, 'test')

    # 复制classes.txt文件到每个子文件夹
    # for split in ['train', 'val', 'test']:
    #     shutil.copy(classes_file, os.path.join(dataset_dir, split))

    import os
    import shutil

    # 设置原始文件夹路径，这里存放着你的图片和标签文件
    original_folder = dataset_path

    for foldername in ['train', 'val', 'test']:
        original_folder_path = os.path.join(original_folder, foldername)
        # print(original_folder_path)
        taget_folder_image = os.path.join(original_folder_path, 'images')
        taget_folder_label = os.path.join(original_folder_path, 'labels')

        os.makedirs(taget_folder_image, exist_ok=True)
        os.makedirs(taget_folder_label, exist_ok=True)

        for filename in os.listdir(original_folder_path):
            # 获取文件的完整路径
            file_path = os.path.join(original_folder_path, filename)

            # 根据文件扩展名，将文件复制到相应的目标文件夹
            if filename.endswith('.png'):
                # 复制图片到images文件夹
                shutil.move(file_path, taget_folder_image)
            elif filename.endswith('.txt'):
                # 复制标签到labels文件夹
                shutil.move(file_path, taget_folder_label)

    print("Split Complete!")


# %%
# Define the YAML data
def make_yaml_file(output_dir, csv_2_path):
    import pandas as pd
    import yaml
    import os

    yaml_path = os.path.join(output_dir, "data.yaml")
    # 读取CSV文件并提取类别列
    df = pd.read_csv(csv_2_path)
    categories = df['Especie'].unique()  # 替换 '类别列名' 为实际列名

    # 创建类别名称列表
    names = {i: category for i, category in enumerate(categories)}

    names = {key: value for key, value in names.items() if value != "Palm unk."}
    print(names)
    # 创建YAML数据结构
    data = {
        'train': '../train/images',
        'val': '../val/images',
        'test': '../test/images',
        'nc': len(names),  # number of classes
        'names': names
    }

    # Write the YAML data to the file
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"Created YAML file: {yaml_path}")



#%%
object_length = 10
img_size = 800
tif_path = "data/JAMACOAQUE2.tif"
csv_1_path = "data/Site_2_data_without_nonpalm_points.csv"
csv_2_path = "data/Site_2_with_pixel_location.csv"
output_dir = "test2"

convert_coordinates_to_pixels(tif_path, csv_1_path, csv_2_path)
bbox_pixel_length = create_bbox(tif_path, csv_2_path, object_length)
create_big_box(csv_2_path, img_size)
check_bbox_is_valid(csv_2_path, tif_path, output_dir, bbox_pixel_length)
split_dataset(output_dir, test_size_of_whole=0.1, val_size_of_train=0.11)
make_yaml_file(output_dir, csv_2_path)