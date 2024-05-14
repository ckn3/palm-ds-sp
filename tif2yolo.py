import os

def remove_non_palm(input_csv):
    import pandas as pd
    original_path = input_csv
    df = pd.read_csv(original_path)
    classes_to_remove = ['Non-palm', 'Palm unk.']
    indices_to_remove = df[df['Especie'].isin(classes_to_remove)].index
    df = df.drop(indices_to_remove)
    print(f"Rows after deletion: {len(df)}")
    df.to_csv(original_path, index=False)

def tif2png(tif_path, png_path, object_length=10, tile_dimension=800):

    # directory containing our orthomosaic
    orthomosaic_path = tif_path

    # destination folder for our tiles
    output_directory = png_path
    # generous estimate of focal object length in meters, to set overlap between tiles
    # note: this should be longer than your longest annotation, or else you'll have
    # messy annotations around your tiling
    object_length = object_length

    # desired tile height and width in pixels
    tile_dimension = tile_dimension

    ### This section defines functions for tiling

    def crop(im, tile_dimension, stride):

        import rasterio
        from rasterio.plot import reshape_as_image
        from rasterio.windows import Window

        img_height, img_width = im.shape
        # print(im.shape)
        count = 0
        for r in range(0, img_height, stride):
            for c in range(0, img_width, stride):
                tile = im.read((1, 2, 3), window=Window(c, r, tile_dimension, tile_dimension))
                tile = reshape_as_image(tile)
                top_pixel = [c, r]
                count += 1
                yield tile, top_pixel

    def tile_ortho(orthomosaic_file, tile_dimension, object_length, output_directory):
        import os, rasterio
        from PIL import Image

        # prj_name = orthomosaic_file.split(".")[0].split("/")[-1]
        prj_name = os.path.splitext(os.path.basename(orthomosaic_file))[0]
        # Image.MAX_IMAGE_PIXELS = 100000000000

        img = Image
        img_dict = {}

        # nodata value in your imagery, if one applies for your input dataset
        nodata_value = (255, 255, 255)

        dataset = rasterio.open(orthomosaic_file)

        pixelSizeY = dataset.transform[0]

        # this code assumes we are working in UTM or WGS84
        # you can customize to your desired spatial reference
        # or, better yet, incoroporate some kind of interpretation
        # but I have not
        if pixelSizeY < 0.0001:
            # we're probably working with degrees, not meters, and 1 deg latitude = 111.3 km
            pixelSizeY = pixelSizeY * 111300

        # set overlap: this should equal 1–2x the pixel-length of our feature of interest
        overlap = round((object_length / pixelSizeY) * 1.1)
        stride = tile_dimension - overlap
        # print(pixelSizeY)

        # use this variable to set output directory
        output_dir = output_directory
        # create the dir if it doesn't already exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # break it up into cropped tiles
        for k, tile_w_point in enumerate(crop(dataset, tile_dimension, stride)):
            empty_data = [(0, 0, 0), nodata_value]
            try:
                img = Image.fromarray(tile_w_point[0])
            except ValueError:
                print("End of set at file " + image_name)
                break
            image_name = prj_name + "---%s.png" % k
            # print(image_name)
            corner1, corner2, corner3, corner4 = img.load()[0, 0], img.load()[0, img.size[1] - 1], img.load()[
                img.size[0] - 1, img.size[1] - 1], img.load()[img.size[0] - 1, 0]
            if corner1 in empty_data and corner2 in empty_data and corner3 in empty_data and corner4 in empty_data:
                # print("empty tile, skipped")
                continue
            img_dict[image_name] = tile_w_point[1]
            path = os.path.join(output_dir, image_name)
            img.save(path)
        print("tiling complete")

        return img_dict, overlap

    def create_metadata_file(orthomosaic_file, img_dict, tile_dimension, overlap, output_directory):

        import rasterio, json

        dataset = rasterio.open(orthomosaic_file)
        full_dict = {"image_name": orthomosaic_file,
                     "image_locations": img_dict,
                     "crs": str(dataset.crs)
                     }
        json_output = output_directory + '/tiling_scheme.json'

        with open(json_output, 'w') as fp:
            json.dump({"orthomosaic_file": orthomosaic_file.split("/")[-1], "spatial_reference": str(dataset.crs),
                       "transform": dataset.transform, "tile_height": tile_dimension, "tile_width": tile_dimension,
                       "tile_overlap": overlap, "tile_pointers": full_dict}, fp)


    img_dict, overlap = tile_ortho(orthomosaic_path, tile_dimension, object_length, output_directory)
    create_metadata_file(orthomosaic_path, img_dict, tile_dimension, overlap, output_directory)

def global_annotations_to_tiles(json_file, output_dir, input_csv, class_cloname='Especie', object_size=10):

    # via_annotations_file = "/media/gregl/Big_Datasets/Grey_Seals/Level_3_1000s/MG2016/MG2016_complete_1st_draft.csv"
    tiling_scheme_file = json_file  # the tiling scheme JSON file generated during the tiling script
    output_dir = output_dir  # folder where any outputs will be dumped
    input_csv = input_csv  # csv of points representing focal objects, here palm trees
    class_colname = class_cloname  # column where the object "class" is located
    object_size = object_size  # estimated size of our focal object, here a palm tree, given in meters

    from pprint import pprint
    from shapely.geometry import mapping
    import geopandas as gpd, rasterio, os
    import pandas as pd

    import pandas as pd
    import shutil

    # this function imports necessary metadata from the tiling scheme file generated during the earlier tiling script
    def import_tiling_scheme(tiling_scheme_file):
        import json
        from affine import Affine

        with open(tiling_scheme_file) as f:
            tiling_scheme = json.load(f)
        gt = tiling_scheme["transform"]
        geotransform = (gt[2], gt[0], gt[1], gt[5], gt[3], gt[4])
        geotransform = Affine.from_gdal(*geotransform)
        tiling_scheme["transform"] = geotransform
        return tiling_scheme

    tiling_scheme = import_tiling_scheme(tiling_scheme_file)

    ### This section converts the points from the CSV file to boxes
    ### not necessary if your annotations are already in boxes

    # reading in the orthomosaic, for spatial reference (CRS)
    # and the point file, as a csv with X and Y coordinate columns for lat/lon
    pts = gpd.read_file(input_csv)  # read in the CSV file to a geodataframe
    gdf = gpd.points_from_xy(pts.POINT_X, pts.POINT_Y)

    # we apply a buffer around each point
    buffer_dist = object_size / 2  # distance from center, in meters
    buffer_dist = buffer_dist / 111300  # converted to lat/lon degrees

    box_list = []
    for pt in gdf:
        box = pt.buffer(buffer_dist).envelope
        box_list.append(box)

    # then we assemble a new geodataframe with these boxes

    d = {'Especie': pts['Especie'], 'geometry': box_list}
    boxes = gpd.GeoDataFrame(d, crs=tiling_scheme['spatial_reference'])

    # we now have a geodataframe of species-labeled boxes extending 'buffer_dist' around each point


    # this function imports necessary metadata from the tiling scheme file generated during the earlier tiling script
    def import_tiling_scheme(tiling_scheme_file):
        import json
        from affine import Affine

        with open(tiling_scheme_file) as f:
            tiling_scheme = json.load(f)
        gt = tiling_scheme["transform"]
        geotransform = (gt[2], gt[0], gt[1], gt[5], gt[3], gt[4])
        geotransform = Affine.from_gdal(*geotransform)
        tiling_scheme["transform"] = geotransform
        return tiling_scheme

    tiling_scheme = import_tiling_scheme(tiling_scheme_file)

    # this function converts box coordinates from global coordinates to orthomosaic coordinates
    def globalboxes_to_orthoboxes(box_list, tiling_scheme):
        entry_list = []
        for box in boxes.iterrows():
            newbox = {}
            coords = mapping(box[1]['geometry'])['coordinates'][0][0:4]
            newbox['class'] = box[1][class_colname]
            newbox['box'] = []
            for point in coords:
                newbox['box'].append(~tiling_scheme["transform"] * point)
            entry_list.append(newbox)
        return entry_list

    orthoboxes = globalboxes_to_orthoboxes(boxes, tiling_scheme)
    pprint(orthoboxes[0:5])

    # this big ugly section just concerns figuring out which tile(s) a box should be shown in
    # it is complicated because boxes can show up in multiple tiles if they straddle the edge
    # or if the sit in the overlap region
    # but we also want to disregard a box if it exists 90% outside the the tile
    # I'm not proud of this function, so if you can do a better job please feel free to!

    def assign_tiles(bbox, tiling_scheme):
        tile_height = tiling_scheme["tile_height"]  # height of each tile
        tile_width = tiling_scheme["tile_width"]  # width of each tile
        tile_overlap = tiling_scheme["tile_overlap"]  # overlap between tiles
        img_data = tiling_scheme["tile_pointers"]  # top-left pixel location for each image, in orthomosaic coordinates
        x_tile_dist = tile_width - tile_overlap  # X-axis stride between tiles
        y_tile_dist = tile_height - tile_overlap  # Y-axis stride between tiles

        # set the tile-pointer corners of the box (leftx dividend, topy dividend)
        # and the in-tile coordinates of the box (leftx remainder, topy remainder
        x_coordinates, y_coordinates = zip(*bbox)
        tilepointer_x, intile_lx = divmod(min(x_coordinates),
                                          x_tile_dist)  # tilepointer tells us the top-left ortho-coordinate of the tile
        tilepointer_y, intile_ty = divmod(min(y_coordinates),
                                          y_tile_dist)  # intile tells the top-left tile-coordinate of the box

        intile_rx = max(x_coordinates) - (tilepointer_x * x_tile_dist)  # rightmost x tile-coordinate of the box
        intile_by = max(y_coordinates) - (tilepointer_y * y_tile_dist)  # bottommost y tile-coordinate of the box

        box_dimensionx = intile_rx - intile_lx  # width of the box in pixels
        box_dimensiony = intile_by - intile_ty  # heigh of the box in pixels

        tile_pointer = "[{x}, {y}]".format(x=int(tilepointer_x * x_tile_dist), y=int(tilepointer_y * y_tile_dist))
        inv_map = {str(v): k for k, v in img_data['image_locations'].items()}
        # tile_pointer sets the location of the tile, which we use to look up the tile name

        # this variable determines how "clipped" an edge box can be before we throw it away
        disregard_threshold = 0.9

        entry = []

        # if the rightmost edge of a box is left of the tile's edge + the "disregard threshold"
        # and the bottommost edge is above the tile's edge and the disregard threshold
        if intile_rx < tile_width + disregard_threshold * box_dimensionx and intile_by < tile_height + disregard_threshold * box_dimensiony:
            # we assign the tile name to this box
            try:
                tile_info = inv_map[tile_pointer]
                entry.append(tile_info)
            except:
                print("first attempted tile does not exist (will try an adjacent tile)")
        # use remainder to determine whether a detection occurs in overlap and needs to be multiple-annotated
        new_tilepointer_x = tilepointer_x
        new_tilepointer_y = tilepointer_y
        # if the rightmost x tile-coordinate of the box is in the left overlap zone, also put it in the left-adjacent box
        if intile_rx < tile_overlap + disregard_threshold * box_dimensionx:
            new_tilepointer_x = tilepointer_x - 1
            # print("left margin")
        # or if the leftmost x tile-coordinate of the box is in the right overlap zone, also put it in the right-adjacent box
        elif intile_lx > tile_width - (tile_overlap + disregard_threshold * box_dimensionx):
            new_tilepointer_x = tilepointer_x + 1
            # print("right margin")
        # if the bottommost y tile-coordinate is in the top overlap zone, also put it in the top-adjacent box
        if intile_by < tile_overlap + disregard_threshold * box_dimensiony:
            new_tilepointer_y = tilepointer_y - 1
            # print("top margin")
        # if the topmost y tile-coordinate is in the bottom overlap zone, also put it in the bottom-adjacent box
        elif intile_ty > tile_height - (tile_overlap + disregard_threshold * box_dimensiony):
            new_tilepointer_y = tilepointer_y + 1
            # print("bottom margin")
        # if the left remainder has changed, change the tile pointer
        if new_tilepointer_x != tilepointer_x:
            tile_pointer = "[{x}, {y}]".format(x=int(new_tilepointer_x * x_tile_dist),
                                               y=int(tilepointer_y * y_tile_dist))
            try:
                tile_info = inv_map[tile_pointer]
                entry.append(tile_info)
            except:
                print("adjacent tile does not exist")
        # if the top remainder has changed, change the tile pointer
        if new_tilepointer_y != tilepointer_y:
            tile_pointer = "[{x}, {y}]".format(x=int(tilepointer_x * x_tile_dist),
                                               y=int(new_tilepointer_y * y_tile_dist))
            try:
                tile_info = inv_map[tile_pointer]
                entry.append(tile_info)
            except:
                print("adjacent tile does not exist")
        # if both have changed, change tile pointer to bishop (diagonal) tile
        if new_tilepointer_x != tilepointer_x and new_tilepointer_y != tilepointer_y:
            # print("double margin")
            tile_pointer = "[{x}, {y}]".format(x=int(new_tilepointer_x * x_tile_dist),
                                               y=int(new_tilepointer_y * y_tile_dist))
            try:
                tile_info = inv_map[tile_pointer]
                entry.append(tile_info)
            except:
                print("bishop tile does not exist")
        return entry

    def SortFunc(e):
        return e['tile_ID']

    # this function converts orthomosaic coordinates to tile coordinates
    def orthoboxes_to_tileboxes(box_list, tiling_scheme):
        tile_entry_list = []
        img_data = tiling_scheme["tile_pointers"]  # top-left pixel location for each image, in orthomosaic coordinates
        for orthobox in orthoboxes:
            tile_ID = assign_tiles(orthobox['box'], tiling_scheme)
            for x in tile_ID:
                x_offset, y_offset = img_data['image_locations'][x]
                tilebox = []
                for point in orthobox['box']:
                    new_x = point[0] - x_offset
                    new_y = point[1] - y_offset
                    if new_x > tiling_scheme["tile_width"]:
                        new_x = tiling_scheme["tile_width"]
                    elif new_x < 0:
                        new_x = 0
                    if new_y > tiling_scheme["tile_height"]:
                        new_y = tiling_scheme["tile_height"]
                    elif new_y < 0:
                        new_y = 0
                    tilebox.append((new_x, new_y))
                # reassemble boxes entries with tile name and coordinates
                try:
                    new_entry = {"tile_ID": x, "box": tilebox, "class": orthobox[
                        'class']}  ####### make sure multiple entries possible per old entry, and take care of class!
                    tile_entry_list.append(new_entry)
                except:
                    print("entry was blank")
        from natsort import natsorted
        tile_entry_list = natsorted(tile_entry_list, key=SortFunc)
        return tile_entry_list

    #
    tileboxes = orthoboxes_to_tileboxes(orthoboxes, tiling_scheme)
    pprint(tileboxes[0:5])

    import os
    import csv

    def tileboxes_to_VIA_file(tileboxes, tiling_scheme_file, output_dir):
        # Ensure the correct path separator is used
        tiling_scheme_file = tiling_scheme_file.replace("\\", "/")
        # Fix the way to get input_dir
        input_dir = os.path.basename(os.path.dirname(tiling_scheme_file))

        via_array = [["filename",
                      "file_size",
                      "file_attributes",
                      "region_count",
                      "region_id",
                      "region_shape_attributes",
                      "region_attributes"]]
        img_data = tiling_scheme["tile_pointers"]
        remnant_tiles = list(img_data["image_locations"].keys())

        filename = ""
        for tilebox in tileboxes:
            if filename != tilebox["tile_ID"]:
                filename = tilebox["tile_ID"]
                if filename in remnant_tiles:
                    remnant_tiles.remove(filename)
                count = 0
            else:
                count += 1
            x_coordinates, y_coordinates = zip(*tilebox["box"])
            x1, y1 = int(min(x_coordinates)), int(min(y_coordinates))
            x2, y2 = int(max(x_coordinates)), int(max(y_coordinates))

            region_shape_attributes = {"name": "rect", "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1}
            region_attributes = {class_colname: tilebox["class"]}  

            via_array.append([filename, "", "{}", "", count, str(region_shape_attributes).replace("'", '"'), str(region_attributes).replace("'", '"')])

        for tile in remnant_tiles:
            via_array.append([tile, '', '{}', '', '', '', ''])

        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Build the output file path
        output_file_path = os.path.join(output_dir, f"{input_dir}_VIA_annotations_{tiling_scheme['tile_width']}_{tiling_scheme['tile_overlap']}.csv")

        # Write the new VIA file
        with open(output_file_path, 'w', newline='') as fp:
            writer = csv.writer(fp)
            writer.writerows(via_array)

        print(f"Annotations rewritten to {output_file_path}")
        return output_file_path
        # Note: Make sure that the `tiling_scheme`, `tileboxes`, `tiling_scheme_file`, and `output_dir` variables are correctly set before calling this function.

    output_path = tileboxes_to_VIA_file(tileboxes, tiling_scheme_file, output_dir)
    return output_path

def csv2yolo(file_path, output_dir, tile_dimenson):
    import pandas as pd
    import json
    import os
    import pandas as pd
    import numpy as np

    # Read the CSV file
    file_path = file_path
    df = pd.read_csv(file_path)

    # Create a temporary directory to save the generated YOLO format files
    output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)


    # For rows that are not strings, provide a default empty dictionary string "{}"
    df['region_shape_attributes'] = df['region_shape_attributes'].apply(lambda x: x if isinstance(x, str) else "{}")
    df['region_attributes'] = df['region_attributes'].apply(lambda x: x if isinstance(x, str) else "{}")

    # Attempt to parse the dictionary data in string format using JSON
    df['region_shape_attributes'] = df['region_shape_attributes'].apply(json.loads)
    df['region_attributes'] = df['region_attributes'].apply(json.loads)

    # Extract the x, y, width, height, and Especie of the bounding boxes
    df['x'] = df['region_shape_attributes'].apply(lambda attr: attr.get('x'))
    df['y'] = df['region_shape_attributes'].apply(lambda attr: attr.get('y'))
    df['width'] = df['region_shape_attributes'].apply(lambda attr: attr.get('width'))
    df['height'] = df['region_shape_attributes'].apply(lambda attr: attr.get('height'))
    df['Especie'] = df['region_attributes'].apply(lambda attr: attr.get('Especie', 'unknown'))  # Provide a default species as 'unknown'

    # Display the modified DataFrame structure to ensure correct data parsing
    df[['filename', 'x', 'y', 'width', 'height', 'Especie']].head()

    # Image size is 800x800 pixels
    img_width, img_height = tile_dimenson, tile_dimenson

    # Calculate the center coordinates and convert them to proportions of the image size
    df['center_x'] = (df['x'] + df['width'] / 2) / img_width
    df['center_y'] = (df['y'] + df['height'] / 2) / img_height
    df['norm_width'] = df['width'] / img_width
    df['norm_height'] = df['height'] / img_height

    # Convert Especie to integer IDs
    # Create a mapping from Especie to ID
    unique_species = df['Especie'].unique()
    species_to_id = {species: i for i, species in enumerate(unique_species)}

    # Apply mapping to convert Especie to ID
    df['species_id'] = df['Especie'].map(species_to_id)

    # Select columns to write in YOLO format files
    yolo_format_df = df[['filename', 'species_id', 'center_x', 'center_y', 'norm_width', 'norm_height']]

    yolo_format_df.head(), species_to_id

    # Create a corresponding .txt file in YOLO format for each image
    for filename, group in yolo_format_df.groupby('filename'):
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w') as f:
            for _, row in group.iterrows():
                if pd.isna(row['center_x']):
                    continue
                else:
                    f.write(
                        f"{row['species_id']} {row['center_x']} {row['center_y']} {row['norm_width']} {row['norm_height']}\n")

    # Provide the path of the output directory for download
    output_dir

    # 创建 classes.txt 文件，列出所有类别的名称
    classes_txt_path = os.path.join(output_dir, 'classes.txt')
    with open(classes_txt_path, 'w') as f:
        for species in species_to_id.keys():
            if species != 'unknown':
                f.write(f"{species}\n")

    # 提供classes.txt文件的路径以便下载
    classes_txt_path

def combine_files(source_folder):
    import os
    import shutil

    # 源文件夹路径
    source_folder = source_folder

    # 获取所有子文件夹
    subfolders = [os.path.join(source_folder, f) for f in os.listdir(source_folder) if
                  os.path.isdir(os.path.join(source_folder, f))]
    subfolders.sort()

    # 确保有至少一个子文件夹存在
    if len(subfolders) < 1:
        print("no subfolders found")
    else:
        # 第一个子文件夹
        first_subfolder = subfolders[0]

        # 遍历所有子文件夹，移动文件到第一个子文件夹
        for folder in subfolders:
            if folder != first_subfolder:
                for file in os.listdir(folder):
                    file_path = os.path.join(folder, file)
                    destination_path = os.path.join(first_subfolder, file)

                    # 如果目标文件已存在，创建新文件名以避免覆盖
                    if os.path.exists(destination_path):
                        base, extension = os.path.splitext(destination_path)
                        i = 1
                        new_destination_path = f"{base}_{i}{extension}"
                        while os.path.exists(new_destination_path):
                            i += 1
                            new_destination_path = f"{base}_{i}{extension}"
                        destination_path = new_destination_path

                    shutil.move(file_path, destination_path)

        # 删除除第一个子文件夹外的所有子文件夹
        for folder in subfolders:
            if folder != first_subfolder:
                shutil.rmtree(folder)

        print("removed to first subfolders")
        print(first_subfolder)
        return first_subfolder

def optimized_box(source_folder):
    import os
    from PIL import Image
    import fnmatch

    # Source folder path
    source_folder = source_folder

    # Set aspect ratio threshold

    # Iterate over all txt files in the source folder
    for filename in os.listdir(source_folder):
        if filename.endswith(".txt"):
            txt_filepath = os.path.join(source_folder, filename)
            # Get the corresponding image file path
            if fnmatch.fnmatch(filename, 'classes*.txt'):
                continue
            else:
                image_filepath = os.path.join(source_folder, filename.replace('.txt', '.png'))

            # Read the txt file and check if it is empty
            with open(txt_filepath, 'r') as file:
                lines = file.readlines()

            # Skip processing if the file is empty
            if not lines:
                continue

            # Get the image size
            with Image.open(image_filepath) as img:
                image_width, image_height = img.size

            # Create a new content list containing only the lines that were not deleted
            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue  # Ignore lines with incorrect format

                # Convert normalized width and height to pixels
                bbox_width = float(parts[3]) * image_width
                bbox_height = float(parts[4]) * image_height

                # Calculate the aspect ratio
                area = bbox_height * bbox_width
                aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height)

                # Only add the bounding box if it has a suitable aspect ratio and area
                if 10000 <= area and aspect_ratio <= 2:
                    new_lines.append(line)

            # Write the lines back to the file if there is data after filtering, otherwise leave the file empty
            if new_lines:
                with open(txt_filepath, 'w') as file:
                    file.writelines(new_lines)
            else:
                # Clear the file content
                with open(txt_filepath, 'w') as file:
                    file.truncate()

    print("optimized bounding box complete!")

def combine_classes(source_folder):
    import os

    # 设置文件夹路径
    source_folder = source_folder

    # 文件名模式匹配
    pattern = 'classes*.txt'

    # 使用集合来存储唯一的行
    unique_lines = set()

    # 遍历文件夹中的所有文件
    for filename in os.listdir(source_folder):
        if filename.startswith('classes') and filename.endswith('.txt'):
            filepath = os.path.join(source_folder, filename)

            # 读取文件并添加到集合中，自动去重
            with open(filepath, 'r', encoding='utf-8') as file:
                unique_lines.update(line.strip() for line in file if line.strip())

    # 将结果写回 classes.txt
    output_filepath = os.path.join(source_folder, 'classes.txt')
    with open(output_filepath, 'w', encoding='utf-8') as file:
        for line in sorted(unique_lines):
            file.write(line + '\n')

    print("all classes moved to classes.txt.")

def move_to_parent_folder(current_foldr):
    import os
    import shutil

    # 指定当前工作的文件夹路径（你想从中移动文件的那个文件夹）
    current_folder = current_foldr

    # 获取上层文件夹路径
    parent_folder = os.path.abspath(os.path.join(current_folder, os.pardir))

    # 获取当前文件夹中的所有文件和子文件夹的名字
    files_and_folders = os.listdir(current_folder)

    # 遍历所有文件和子文件夹
    for item in files_and_folders:
        # 构建源文件/文件夹的完整路径
        source_path = os.path.join(current_folder, item)

        # 构建目标路径
        destination_path = os.path.join(parent_folder, item)

        # 移动文件/文件夹
        shutil.move(source_path, destination_path)
        print(f"Moved {item} to {parent_folder}")

    print("All files and folders have been moved to the parent directory.")

    # 检查文件夹是否存在
    if os.path.exists(current_folder):
        # 删除文件夹及其所有内容
        shutil.rmtree(current_folder)


def split_dataset(dataset_path, test_size_of_whole=0.1, val_size_of_train=0.11):

    import os
    import shutil
    from sklearn.model_selection import train_test_split

    # 数据集目录
    dataset_dir = dataset_path
    classes_file = os.path.join(dataset_dir, 'classes.txt')

    # 获取所有图片文件名（不包括扩展名）
    image_files = [f.split('.')[0] for f in os.listdir(dataset_dir) if f.endswith('.png')]

    # 划分数据集
    train_val_files, test_files = train_test_split(image_files, test_size=test_size_of_whole, random_state=42)
    train_files, val_files = train_test_split(train_val_files, test_size=val_size_of_train, random_state=42)  # 0.11 x 0.9 ≈ 0.1

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
    for split in ['train', 'val', 'test']:
        shutil.copy(classes_file, os.path.join(dataset_dir, split))

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

    print("Complete!")


def tif2yolo(tif_path, output_path, csv_path, tile_dimenson=800):
    tif_path = tif_path
    output_path = output_path
    csv_path = csv_path
    tile_dimenson = tile_dimenson

    remove_non_palm(csv_path)
    # tif to png&json
    #object_length, tile_dimension默认值为->object_length=10, tile_dimension=800
    tif2png(tif_path, output_path,object_length=10, tile_dimension=tile_dimenson)

    json_path = os.path.join(output_path, "tiling_scheme.json")
    output_dir = os.path.dirname(tif_path)
    #class_cloname, object_size 默认值为->class_cloname='Especie', object_size=10
    new_csv_path = global_annotations_to_tiles(json_path, output_dir, csv_path, class_cloname='Especie', object_size=10)

    csv2yolo(new_csv_path, output_path, tile_dimenson)

main_directory = "images"
output_directory = "datasets"

for folder_name in os.listdir(main_directory):
    folder_path = os.path.join(main_directory, folder_name)
    print(f"Processing folder: {folder_path}")
    
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".tif"):
            tif_path = os.path.join(folder_path, file_name)
            base_name = os.path.splitext(file_name)[0]  # Get the base name without the extension
            csv_file_name = f"{base_name}.csv"  # Expected CSV file name
            csv_path = os.path.join(folder_path, csv_file_name)
            
            if os.path.exists(csv_path):
                print(f"Found TIF: {tif_path}")
                print(f"Matching CSV: {csv_path}")
                
                output_path = os.path.join(output_directory, folder_name)
                os.makedirs(output_path, exist_ok=True)
                
                # Process the TIF and CSV pair
                tif2yolo(tif_path, output_path, csv_path, tile_dimenson=800)
            else:
                print(f"No matching CSV file found for {tif_path}")

source_folder = combine_files(output_directory)
print(source_folder)
optimized_box(source_folder)
combine_classes(source_folder)
move_to_parent_folder(source_folder)
split_dataset(output_directory,  test_size_of_whole=0.1, val_size_of_train=0.11)

def clean_background_pairs(directory):
    labels_dir = os.path.join(directory, 'labels')
    images_dir = os.path.join(directory, 'images')
    
    # Check and remove any empty label files and their corresponding images
    for label_file in os.listdir(labels_dir):
        label_path = os.path.join(labels_dir, label_file)
        
        # Check if the label file is empty
        if os.path.getsize(label_path) == 0:
            # Remove the empty label file
            os.remove(label_path)
            print(f"Removed empty label file: {label_path}")
            
            # Build the path to the corresponding image file
            # Assumes image file has the same basename with a '.png' extension
            image_file = label_file.replace('.txt', '.png')
            image_path = os.path.join(images_dir, image_file)
            
            # Remove the corresponding image file if it exists
            if os.path.exists(image_path):
                os.remove(image_path)
                print(f"Removed corresponding image file: {image_path}")

# List of dataset directories to clean
dataset_dirs = ['datasets/train', 'datasets/val', 'datasets/test']

# Loop through each directory and clean it
for dataset_dir in dataset_dirs:
    clean_background_pairs(dataset_dir)
    print(f"Cleaned background pairs in {dataset_dir}")