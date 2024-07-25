# Visualization tools for a sample region, of segmentation and detection results

import cv2
import os

def extract_region(image_path, output_path, row_start, row_end, col_start, col_end):
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"Error: File does not exist at {image_path}")
        return
    
    # Read the image
    image = cv2.imread(image_path)
    print(image.shape)
    
    # Check if image is loaded successfully
    if image is None:
        print(f"Error: Unable to load image at {image_path}. Check file path/integrity.")
        return
    
    # Extract the specified region
    region = image[row_start:row_end, col_start:col_end]
    
    # Save the extracted region
    if cv2.imwrite(output_path, region):
        print(f"Extracted region saved to {output_path}")
    else:
        print(f"Error: Failed to save the extracted region to {output_path}")

# Define the paths to the images
image1_path = 'images/site1/JAMACOAQUE1.tif'
image2_path = 'images/site1/SAM_JAMACOAQUE1.png'
image3_path = 'images/site1/SAMm_JAMACOAQUE1.png'

# Define the output paths for the extracted regions
output1_path = 'extracted_raw.jpg'
output2_path = 'extracted_sam.jpg'
output3_path = 'extracted_mobile.jpg'

# Define the region to be extracted (rows 2000-6000, columns 2400-5300)
row_start = 5000
row_end = 8000
col_start = 5000
col_end = 7500

# Extract the region from the first image and save it
extract_region(image1_path, output1_path, int(row_start*1.2987), int(row_end*1.2987), int(col_start*1.2987), int(col_end*1.2987))

# Extract the region from the second image and save it
extract_region(image2_path, output2_path, row_start, row_end, col_start, col_end)

# Extract the region from the second image and save it
extract_region(image3_path, output3_path, row_start, row_end, col_start, col_end)

