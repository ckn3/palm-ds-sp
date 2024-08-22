import os

def count_lines_in_txt_files(directory):
    total_lines = 0
    
    # Walk through all subdirectories and files in the specified directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                
                # Open the file and count the number of lines
                with open(file_path, 'r') as f:
                    total_lines += len(f.readlines())
    
    return total_lines

# Specify the directory containing the .txt files
directory = "/home/kangnicui2/yolo/datasets1"

# Get the total number of lines
total_lines = count_lines_in_txt_files(directory)

print(f"Total number of lines across all .txt files: {total_lines}")
