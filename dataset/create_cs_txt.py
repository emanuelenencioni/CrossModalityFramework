import os

def scan_folders_and_create_txt(folder1, folder2, output_file1, output_file2):
    """
    Scan two folders and create separate txt files with file paths from each folder.
    
    Args:
        folder1 (str): Path to first folder
        folder2 (str): Path to second folder
        output_file1 (str): Path to output txt file for folder1
        output_file2 (str): Path to output txt file for folder2
    """
    files1 = []
    files2 = []
    
    # Scan first folder
    for root, dirs, files in os.walk(folder1):
        for file in files:
            file_path = os.path.join(root, file)
            files1.append(file_path)
    
    # Scan second folder
    for root, dirs, files in os.walk(folder2):
        for file in files:
            file_path = os.path.join(root, file)
            files2.append(file_path)
    
    # Write file paths from folder1 to first output file
    with open(output_file1, 'w') as f:
        for file_path in sorted(files1):
            f.write(file_path + '\n')
    
    # Write file paths from folder2 to second output file
    with open(output_file2, 'w') as f:
        for file_path in sorted(files2):
            f.write(file_path + '\n')
    
    print(f"Created {output_file1} with {len(files1)} file paths")
    print(f"Created {output_file2} with {len(files2)} file paths")

if __name__ == "__main__":
    # Specify your folder paths here
    dataset_folder = os.path.dirname(os.path.abspath(__file__))
    folder1_path = os.path.join(dataset_folder.split("dataset")[0],"data/cityscapes/leftImg8bit/train")
    folder2_path = os.path.join(dataset_folder.split("dataset")[0], "data/cityscapes/leftImg8bit/val")
    output_txt_file1 = os.path.join(dataset_folder, "cs_train.txt")
    output_txt_file2 = os.path.join(dataset_folder, "cs_val.txt")

    scan_folders_and_create_txt(folder1_path, folder2_path, output_txt_file1, output_txt_file2)