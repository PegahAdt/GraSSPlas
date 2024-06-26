import os
import shutil

# def delete_files_outside_stat(root_dir):
#     """Delete .txt and .png files that are not in the 'stat' folder within each subfolder."""
#     for root, dirs, files in os.walk(root_dir):
#         # Skip the 'stat' folder itself
#         if os.path.basename(root) == 'stat':
#             continue
#         for file in files:
#             if file.endswith('.txt') or file.endswith('.png'):
#                 file_path = os.path.join(root, file)
#                 # Check if the file is not in a 'stat' folder
#                 if 'stat' not in file_path.split(os.sep)[-2:]:
#                     print(f"Deleting: {file_path}")
#                     os.remove(file_path)

def delete_unique_subfolders(target_dir, reference_dir):
    """Delete subfolders in target_dir that do not exist in reference_dir."""
    # List subfolders in target_dir
    target_subfolders = {name for name in os.listdir(target_dir)
                         if os.path.isdir(os.path.join(target_dir, name))}
    
    # List subfolders in reference_dir
    reference_subfolders = {name for name in os.listdir(reference_dir)
                            if os.path.isdir(os.path.join(reference_dir, name))}
    
    # Find subfolders in target_dir not in reference_dir
    unique_subfolders = target_subfolders - reference_subfolders
    
    # Delete these unique subfolders
    for subfolder in unique_subfolders:
        subfolder_path = os.path.join(target_dir, subfolder)
        print(f"Deleting: {subfolder_path}")
        shutil.rmtree(subfolder_path)

# # replace 'your_root_directory_path' with the path to your root directory
# root_directory_path = '/Users/mrsadeghian/Desktop/MrS/Research/GraSSPlas/Data/Ecoli_AssemblyGraph'
# delete_files_outside_stat(root_directory_path)

# replace 'your_target_directory_path' and 'your_reference_directory_path' with the paths to your target and reference directories
target_directory_path = '/Users/mrsadeghian/Desktop/MrS/Research/GraSSPlas/Data/Ecoli_AssemblyGraph'
reference_directory_path = '/Users/mrsadeghian/Desktop/MrS/Research/GraSSPlas/Data/Ecoli_Features'
delete_unique_subfolders(target_directory_path, reference_directory_path)