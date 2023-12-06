import os



def rename_files(directory_path):
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if "none" in filename.lower():
                old_path = os.path.join(root, filename)
                new_filename = filename.replace("none", "implicit")
                new_path = os.path.join(root, new_filename)
                os.rename(old_path, new_path)
                print(f'Renamed: {old_path} to {new_path}')

# Replace 'your_directory_path' with the actual path of the directory you want to search
directory_path = "data/"
rename_files(directory_path)
