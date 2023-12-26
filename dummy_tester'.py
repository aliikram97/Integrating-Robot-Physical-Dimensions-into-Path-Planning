import os

directory_path = r'C:\Users\Asus\robot dimension integrator\Integrating-Robot-Physical-Dimensions-into-Path-Planning\selected_maps/'

# Get all files in the directory
files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

print(files)