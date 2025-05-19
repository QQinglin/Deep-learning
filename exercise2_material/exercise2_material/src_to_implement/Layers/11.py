import sys
import os

# Define the path to add
path_to_add = r"E:\DL\exercise2_material\exercise2_material\src_to_implement\Optimization\Optimizers.py"

# Normalize the path to use the correct separators for the operating system
path_to_add = os.path.normpath(path_to_add)

# Check if the path exists
if not os.path.exists(path_to_add):
    raise FileNotFoundError(f"The path {path_to_add} does not exist. Please check the directory.")

# Check if the path is already in sys.path to avoid duplicates
if path_to_add not in sys.path:
    sys.path.append(path_to_add)

# Print sys.path to verify the addition
print("Updated sys.path:", sys.path)