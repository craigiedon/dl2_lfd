from glob import glob
from os import rename
import re

joint_name_paths = glob("demos/**/joint_names*", recursive=True)

for j_path in joint_name_paths:
    prefix_pattern = r'joint_names(\d+)'
    replacement_pattern = r'joint_names_\1'
    corrected_path = re.sub(prefix_pattern, replacement_pattern, j_path)
    rename(j_path, corrected_path)

