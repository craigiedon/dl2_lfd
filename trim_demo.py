import os
import sys
from os.path import join
from glob import glob
import re


def trim_demo(demos_folder, demo_num, frames_to_trim):
    demo_path = os.listdir(demos_folder)[demo_num]
    file_template = join(demos_folder, demo_path, "joint_position*")

    num_regex = r'(\d+)\.txt'
    sorted_joint_paths = sorted(glob(file_template))
    extracted_ts = [re.search(num_regex, x).group(1) for x in sorted_joint_paths]
    trim_ts = extracted_ts[0:frames_to_trim]

    print(trim_ts)
    for trim_t in trim_ts:
        files_to_delete = glob(join(demos_folder, demo_path, "*{}*".format(trim_t)))
        for d in files_to_delete:
            if "joint_names" not in d:
                os.remove(d)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        "Usage: python trim_demo.py <demos_folder> <demo_num> <frames_to_trim>"
        sys.exit(0)
    trim_demo(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

    

# Load all the joint position things in here
# Sort them
# Extract the numbers at the end using a regular expression.
# Print out the numbers
# Grep for all files in that folder with similar number
# Print them out
# Delete those files