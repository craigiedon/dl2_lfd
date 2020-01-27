import numpy as np
from os.path import join, isfile
import sys
# Get appropriate log folders (loop the names in a list)

root_folder = sys.argv[1]

enforce_types = ["unconstrained", "train", "adversarial"]
for demo_type in ["avoid", "patrol", "stable", "slow"]:
    print("{}:".format(demo_type))
    for et in enforce_types:
        f_name = join(root_folder, "{}-{}/val_losses.txt".format(demo_type, et))
        # f_name = "logs/generalized-experiments/{}-{}".format(demo_type, et)
        if isfile(f_name):
            results = np.loadtxt(f_name)[-1, 0:2]

            print("{:.4f} & ".format(results[1]), end='')
    print()

# So, for each task
# for each of constrained, unconstrained, and adversarial
# print off each of the constraint losses (and demo losses too?)