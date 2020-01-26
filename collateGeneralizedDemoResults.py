import numpy as np
# Get appropriate log folders (loop the names in a list)

enforce_types = ["unconstrained", "train", "adversarial"]
for demo_type in ["avoid", "patrol", "stable", "slow"]:
    print("{}:".format(demo_type))
    for et in enforce_types:
        f_name = "logs/generalized-experiments/{}-{}".format(demo_type, et)
        results = np.loadtxt(f_name)[-1, 0:2]

        print("{:.4f} & {:.4f} & ".format(results[0], results[1]), end='')
    print()

# So, for each task
# for each of constrained, unconstrained, and adversarial
# print off each of the constraint losses (and demo losses too?)