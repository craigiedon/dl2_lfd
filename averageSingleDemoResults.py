import numpy as np
# Get appropriate log folders (loop the names in a list)
for demo_type in ["avoid", "patrol", "stable", "slow"]:
    print("{}:".format(demo_type))
    means = []
    stds = []
    for enforce_constraint in [False, True]:
        f_names = ["logs/single-shot-experiments/{}-{}-enforce{}/train_losses.txt".format(demo_type, i, enforce_constraint) for i in range(20)]
        results = np.stack([np.loadtxt(f_name) for f_name in f_names])[:, -1, 0:2]
        # print(results)
        # print(results.shape)
        mean = np.mean(results, 0)
        std = np.std(results, 0)

        means.extend(mean)
        stds.extend(std)
        # print("Imitation \t Constraint")
        print("{:.4f} & {:.4f} & ".format(mean[0], mean[1]), end='')
    # print(means)
    # print(stds)
    print()

# Remember, you are looping through the four tasks, but also the constrained v unconstrained versions of these...
# Look for all train.txt files (perhaps its a regex?)
# Loop (or list comprehension), use numpy loader to load them in
# Take the first two (imitation and constraint loss), do mean on each of them
# Do Confidence interval on each of them (look this up)
# Print out each of the values on a single line