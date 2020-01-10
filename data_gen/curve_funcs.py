import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as unif
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import optimize


def equally_space_curve(curve_interp, num_steps):
    squashed_spline = curve_interp(np.linspace(0, 1, num_steps))
    dists = np.linalg.norm(squashed_spline[:-1] - squashed_spline[1:],  axis=1)
    total_dist = dists.sum()
    dist_thresh = total_dist / (num_steps - 1)

    eq_space_spline = []

    current_point = squashed_spline[0]
    next_i = 1

    while next_i < len(squashed_spline):
        eq_space_spline.append(current_point)

        while next_i < len(squashed_spline) and np.linalg.norm(squashed_spline[next_i] - current_point) < dist_thresh:
            next_i += 1

        if next_i >= len(squashed_spline):
            break

        diff = squashed_spline[next_i] - current_point
        current_point = current_point + (diff / np.linalg.norm(diff)) * dist_thresh
    
    eq_space_spline.append(squashed_spline[-1])
    eq_space_spline = np.array(eq_space_spline)
        

    # print(len(eq_space_spline))
    # print(np.linalg.norm(eq_space_spline[1:] - eq_space_spline[:-1], axis=1))
    return eq_space_spline


def simpleCurveWithAvoidPoint(start_range, goal_range, attractor_point, avoid_point):
    fig, ax = plt.subplots()
    start = unif(start_range[0], start_range[1])
    goal = unif(goal_range[0], goal_range[1])

    ts = np.array([0.0, 0.5, 1.0])

    relevant_keypoints = np.array([start, attractor_point, goal])
    all_features = np.array([start, attractor_point, avoid_point, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(spline, 100)

    # noisy_xys =  true_xys + np.random.randn(100, 2) * 0.02
    #ax.scatter(true_xys[:, 0], true_xys[:, 1], alpha=0.4, label='squashed')
    # ax.scatter(noisy_xys[:, 0], noisy_xys[:, 1], alpha=0.25)

    ax.scatter(unif_xys[:, 0], unif_xys[:, 1], label='uniform')
    ax.scatter(all_features[:, 0], all_features[:, 1], marker='x', c='r')
    # ax.scatter(avoid_point[0], avoid_point[1], s=30**2, alpha=0.3, c='r')
    ax.add_artist(plt.Circle(attractor_point, radius=0.1, alpha=0.3))


    return all_features, unif_xys 

    # np.savetxt("{}/train/start-state-{}.txt".format(demo_dir, i), all_keypoints, fmt='%5f')
    # np.savetxt("{}/train/rollout-{}.txt".format(demo_dir, i), unif_xys, fmt='%5f')