import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform as unif
from scipy import interpolate
from scipy.interpolate import CubicSpline
from scipy import optimize


def equally_space_curve(squashed_spline, num_steps):
    # squashed_spline = curve_interp(np.linspace(0, 1, num_steps))
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


def simpleCurveWithAvoidPoint(start_range, goal_range, attractor_range, plot=False):
    # fig, ax = plt.subplots()
    start = unif(start_range[0], start_range[1])
    goal = unif(goal_range[0], goal_range[1])

    attractor_point = unif(attractor_range[0], attractor_range[1])
    avoid_point = attractor_point + np.random.rand(2) * 0.05

    distractor_1 = unif([0.1, 0.1], [0.9, 0.9])
    distractor_2 = unif([0.1, 0.1], [0.9, 0.9])

    ts = np.array([0.0, unif(0.4, 0.6), 1.0])

    relevant_keypoints = np.array([start, attractor_point, goal])
    all_features = np.array([start, avoid_point, attractor_point, distractor_1, distractor_2, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(true_xys, 100)

    if plot:
        # noisy_xys =  true_xys + np.random.randn(100, 2) * 0.02
        #ax.scatter(true_xys[:, 0], true_xys[:, 1], alpha=0.4, label='squashed')
        # ax.scatter(noisy_xys[:, 0], noisy_xys[:, 1], alpha=0.25)

        plt.scatter(unif_xys[:, 0], unif_xys[:, 1], label='uniform', alpha=0.5)
        plt.scatter(all_features[0:2, 0], all_features[0:2, 1], marker='x', c='r', s=40*2)
        # ax.scatter(avoid_point[0], avoid_point[1], s=30**2, alpha=0.3, c='r')
        # plt.add_artist(plt.Circle(avoid_point, radius=0.1, alpha=0.2, color='r'))
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()


    return all_features, unif_xys 

def simpleCurveWithTwoPatrols(start_range, goal_range, ar_1, ar_2, plot=False):
    start = unif(start_range[0], start_range[1])
    goal = unif(goal_range[0], goal_range[1])

    attractor_1 = unif(ar_1[0], ar_1[1])
    attractor_2 = unif(ar_2[0], ar_2[1])

    distractor_1 = unif([0.1, 0.1], [0.9, 0.9])
    distractor_2 = unif([0.1, 0.1], [0.9, 0.9])

    ts = np.array([0.0, unif(0.2, 0.3), unif(0.6, 0.8), 1.0])
    relevant_keypoints = np.array([start, attractor_1, attractor_2, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(true_xys, 100)

    patrol_1 = unif_xys[25] + unif([-0.05, 0.05], [-0.1, 0.1])
    patrol_2 = unif_xys[75] + unif([0.05, -0.05], [0.1, -0.1])

    all_features = np.array([start, patrol_1, patrol_2,
                             attractor_1, attractor_2, distractor_1, distractor_2, goal])

    if plot:
        plt.scatter(unif_xys[:, 0], unif_xys[:, 1], label='uniform', alpha=0.5)
        plt.scatter(all_features[0:3, 0], all_features[0:3, 1], marker='x', c='r', s=40*2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()


    return all_features, unif_xys


def movingSinWave(start_range, goal_range, attractor_range, plot=False):
    start = unif(start_range[0], start_range[1])
    goal = unif(goal_range[0], goal_range[1])

    attractor = unif(attractor_range[0], attractor_range[1])
    distractor_1 = unif([0.1, 0.1], [0.9, 0.9])
    distractor_2 = unif([0.1, 0.1], [0.9, 0.9])

    ts = np.array([0.0, unif(0.4, 0.6), 1.0])
    relevant_keypoints = np.array([start, attractor, goal])
    spline = CubicSpline(ts, relevant_keypoints)


    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(true_xys, 100)

    sin_noise = np.sin(2 * np.linspace(0, np.pi * 2, len(unif_xys))) * 0.2
    unif_xys[:, 1] += sin_noise

    unif_xys = equally_space_curve(unif_xys, 100)

    all_features = np.array([start, attractor, distractor_1, distractor_2, goal])

    if plot:
        plt.scatter(unif_xys[:, 0], unif_xys[:, 1], label='uniform', alpha=0.5)
        plt.scatter(all_features[0:2, 0], all_features[0:2, 1], marker='x', c='r', s=40*2)
        plt.scatter(all_features[-1, 0], all_features[-1, 1], marker='x', c='r', s=40*2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(bottom=0, top=1)
        plt.tight_layout()


    return all_features, unif_xys

def unevenSpeeds(start_range, goal_range, attractor_range, plot=False):
    start = unif(start_range[0], start_range[1])
    goal = unif(goal_range[0], goal_range[1])

    attractor = unif(attractor_range[0], attractor_range[1])
    distractor_1 = unif([0.1, 0.1], [0.9, 0.9])
    distractor_2 = unif([0.1, 0.1], [0.9, 0.9])

    ts = np.array([0.0, unif(0.4, 0.6), 1.0])
    relevant_keypoints = np.array([start, attractor, goal])
    spline = CubicSpline(ts, relevant_keypoints)

    true_xys = spline(np.linspace(0, 1, 100))
    unif_xys = equally_space_curve(true_xys, 100)

    uneven_xys = np.concatenate((
        equally_space_curve(unif_xys[0:30], 40),
        equally_space_curve(unif_xys[30:70], 20),
        equally_space_curve(unif_xys[70:], 40)
    ))

    # print("Diffs: ", np.linalg.norm(uneven_xys[:-1] - uneven_xys[1:],  axis=1))
    # unif_xys = equally_space_curve(unif_xys, 100)

    all_features = np.array([start, attractor, distractor_1, distractor_2, goal])

    if plot:
        # plt.scatter(unif_xys[:, 0], unif_xys[:, 1], label='uniform', alpha=0.5)
        plt.scatter(uneven_xys[:, 0], uneven_xys[:, 1], label='uniform', alpha=0.5)
        plt.scatter(all_features[0:2, 0], all_features[0:2, 1], marker='x', c='r', s=40*2)
        plt.scatter(all_features[-1, 0], all_features[-1, 1], marker='x', c='r', s=40*2)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.ylim(bottom=0, top=1)
        plt.tight_layout()


    return all_features, uneven_xys