import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import sys
from math import ceil
from model import load_model, ImageOnlyMDN, ZhangNet, PosePlusStateNet, ImageOnlyNet
from load_data import load_demos, load_rgbd_demos, nn_input_to_imshow, show_torched_im, load_pose_state_demos, image_demo_paths, get_pose_hist, quat_pose_to_rpy, ImageRGBDPoseHist, DeviceDataLoader, ImagePoseFuturePose, PoseAndGoal
import numpy as np
import torch
from torch.utils.data import DataLoader
from helper_funcs.utils import zip_chunks, load_json, load_json_lines

# from torchvision.transforms import Compose
from helper_funcs.transforms import get_trans, get_grayscale_trans

"""
def chart_train_validation_error(train_results_path, validation_results_path):
    training_df = pd.read_csv(train_results_path, sep=" ", header=None, names=["error"])
    print(training_df)
    validation_df = pd.read_csv(validation_results_path, sep=" ", header=None, names=["error"])
    plt.plot(training_df.error, label="Train")
    plt.plot(validation_df.error, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.show()
"""

def plot_csv(csv_path, save_path=None, show_fig=False, col_subset=None):
    df = pd.read_csv(csv_path, sep=",")
    # print(training_df)
    # df.plot(subplots=True)

    if col_subset == None:
        display_cols = df.columns
    else:
        display_cols = col_subset

    for col in display_cols:
        plt.plot(df[col], label=col)

    # plt.plot(training_df.error, label="Results")
    # # plt.plot(validation_df.error, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

    if show_fig:
        plt.show()

    plt.close()

def plot_csv_train_val(csv_path, save_path=None, show_fig=False):
    df = pd.read_csv(csv_path, sep=",")
    # print(training_df)
    # df.plot(subplots=True)

    train_cols = [c for c in df.columns if c.startswith("T")]
    val_cols = [c for c in df.columns if c.startswith("V")]
    for i, (t_col, v_col) in enumerate(zip(train_cols, val_cols)):
        n_rows = ceil(len(train_cols) / 3)
        n_cols = np.min([len(train_cols), 3])
        plt.subplot(n_rows, n_cols, i + 1)
        plt.plot(df[t_col], label=t_col)
        plt.plot(df[v_col], label=v_col)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.ylim(top=0.1)
        plt.legend()

    if save_path is not None:
        plt.savefig(save_path)

    if show_fig:
        plt.show()

    plt.close()

def plot_vae_metrics(csv_path, save_path=None, show_fig=False):
    df = pd.read_csv(csv_path, sep=",")


    subplots = [["T-Full", "V-Full"], ["T-MSE", "V-MSE"], ["T-VAE", "V-VAE"], ["T-Recon", "V-Recon"], ["T-KL", "V-KL"]]
    for i, cols in enumerate(subplots):
        plt.subplot(3,2,i + 1)
        for col in cols:
            plt.plot(df[col], label=col)
        plt.xlabel("Epoch")
        plt.ylabel("Losses")
        plt.legend()


    if save_path is not None:
        plt.savefig(save_path)
    
    if show_fig:
        plt.show()
    
    plt.close()

def chart_error_batches(results_path, group_every=1):
    training_df = pd.read_csv(results_path, sep=" ", header=None, names=["error"])
    training_df = training_df.groupby(training_df.index // group_every).mean()
    print(training_df)
    # validation_df = pd.read_csv(val_path, sep=" ", header=None, names=["error"])
    plt.plot(training_df.error, label="Results")
    # plt.plot(validation_df.error, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.show()


def chart_train_validation_error(train_path, val_path):
    # Load in the states from both as json lists
    training_stats = load_json_lines(train_path)
    val_stats = load_json_lines(val_path)

    plt.plot(training_stats["training_loss"], label="Train")
    plt.plot(val_stats["training_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.show()


def animate_spatial_features(model_path, demos_folder, demo_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_names = np.genfromtxt("config/arm_joint_names.txt", np.str)

    # TODO: These need to come out as config i reckon...
    im_h, im_w = (240, 240)
    crop_t, crop_b, crop_l, crop_r = (150, 475, 50, 800)

    model = load_model(model_path, device, im_h, im_w, joint_names)
    im_trans = Compose([Crop(crop_t, crop_b, crop_l, crop_r), Resize(im_h, im_w)])

    demo_ds, demo_loader = load_demos(demos_folder, "kinect2_qhd_image_color_rect_*.jpg", 32, joint_names, im_trans, False, device, from_demo=demo_num, to_demo=demo_num + 1)

    model.eval()

    # Setup subplots and axes
    fig = plt.figure(figsize=(15,5))
    img_ax = fig.add_subplot(1,1,1)

    resizer = Resize(model.cout_h, model.cout_w)
    resized_im = resizer(nn_input_to_imshow(demo_ds[0][0][0]))
    im_canvas = img_ax.imshow(resized_im)

    feature_points = []
    display_ims = []

    with torch.no_grad():
        # For each image, pass it through, then plot it (except its not plotting it, it will be animating it...)
        for (ins, _) in demo_loader:
            img_ins, _ = ins
            model(*ins)
            fp_batch = model.aux_outputs["feature_points"]
            fp_batch = zip_chunks(fp_batch, 2, dim=1)

            # Append the current feature points to list
            feature_points.extend(fp_batch.tolist())
            # Append the current image to a list
            # TODO: Resize to the conv_out size, or just show the conv_outs!
            display_batch = [nn_input_to_imshow(x) for x in img_ins]
            display_batch = [resizer(x) for x in display_batch]
            display_ims.extend(display_batch)
            

    def anim_step(i):
        del img_ax.patches[:]
        ps = []
        for (fx, fy) in feature_points[i]:
            ps.append(img_ax.add_patch(patches.Circle((fx, fy), radius=1, color="red")))
        im_canvas.set_data(display_ims[i])
        return [im_canvas] + ps

    ani = animation.FuncAnimation(fig, anim_step, interval=200, frames=len(display_ims), blit=True, repeat=True)
    plt.show()
    return ani
            

def chart_all_poses_3d(demo_path):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    demo_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=5)

    # Load all of the poses into a list, reuse some load-data functions here
    demo_trajectories = [get_pose_hist(d, "l_wrist_roll_link") for d in demo_paths]
    demo_trajectories = [np.array([quat_pose_to_rpy(p) for p in dt]).transpose(1,0) for dt in demo_trajectories]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("t")

    for d in demo_trajectories:
        time = np.linspace(0, 1, len(d[2]))
        ax.scatter(d[0], d[1], time)

    plt.show()

    # pose_dim_names = ["x", "y", "z", "r-x", "r-y", "r-z"]
    #     plt.plot(d[i], alpha=0.15, color="green")

def chart_all_poses(demo_path):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    demo_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0)

    # Load all of the poses into a list, reuse some load-data functions here
    demo_trajectories = [get_pose_hist(d, "l_wrist_roll_link") for d in demo_paths]
    demo_trajectories = [np.array([quat_pose_to_rpy(p) for p in dt]).transpose(1,0) for dt in demo_trajectories]

    pose_dim_names = ["x", "y", "z", "r-x", "r-y", "r-z"]
    for i in range(6):
        plt.subplot(2,3, i + 1)
        plt.xlabel("time")
        plt.ylabel((pose_dim_names[i]))
        for d in demo_trajectories:
            plt.plot(d[i], alpha=0.15, color="green")


    plt.show()
        
    # chart in similar manner as before. Shape should be demos X demo_length X 6

def chart_current_v_next(demo_path):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    demo_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=1)

    # Load all of the poses into a list, reuse some load-data functions here
    demo_trajectories = [get_pose_hist(d, "l_wrist_roll_link") for d in demo_paths]
    demo_trajectories = [np.array([quat_pose_to_rpy(p) for p in dt]).transpose(1,0) for dt in demo_trajectories]

    pose_dim_names = ["x", "y", "z", "r-x", "r-y", "r-z"]
    for i in range(6):
        plt.subplot(2,3, i + 1)
        plt.xlabel("current {}".format(pose_dim_names[i]))
        plt.ylabel("next {}".format(pose_dim_names[i]))
        for d in demo_trajectories:
            plt.scatter(d[i][0:-5], d[i][5:], alpha=0.05,s=10, color="blue")


    plt.show()

def chart_pred_goal_pose(model_path, demo_path, demo_num):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=demo_num, to_demo=demo_num + 1)
    train_set = PoseAndGoal(train_paths, "l_wrist_roll_link", "r_wrist_roll_link", skip_count=10)
    demo_loader = DeviceDataLoader(DataLoader(train_set, exp_config["batch_size"], shuffle=False), torch.device("cuda"))

    model = PosePlusStateNet(100)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.to(torch.device("cuda"))
    model.eval()

    next_pred_all, next_targets_all, current_pose_all = [], [], []

    with torch.no_grad():
        for ins in demo_loader:
            current_pose, goal_pose, target_pose = ins 
            next_pred = model(current_pose, goal_pose)
            next_pred_all.append(next_pred)
            next_targets_all.append(target_pose)
            current_pose_all.append(current_pose)
    
    next_pred_all = torch.cat(next_pred_all)
    next_targets_all = torch.cat(next_targets_all)
    current_pose_all = torch.cat(current_pose_all)

    # Shapes: N X 7

    next_pred_all = next_pred_all.permute(1,0).cpu().numpy()
    next_targets_all = next_targets_all.permute(1,0).cpu().numpy()
    current_pose_all = current_pose_all.permute(1,0).cpu().numpy()


    n_pose_dims, n_samples = next_pred_all.shape[0], next_pred_all.shape[1]
    print(n_pose_dims)
    dim_names = ["p-x", "p-y", "p-z", "roll", "pitch", "yaw"]
    for dim_id in range(n_pose_dims):
        plt.subplot(ceil(n_pose_dims / 3.0), 3, dim_id + 1)
        plt.plot(next_pred_all[dim_id], label="Predicted Pose")
        plt.plot(current_pose_all[dim_id], label="Current Pose")
        # plt.plot(current_pred_all[dim_id], label="Aux Prediction")
        plt.plot(next_targets_all[dim_id], label="Target Pose")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel(dim_names[dim_id])

    plt.show()


def chart_pred_pose(model_path, demo_path, demo_num):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=demo_num, to_demo=demo_num + 1)
    train_set = ImagePoseFuturePose(train_paths, "l_wrist_roll_link", get_trans(im_params, distorted=True), 10)
    demo_loader = DeviceDataLoader(DataLoader(train_set, exp_config["batch_size"], shuffle=False), torch.device("cuda"))

    model = ImageOnlyNet(im_params["resize_height"], im_params["resize_width"], 6)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.to(torch.device("cuda"))

    model.eval()
    next_pred_all = []
    # current_pred_all = []
    current_targets_all = []
    next_targets_all = []

    with torch.no_grad():
        for ins in demo_loader:
            rgb_ins, current_ins, target_ins = ins 
            next_pred = model(rgb_ins)
            next_pred_all.append(next_pred)
            # current_pred_all.append(current_pred)
            current_targets_all.append(current_ins)
            next_targets_all.append(target_ins)
    
    next_pred_all = torch.cat(next_pred_all)
    # current_pred_all = torch.cat(current_pred_all)
    current_targets_all = torch.cat(current_targets_all)
    next_targets_all = torch.cat(next_targets_all)

    # Shapes: N X 7

    next_pred_all = next_pred_all.permute(1,0).cpu().numpy()
    # current_pred_all = current_pred_all.permute(1,0).cpu().numpy()
    current_targets_all = current_targets_all.permute(1,0).cpu().numpy()
    next_targets_all = next_targets_all.permute(1,0).cpu().numpy()


    n_pose_dims, n_samples = next_pred_all.shape[0], next_pred_all.shape[1]
    print(n_pose_dims)
    dim_names = ["p-x", "p-y", "p-z", "roll", "pitch", "yaw"]
    for dim_id in range(n_pose_dims):
        plt.subplot(ceil(n_pose_dims / 3.0), 3, dim_id + 1)
        plt.plot(next_pred_all[dim_id], label="Predicted Pose")
        # plt.plot(current_pred_all[dim_id], label="Aux Prediction")
        plt.plot(current_targets_all[dim_id], label="Current Pose")
        plt.plot(next_targets_all[dim_id], label="Target Pose")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel(dim_names[dim_id])

    plt.show()

    

def chart_demo_joint_trajectories(demo_path, demo_num):
    exp_config = load_json("config/experiment_config.json")
    # TODO: This is in multiple places, so I think it needs to be config
    im_params = exp_config["image_config"]

    im_trans = Compose([Crop(im_params["crop_top"], im_params["crop_left"],
                             im_params["crop_height"], im_params["crop_width"]),
                        Resize(im_params["resize_height"], im_params["resize_width"])])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    demo_set, demo_loader = load_demos(
        demo_path,
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        False,
        device,
        from_demo=demo_num,
        to_demo=demo_num + 1
    )

    joint_poses = []
    next_poses = []
    with torch.no_grad():
        for ins, targets in demo_loader:
            _, in_pos = ins
            joint_pos = [x.cpu().numpy() for x in in_pos]
            next_pos = [x.cpu().numpy() for x in targets]

            joint_poses.extend(joint_pos)
            next_poses.extend(next_pos)

    joint_poses = np.array(joint_poses).transpose()
    next_poses = np.array(next_poses).transpose()

    fig = plt.figure()

    for i, _ in enumerate(joint_poses):
        c_ax = fig.add_subplot(ceil(len(joint_poses) / 3.0), 3, i + 1)
        c_ax.plot(joint_poses[i], label="Joint Poses")
        c_ax.plot(next_poses[i], label="Next Poses")
        c_ax.legend()
        c_ax.title.set_text(exp_config["nn_joint_names"][i])
        c_ax.set_xlabel("t")
        c_ax.set_ylabel("Normed Angle")
    
    plt.show()

def chart_mdn_means_image(model_path, demo_path, demo_num):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=demo_num, to_demo=demo_num + 1)
    train_set = ImageRGBDPoseHist(train_paths, "l_wrist_roll_link", get_trans(im_params, distorted=True), get_grayscale_trans(im_params))
    demo_loader = DeviceDataLoader(DataLoader(train_set, exp_config["batch_size"], shuffle=False), torch.device("cuda"))

    model = ImageOnlyMDN(im_params["resize_height"], im_params["resize_width"], 6, 5)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.to(torch.device("cuda"))
    model.eval()
    mu_all = []
    std_all = []
    pis_all = []
    targets_all = []

    with torch.no_grad():
        for ins in demo_loader:
            rgb_in, depth_in, pose_hist, targets = ins
            pis, stds, mus = model(rgb_in)
            print("mu {} std {} pis {}".format(mus.shape, stds.shape, pis.shape))
            mu_all.append(mus)
            std_all.append(stds)
            pis_all.append(pis)
            targets_all.append(targets)
    
    mu_all = torch.cat(mu_all)
    std_all = torch.cat(std_all)
    pis_all = torch.cat(pis_all)
    targets_all = torch.cat(targets_all)

    # Im expecting: mu: N x 5 X 7, std: N x 5, pis: N * 5, targets_all: N * 7
    # print("Shapes: mu: {}, std: {}, pis {}, targets{}".format(mu_all.shape, std_all.shape, pis_all.shape, targets_all.shape))

    # I want : mu: 7 X 5 X N, std: 5 X N, pis: 5 X N, targets_all: 7 * N
    mu_all = mu_all.permute(2,1,0).cpu().numpy()
    std_all = std_all.permute(2,1,0).cpu().numpy()
    pis_all = pis_all.permute(1,0).cpu().numpy()
    targets_all = targets_all.permute(1,0).cpu().numpy()

    print("Shapes: mu: {}, std: {}, pis {}, targets{}".format(mu_all.shape, std_all.shape, pis_all.shape, targets_all.shape))


    fig = plt.figure()

    print("STD example", std_all[0][0])

    n_joints, n_components, n_samples = mu_all.shape[0], mu_all.shape[1], mu_all.shape[2]
    print("joints {} components {} samples {}".format(n_joints, n_components, n_samples))
    for joint_id in range(n_joints):
        plt.subplot(ceil(n_joints / 3.0), 4, joint_id + 1)
        for pi_id in range(n_components):
            plt.plot(mu_all[joint_id, pi_id], label="pi-{}".format(pi_id))
            plt.fill_between(range(n_samples), mu_all[joint_id, pi_id] - std_all[joint_id, pi_id], mu_all[joint_id, pi_id] + std_all[joint_id, pi_id], alpha=0.1)


        weighted_mus = mu_all[joint_id] * pis_all
        averaged_mus = weighted_mus.sum(0)
        print("muall: {}, pis_all: {} Weighted: {}, Averaged: {}".format(mu_all.shape, pis_all.shape, weighted_mus.shape, averaged_mus.shape))
        plt.plot(averaged_mus, label="Averaged")
        plt.plot(targets_all[joint_id], label="Targets")

        plt.legend()
        plt.title(["x", "y", "z", "r-x", "r-y", "r-z"][joint_id])
        plt.xlabel("t")
        plt.ylabel("Normed Radians")

    plt.subplot(ceil(n_joints / 3.0), 4, n_joints + 1)
    for pi_id in range(n_components):
        plt.plot(pis_all[pi_id], label="pi_{}".format(pi_id))
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Component Weight")
    plt.title("Pis")

    plt.show()
def chart_mdn_means(model_path, demo_path, demo_num):
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    _, demo_loader = load_pose_state_demos(
        image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=demo_num, to_demo=demo_num + 1),
        exp_config["batch_size"],
        "l_wrist_roll_link",
        "r_wrist_roll_link",
        False,
        torch.device("cuda"))

    model = PosePlusStateNet(100, 2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
    model.to(torch.device("cuda"))
    model.eval()
    mu_all = []
    std_all = []
    pis_all = []
    targets_all = []

    with torch.no_grad():
        for ins in demo_loader:
            current_pose, goal_pose, targets = ins
            pis, stds, mus = model(current_pose, goal_pose)
            print("mu {} std {} pis {}".format(mus.shape, stds.shape, pis.shape))
            mu_all.append(mus)
            std_all.append(stds)
            pis_all.append(pis)
            targets_all.append(targets)
    
    mu_all = torch.cat(mu_all)
    std_all = torch.cat(std_all)
    pis_all = torch.cat(pis_all)
    targets_all = torch.cat(targets_all)

    # Im expecting: mu: N x 5 X 7, std: N x 5, pis: N * 5, targets_all: N * 7
    # print("Shapes: mu: {}, std: {}, pis {}, targets{}".format(mu_all.shape, std_all.shape, pis_all.shape, targets_all.shape))

    # I want : mu: 7 X 5 X N, std: 5 X N, pis: 5 X N, targets_all: 7 * N
    mu_all = mu_all.permute(2,1,0).cpu().numpy()
    std_all = std_all.permute(2,1,0).cpu().numpy()
    pis_all = pis_all.permute(1,0).cpu().numpy()
    targets_all = targets_all.permute(1,0).cpu().numpy()

    print("Shapes: mu: {}, std: {}, pis {}, targets{}".format(mu_all.shape, std_all.shape, pis_all.shape, targets_all.shape))


    fig = plt.figure()

    print("STD example", std_all[0][0])

    n_joints, n_components, n_samples = mu_all.shape[0], mu_all.shape[1], mu_all.shape[2]
    print("joints {} components {} samples {}".format(n_joints, n_components, n_samples))
    for joint_id in range(n_joints):
        plt.subplot(ceil(n_joints / 3.0), 4, joint_id + 1)
        for pi_id in range(n_components):
            plt.plot(mu_all[joint_id, pi_id], label="pi-{}".format(pi_id))
            plt.fill_between(range(n_samples), mu_all[joint_id, pi_id] - std_all[joint_id, pi_id], mu_all[joint_id, pi_id] + std_all[joint_id, pi_id], alpha=0.1)


        weighted_mus = mu_all[joint_id] * pis_all
        averaged_mus = weighted_mus.sum(0)
        print("muall: {}, pis_all: {} Weighted: {}, Averaged: {}".format(mu_all.shape, pis_all.shape, weighted_mus.shape, averaged_mus.shape))
        plt.plot(averaged_mus, label="Averaged")
        plt.plot(targets_all[joint_id], label="Targets")

        plt.legend()
        plt.title(["x", "y", "z", "r-x", "r-y", "r-z"][joint_id])
        plt.xlabel("t")
        plt.ylabel("Normed Radians")

    plt.subplot(ceil(n_joints / 3.0), 4, n_joints + 1)
    for pi_id in range(n_components):
        plt.plot(pis_all[pi_id], label="pi_{}".format(pi_id))
    plt.legend()
    plt.xlabel("t")
    plt.ylabel("Component Weight")
    plt.title("Pis")

    plt.show()


def chart_demo_predictions(model_path, demo_path, demo_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_config = load_json("config/experiment_config.json")
    # TODO: This is in multiple places, so I think it needs to be config
    im_params = exp_config["image_config"]

    im_trans = Compose([Crop(im_params["crop_top"], im_params["crop_left"],
                             im_params["crop_height"], im_params["crop_width"]),
                        Resize(im_params["resize_height"], im_params["resize_width"])])
    model = load_model(model_path, device, im_params["resize_height"], im_params["resize_width"], exp_config["nn_joint_names"])

    demo_set, demo_loader = load_demos(
        demo_path,
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        False,
        device,
        from_demo=demo_num,
        to_demo=demo_num + 1
    )

    # show_torched_im(demo_set[0][0][0])

    model.eval()
    currents = []
    ests = []
    trues = []

    with torch.no_grad():
        for (ins, targets) in demo_loader:
            _, current_pos = ins
            est_next = model(*ins)
            est_next = [x.cpu().numpy() for x in est_next]
            true_next = [x.cpu().numpy() for x in targets]
            current_pos = [x.cpu().numpy() for x in current_pos]

            ests.extend(est_next)
            trues.extend(true_next)
            currents.extend(current_pos)

    ests = np.array(ests).transpose()
    trues = np.array(trues).transpose()
    currents = np.array(currents).transpose()

    fig = plt.figure()


    for i, _ in enumerate(ests):
        c_ax = fig.add_subplot(ceil(len(ests) / 3.0), 3, i + 1)
        c_ax.plot()
        # c_ax.plot(currents[i], label="Current Joint")
        c_ax.plot(ests[i], label="Estimated Next Joints")
        c_ax.plot(trues[i], label="True Next Joints")
        c_ax.legend()
        c_ax.title.set_text(exp_config["nn_joint_names"][i])
        c_ax.set_xlabel("t")
        c_ax.set_ylabel("Normed Radians")
    
    plt.show()
    

# Run it through the device, get the results, and display them. This should already be in the animation thing
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: chartResults.py <results-path> <demos-folder> <demo-num>")
        sys.exit(0)

    log_path = sys.argv[1]
    demos_folder = sys.argv[2]
    demo_num = int(sys.argv[3])
    model_path = "{}/e2e_control_full.pt".format(log_path)

    chart_train_validation_error("{}/train.txt".format(log_path),
                                 "{}/validation.txt".format(log_path))

    chart_demo_predictions(model_path, demos_folder, demo_num)
