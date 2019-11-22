import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

from model import PosePlusStateNet

from helper_funcs.utils import load_json, t_stamp
from helper_funcs.transforms import get_trans, get_grayscale_trans
from load_data import load_pose_state_demos, image_demo_paths, show_torched_im, nn_input_to_imshow, append_tensors_as_csv, PoseAndGoal, DataLoader, DeviceDataLoader
from chartResults import plot_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd
import numpy as np
from mdn import mdn_loss

from oracle import evaluate_constraint
from constraints import StayInZone, MoveSlowly

def L1L2Cost(l1_weight, l2_weight):
    def lf(prediction, target):
        l1_loss = F.l1_loss(prediction, target)
        l2_loss = F.mse_loss(prediction, target)
        full_loss = (l1_weight * l1_loss) + (l2_weight * l2_loss)
        # full_loss = l1_loss
        return full_loss, l1_loss, l2_loss
    return lf

def batch_run(in_batch, model, loss_fn, constraint=None, constraint_args=None, optimizer=None):
    current_pose, goal_pose, target_pose = in_batch
    pred_pose = model(current_pose, goal_pose)
    loss = loss_fn(pred_pose, target_pose)

    if constraint is not None:
        constraint_loss, constraint_acc = evaluate_constraint((current_pose, goal_pose), target_pose, constraint, constraint_args)
        full_loss = 0.1 * constraint_loss + loss
    else:
        full_loss = loss
        constraint_loss = torch.tensor(0.0)
        constraint_acc = torch.tensor(1.0)
    

    if optimizer is not None:
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
    return loss, constraint_loss, constraint_acc

if __name__ == "__main__":
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=30)
    train_set = PoseAndGoal(train_paths, "l_wrist_roll_link", "r_wrist_roll_link", skip_count=10)
    train_loader = DeviceDataLoader(DataLoader(train_set, exp_config["batch_size"], shuffle=True), torch.device("cuda"))

    val_paths = image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=30)
    val_set = PoseAndGoal(val_paths, "l_wrist_roll_link", "r_wrist_roll_link", skip_count=10)
    val_loader = DeviceDataLoader(DataLoader(val_set, exp_config["batch_size"], shuffle=False), torch.device("cuda"))


    model = PosePlusStateNet(100)
    model.to(torch.device("cuda"))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_criterion = nn.L1Loss()

    results_folder = "logs/cupPourPoseGoalZoneStay-{}".format(t_stamp())
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    min_bounds = torch.tensor([0.55, -0.1, 0.8], device=torch.device("cuda"))
    max_bounds = torch.tensor([0.765, 0.24, 1.24], device=torch.device("cuda"))
    constraint = StayInZone(model, min_bounds, max_bounds)
    # constraint = MoveSlowly(model, 0.02)
    # constraint=None

    for epoch in range(exp_config["epochs"]):
        model.train()
        train_losses = []
        for i, in_batch in enumerate(train_loader):
            print("T Batch {}/{}".format(i, len(train_loader)), end='\r', flush=True)
            train_loss = batch_run(in_batch, model, loss_criterion, constraint=constraint, constraint_args=exp_config, optimizer=optimizer)
            train_losses.append([t.item() for t in train_loss])

        model.eval()
        val_losses = []
        for i, in_batch in enumerate(val_loader):
            print("V Batch {}/{}".format(i, len(val_loader)), end='\r', flush=True)
            val_loss = batch_run(in_batch, model, loss_criterion, constraint=constraint, constraint_args=exp_config)
            val_losses.append([t.item() for t in val_loss])

        t_loss_mean = np.mean(train_losses, 0)
        v_loss_mean = np.mean(val_losses, 0)

        # metrics = ["T-Full", "T-L1", "T-L2", "V-Full", "V-L1", "V-L2"]
        metrics = ["T-Full", "T-ConL", "T-ConA", "V-Full", "V-ConL", "V-ConA"]

        append_tensors_as_csv(np.concatenate([t_loss_mean, v_loss_mean]), file_path=join(results_folder, "losses.csv"), cols=metrics)
        plot_csv(join(results_folder, "losses.csv"), join(results_folder, "losses.pdf"))

        print("{} T-Loss: {:1.3e}, T-ConL {:1.3e}, T-ConA {}, V-Loss {:1.3e}, V-ConL {:1.3e}, V-ConA {}"
        .format(epoch, t_loss_mean[0], t_loss_mean[1], t_loss_mean[2],
                v_loss_mean[0], v_loss_mean[1], v_loss_mean[2]))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
