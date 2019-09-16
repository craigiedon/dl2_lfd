import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

from model import PosePlusStateNet

from helper_funcs.utils import load_json, t_stamp
from helper_funcs.transforms import get_trans, get_grayscale_trans
from load_data import load_pose_state_demos, image_demo_paths, show_torched_im, nn_input_to_imshow, append_tensors_as_csv
from chartResults import plot_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd
import numpy as np
from mdn import mdn_loss

def L1L2Cost(l1_weight, l2_weight):
    def lf(prediction, target):
        l1_loss = F.l1_loss(prediction, target)
        l2_loss = F.mse_loss(prediction, target)
        full_loss = (l1_weight * l1_loss) + (l2_weight * l2_loss)
        # full_loss = l1_loss
        return full_loss, l1_loss, l2_loss
    return lf

def batch_run(in_batch, model, loss_fn, optimizer=None):
    current_pose, goal_pose, target_pose = in_batch
    pred_pi, pred_std, pred_mu = model(current_pose, goal_pose)
    loss = loss_fn(pred_pi, pred_std, pred_mu, target_pose)
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss

if __name__ == "__main__":
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_set, train_loader = load_pose_state_demos(
        image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=99),
        exp_config["batch_size"],
        "l_wrist_roll_link",
        "r_wrist_roll_link",
        True,
        torch.device("cuda"))

    val_set, validation_loader = load_pose_state_demos(
        image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=99),
        exp_config["batch_size"],
        "l_wrist_roll_link",
        "r_wrist_roll_link",
        False,
        torch.device("cuda"))

    model = PosePlusStateNet(100, 2)
    model.to(torch.device("cuda"))
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_criterion = mdn_loss

    results_folder = "logs/poseGoalStateMDN20-{}".format(t_stamp())
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for epoch in range(exp_config["epochs"]):
        model.train()
        train_losses = []
        for i, in_batch in enumerate(train_loader):
            print("T Batch {}/{}".format(i, len(train_loader)), end='\r', flush=True)
            train_loss = batch_run(in_batch, model, loss_criterion, optimizer)
            train_losses.append(train_loss.item())

        model.eval()
        val_losses = []
        for i, in_batch in enumerate(validation_loader):
            print("V Batch {}/{}".format(i, len(validation_loader)), end='\r', flush=True)
            with torch.no_grad():
                val_loss = batch_run(in_batch, model, loss_criterion)
                val_losses.append(val_loss.item())

        t_loss_mean = np.mean(train_losses, 0)
        v_loss_mean = np.mean(val_losses, 0)

        # metrics = ["T-Full", "T-L1", "T-L2", "V-Full", "V-L1", "V-L2"]
        metrics = ["T-Full", "V-Full"]

        append_tensors_as_csv([t_loss_mean, v_loss_mean], file_path=join(results_folder, "losses.csv"), cols=metrics)
        plot_csv(join(results_folder, "losses.csv"), join(results_folder, "losses.pdf"))

        print("{} T-Loss: {}, V-Loss {}".format(epoch, t_loss_mean, v_loss_mean))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))
