import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math

from model import ZhangNet

from helper_funcs.utils import load_json, t_stamp
from helper_funcs.transforms import get_trans, get_grayscale_trans
from load_data import load_demos, show_torched_im, nn_input_to_imshow, append_tensors_as_csv
from chartResults import plot_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd
import numpy as np
from load_data import load_rgbd_demos, image_demo_paths

def ZhangLoss():
    def lf(next_pose_pred, aux_pred, target_pred, target_aux):
        l2_loss = F.mse_loss(next_pose_pred, target_pred)
        l1_loss = F.l1_loss(next_pose_pred, target_pred)

        eps = 1e-7
        cos_sims = torch.clamp(F.cosine_similarity(next_pose_pred, target_pred), -1 + eps, 1 - eps)
        angle_loss = torch.acos(cos_sims).mean()

        aux_loss = F.l1_loss(aux_pred, target_aux)
        full_loss = 1e-2 * l2_loss + 1.0 * l1_loss + 5e-3 * angle_loss + 1.0 * aux_loss
        if math.isnan(full_loss):
            print("Nan Encountered")

        return full_loss, l2_loss, l1_loss, angle_loss, aux_loss
    return lf


if __name__ == "__main__":
    # Test out data loading accuracy
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    train_set, train_loader = load_rgbd_demos(
        image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=0, to_demo=90)
        exp_config["batch_size"],
        "l_wrist_roll_link",
        get_trans(im_params, distorted=True),
        get_grayscale_trans(im_params),
        True,
        torch.device("cuda"))

    val_set, validation_loader = load_rgbd_demos(
        image_demo_paths(exp_config["demo_folder"], im_params["file_glob"], from_demo=90)
        exp_config["batch_size"],
        "l_wrist_roll_link",
        get_trans(im_params, distorted=True),
        get_grayscale_trans(im_params),
        False,
        torch.device("cuda"))

    model = ZhangNet(im_params["resize_height"], im_params["resize_width"])
    model.to(torch.device("cuda"))
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-4)
    loss_criterion = ZhangLoss()

    results_folder = "logs/zhang-{}".format(t_stamp())
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    for epoch in range(exp_config["epochs"]):
        model.train()
        train_losses = []
        with autograd.detect_anomaly():
            for i, in_batch in enumerate(train_loader):
                print("T Batch {}/{}".format(i, len(train_loader)), end='\r', flush=True)
                rgb_ins, depth_ins, past_ins, target_ins = in_batch 
                next_pred, aux_pred = model(rgb_ins, depth_ins, past_ins)
                train_loss = loss_criterion(next_pred, aux_pred, target_ins, past_ins[:, 4])
                train_losses.append([t.item() for t in train_loss])
                optimizer.zero_grad()
                train_loss[0].backward()
                optimizer.step()
        
        model.eval()
        val_losses = []
        for i, in_batch in enumerate(validation_loader):
            print("V Batch {}/{}".format(i, len(validation_loader)), end='\r', flush=True)
            with torch.no_grad():
                rgb_ins, depth_ins, past_ins, target_ins = in_batch 
                next_pred, aux_pred = model(rgb_ins, depth_ins, past_ins)
                val_loss = loss_criterion(next_pred, aux_pred, target_ins, past_ins[:, 4])
                val_losses.append([v.item() for v in val_loss])

        t_loss_mean = np.mean(train_losses, 0)
        v_loss_mean = np.mean(val_losses, 0)

        metrics = ["T-Full", "T-L2", "T-L1", "T-Ang", "T-Aux",
                   "V-Full", "V-L2", "V-L1", "V-Ang", "V-Aux"]

        append_tensors_as_csv(np.concatenate((t_loss_mean, v_loss_mean)), file_path=join(results_folder, "losses.csv"), cols=metrics)
        plot_csv(join(results_folder, "losses.csv"), join(results_folder, "losses.pdf"))

        print("{} T-Loss: {}, V-Loss {}".format(epoch, t_loss_mean[0], v_loss_mean[0]))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))