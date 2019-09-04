from model import ImageOnlyNet
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from helper_funcs.utils import load_json, temp_print, t_stamp
from helper_funcs.transforms import get_trans
from load_data import load_demos, show_torched_im, nn_input_to_imshow, append_tensors_as_csv
from chartResults import plot_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd
import numpy as np

def Weighted_MSE(joint_weights):
    def lf(predicted_joints, target_joints):
        squared_diffs = (predicted_joints - target_joints).pow(2)
        return (squared_diffs * joint_weights).mean()
    return lf

if __name__ == "__main__":
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_set, train_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        get_trans(im_params, distorted=True),
        True,
        torch.device("cuda"),
        from_demo=0,
        to_demo=60,
        skip_count=5)

    validation_set, validation_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        get_trans(im_params, distorted=False),
        False,
        torch.device("cuda"),
        from_demo=60,
        to_demo=80,
        skip_count=5)

    model = ImageOnlyNet(im_params["resize_height"], im_params["resize_width"], len(exp_config["nn_joint_names"]))
    model.to(torch.device("cuda"))
    optimizer = optim.Adam(model.parameters())
    joint_weights = torch.linspace(10, 1, len(exp_config["nn_joint_names"]), device=torch.device("cuda"))
    loss_criterion = Weighted_MSE(joint_weights) #nn.MSELoss()

    results_folder = "logs/image-only-weighted-joints{}".format(t_stamp())
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)


    for epoch in range(exp_config["epochs"]):
        model.train()
        train_losses = []
        for i, in_batch in enumerate(train_loader):

            temp_print("T Batch {}/{}".format(i, len(train_loader)))
            (img_ins, _), target_joints = in_batch
            predicted_joints = model(img_ins)
            train_loss = loss_criterion(predicted_joints, target_joints)
            train_losses.append(train_loss.item())
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        
        model.eval()
        val_losses = []
        for i, in_batch in enumerate(validation_loader):
            temp_print("V Batch {}/{}".format(i, len(validation_loader)))
            with torch.no_grad():
                (img_ins, _), target_joints = in_batch
                predicted_joints = model(img_ins)
                val_loss = loss_criterion(predicted_joints, target_joints)
                val_losses.append(val_loss.item())

        t_loss_mean = np.mean(train_losses)
        v_loss_mean = np.mean(val_losses)

        # df.loc[epoch] = [train_full_loss.item(), train_recon_loss.item(), train_kl_loss.item(), val_full_loss.item(), val_recon_loss.item(), val_kl_loss.item()]
        # print(df.loc[epoch])
        print("{} T-MSE: {}, V-MSE {}".format(epoch, t_loss_mean, v_loss_mean))

        metrics = ["T-MSE", "V-MSE"]
        append_tensors_as_csv([t_loss_mean, v_loss_mean], file_path=join(results_folder, "losses.csv"), cols=metrics)
        plot_csv(join(results_folder, "losses.csv"), join(results_folder, "losses.pdf"))

        if epoch % 10 == 0:
            torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))