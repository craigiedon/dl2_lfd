from model import ImageOnlyMDN
from mdn import mdn_loss, mdn_sample
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
        to_demo=1,
        skip_count=5)

    validation_set, validation_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        get_trans(im_params, distorted=False),
        False,
        torch.device("cuda"),
        from_demo=1,
        to_demo=2,
        skip_count=5)

    model = ImageOnlyMDN(im_params["resize_height"], im_params["resize_width"], len(exp_config["nn_joint_names"]), 5)
    model.to(torch.device("cuda"))
    current_lr = 1e-5
    optimizer = optim.Adam(model.parameters(), lr=current_lr)
    # joint_weights = torch.linspace(10, 1, len(exp_config["nn_joint_names"]), device=torch.device("cuda"))
    loss_criterion = mdn_loss #Weighted_MSE(joint_weights) #nn.MSELoss()

    results_folder = "logs/image-only-mdn-dropout-{}".format(t_stamp())
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)


    for epoch in range(exp_config["epochs"]):
        model.train()
        train_losses = []
        for i, in_batch in enumerate(train_loader):

            temp_print("T Batch {}/{}".format(i, len(train_loader)))
            (img_ins, _), target_joints = in_batch
            mu, std, pi = model(img_ins)
            train_loss = loss_criterion(mu, std, pi, target_joints)
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
                mu, std, pi = model(img_ins)
                val_loss = loss_criterion(mu, std, pi, target_joints)
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
        
        if (epoch + 1) % 100 == 0:
            current_lr *= 0.1
            optimizer = optim.Adam(model.parameters(), lr = current_lr)
            print("Updated optimizer...")
    torch.save(model.state_dict(), join(results_folder, "learned_model_epoch_final.pt"))