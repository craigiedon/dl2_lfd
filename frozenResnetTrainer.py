import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torchvision.models import resnet50


from helper_funcs.utils import load_json, temp_print, t_stamp
from helper_funcs.transforms import get_resnet_trans
from load_data import load_demos, show_torched_im, nn_input_to_imshow, append_tensors_as_csv
from chartResults import plot_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd
import numpy as np

# Oh... will probably have to change the input file format and stuff too!
class ResnetJointPredictor(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ResnetJointPredictor, self).__init__()
        self.resnet = resnet50(pretrained=True, progress=True)
        for param in self.resnet.parameters():
            param.requires_grad = False

        linear_input_size = 1000 # TODO: this is clearly wrong

        hidden_layer_dim = 100

        self.linear1 = nn.Linear(linear_input_size, hidden_layer_dim)
        self.lin_drop = nn.Dropout()

        self.linear2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.linear3 = nn.Linear(hidden_layer_dim, joint_dim)

    def forward(self, img_ins):
        c_out = self.resnet(img_ins)
        # flattened_conv = torch.flatten(c_out, 1)

        lin1_out = self.lin_drop(F.leaky_relu(self.linear1(c_out)))
        lin2_out = F.leaky_relu(self.linear2(lin1_out))
        output = self.linear3(lin2_out)

        return output

# For data loading setup, have to ensure image size is 224, that images are CHW, that they are in range 0,1, then normalized using
# mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]. E.g., use transforms.Normalize (or some sped up custom version of this...)
# Setup data loading, optimizer, loss criterion etc
# Load in the resnet model (just define it here I guess)
# Run the training loop as usual and log results as in image one... (can do a weighted an unweighted thing here if you want)
if __name__ == "__main__":
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]

    train_set, train_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        get_resnet_trans(im_params, distorted=True),
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
        get_resnet_trans(im_params, distorted=False),
        False,
        torch.device("cuda"),
        from_demo=60,
        to_demo=80,
        skip_count=5)

    model = ResnetJointPredictor(im_params["resize_height"], im_params["resize_width"], len(exp_config["nn_joint_names"]))
    model.to(torch.device("cuda"))
    optimizer = optim.Adam(model.parameters())
    # joint_weights = torch.linspace(10, 1, len(exp_config["nn_joint_names"]), device=torch.device("cuda"))
    loss_criterion = nn.MSELoss()

    results_folder = "logs/frozen-resnet{}".format(t_stamp())
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