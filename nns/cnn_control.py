import numpy as np
import torch
from torch import nn, optim
from load_data import load_demos, save_dict_append, save_list_append, show_torched_im, load_constant_joint_vals
from constraints import JointLimitsConstraint, EndEffectorPosConstraint, StayInZone, MoveSlowly, MatchOrientation, SmoothMotion

from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize
from helper_funcs.rm import RobotModel, joints_lower_limits, joints_upper_limits, forward_kinematics

from model import setup_model, setup_joints_model
from oracle import evaluate_constraint

from collections import defaultdict


import os
from os.path import join 
from helper_funcs.utils import t_stamp, temp_print, load_json

# matplotlib.rcParams['animation.embed_limit'] = 100.0

def loss_batch(model, loss_func, batch_id, num_batches, in_batch, targets, args, constraint=None, optimizer=None):
    # forward + backward + optimize
    training_loss = loss_func(model(*in_batch), targets)

    if constraint is not None:
        old_model_mode = model.training
        model.eval()
        constr_loss, constr_acc = evaluate_constraint(in_batch, targets, constraint, args)
        model.train(old_model_mode)
    else:
        constr_loss, constr_acc = 0.0, 1.0

    full_loss = args["constr_weight"] * constr_loss + training_loss
        

    if optimizer is not None:
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

    temp_print("Batch {}/{} train_loss: {}, constr_loss: {}, constr_acc: {}, full_loss: {}"
               .format(batch_id + 1, num_batches, training_loss.item(), constr_loss, constr_acc, full_loss.item()))

    return {"training_loss": training_loss.item(),
            "constr_loss": constr_loss.item() if constraint is not None else constr_loss,
            "constr_acc": constr_acc.item() if constraint is not None else constr_acc,
            "full_loss": full_loss.item()}


def loss_epoch(model, loss_func, d_loader, args, constraint=None, optimizer=None):
    epoch_losses = defaultdict(list)

    for i, (ins, controls) in enumerate(d_loader):
        loss_metrics = loss_batch(model, loss_func, i, len(d_loader), ins, controls, args, constraint, optimizer)

        for k, v in loss_metrics.items():
            epoch_losses[k].append(v)

    return epoch_losses


def mean_dict(d):
    return {metric: np.mean(vals) for metric, vals in d.items()}


def train(model, train_loader, validation_loader, epochs, constraint, args, save_path):

    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.MSELoss()

    print("Beginning Training")

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print("Epoch 0")
    # Getting the starting error for train and validation set, so no need to record grads
    model.eval()
    with torch.no_grad():
        # Training
        avg_loss_train = mean_dict(loss_epoch(model, loss_criterion, train_loader, args, constraint))
        save_dict_append(avg_loss_train, join(save_path, "train.txt"))

        # Validation
        avg_loss_validation = mean_dict(loss_epoch(model, loss_criterion, validation_loader, args, constraint))
        save_dict_append(avg_loss_validation, join(save_path, "validation.txt"))
    
    print()
    
    # Actual learning Epochs
    for epoch in range(1, epochs + 1):
        print("Epoch {}".format(epoch))
        # Training
        model.train()
        losses_train = loss_epoch(model, loss_criterion, train_loader, args, constraint, optimizer)
        avg_loss_train = mean_dict(losses_train)

        save_list_append(losses_train["training_loss"], join(save_path, "batches-train.txt"))
        save_dict_append(avg_loss_train, join(save_path, "train.txt"))

        temp_print("{}\t{}\t-".format(epoch, avg_loss_train))

        # Validation
        model.eval()
        with torch.no_grad():
            losses_validation = loss_epoch(model, loss_criterion, validation_loader, args, constraint)
            avg_loss_validation = mean_dict(losses_validation)

            save_list_append(losses_validation["training_loss"], join(save_path, "batches-validation.txt"))
            save_dict_append(avg_loss_validation, join(save_path, "validation.txt"))

        print("Train: {}".format(avg_loss_train))
        print("Validation: {}".format(avg_loss_validation))
        print()

        if epoch % 5 == 0:
            torch.save(full_model.state_dict(), join(save_path, "e2e_control_e{}.pt".format(epoch)))

    torch.save(full_model.state_dict(), join(save_path, "e2e_control_full.pt"))
    print("Finished Training")
    return model


# Initial Setup

exp_config = load_json("config/experiment_config.json")

constant_param_map = load_constant_joint_vals(exp_config["demo_folder"], exp_config["constant_joint_names"])

K_kinect = np.array([366.096588, 0 , 268, 0, 366.096588, 192,0,0,1]).reshape(3,3)
robot_model = RobotModel(urdf_path="config/pr2.xml",
                         base_frame='base_link',
                         end_effector_frame='r_gripper_tool_frame',
                         camera_model=K_kinect,
                         constant_params=constant_param_map)

im_params = exp_config["image_config"]
im_trans = Compose([
    Crop(im_params["crop_top"], im_params["crop_left"],
         im_params["crop_height"], im_params["crop_width"]),
    Resize(im_params["resize_height"], im_params["resize_width"])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set, train_loader = load_demos(
    exp_config["demo_folder"],
    im_params["file_glob"],
    exp_config["batch_size"],
    exp_config["nn_joint_names"],
    im_trans,
    True,
    device,
    from_demo=0,
    to_demo=90)

validation_set, validation_loader = load_demos(
    exp_config["demo_folder"],
    im_params["file_glob"],
    exp_config["batch_size"],
    exp_config["nn_joint_names"],
    im_trans,
    False,
    device,
    from_demo=90,
    to_demo=100)


"""
for i in range(0, 9):
    show_torched_im(train_set[i][0][0])
"""


# Train the model
log_path = "./logs/{}-{}".format(exp_config["experiment_name"], t_stamp())
full_model = setup_model(device, im_params["resize_height"], im_params["resize_width"], exp_config["nn_joint_names"])

lower_bounds = joints_lower_limits(exp_config["nn_joint_names"], robot_model)
upper_bounds = joints_upper_limits(exp_config["nn_joint_names"], robot_model)

print(lower_bounds)
print(upper_bounds)

# constraint = StayInZone(full_model, mbs, maxbs, nn_joint_names, robot_model)
# constraint = MoveSlowly(full_model, 2.0, nn_joint_names, robot_model)
# constraint = SmoothMotion(full_model, 0.5, nn_joint_names, robot_model)

# constraint = MatchOrientation(full_model, target_orientation, exp_config["nn_joint_names"], robot_model)
constraint = None

print(constraint)
full_model = train(full_model, train_loader, validation_loader, exp_config["epochs"], constraint, exp_config, log_path)
