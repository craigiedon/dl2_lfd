import numpy as np
import torch
from torch import nn, optim
from load_data import load_demos, save_num_append, show_torched_im, load_constant_joint_vals
from constraints import JointLimitsConstraint, EndEffectorPosConstraint, StayInZone, MoveSlowly, MatchOrientation, SmoothMotion

from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize
from helper_funcs.rm import RobotModel, joints_lower_limits, joints_upper_limits, forward_kinematics

from model import setup_model, setup_joints_model
from oracle import evaluate_constraint


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
        constr_loss, constr_acc = 0, 1

    full_loss = args["constr_weight"] * constr_loss + training_loss
        

    if optimizer is not None:
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()

    temp_print("Batch {}/{} train_loss: {}, constr_loss: {}, constr_acc: {}, full_loss: {}"
               .format(batch_id + 1, num_batches, training_loss.item(), constr_loss, constr_acc, full_loss.item()))

    return {"training_loss": training_loss.item(),
            "constr_loss": constr_loss.item(),
            "constr_acc": constr_acc.item(),
            "full_loss": full_loss.item()}

def loss_epoch(model, loss_func, d_loader, args, constraint=None, optimizer=None):
    epoch_losses = {"training_loss": [],
                    "constr_loss": [],
                    "constr_acc": [],
                    "full_loss": []}
    for i, (ins, controls) in enumerate(d_loader):
        loss_metrics = loss_batch(model, loss_func, i, len(d_loader), ins, controls, args, constraint, optimizer)

        epoch_losses["training_loss"].append(loss_metrics["training_loss"])
        epoch_losses["constr_loss"].append(loss_metrics["constr_loss"])
        epoch_losses["constr_acc"].append(loss_metrics["constr_acc"])
        epoch_losses["full_loss"].append(loss_metrics["full_loss"])

    return {metric: np.mean(vals) for metric, vals in epoch_losses.items()}


def train(model, train_loader, validation_loader, epochs, constraint, args, save_path):

    optimizer = optim.Adam(model.parameters())
    loss_criterion = nn.MSELoss()

    print("Beginning Training")

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for epoch in range(epochs):
        print("Epoch {}".format(epoch))
        # Training
        model.train()
        avg_loss_train = loss_epoch(model, loss_criterion, train_loader, args, constraint, optimizer)
        save_num_append(avg_loss_train, join(save_path, "train.txt"))

        temp_print("{}\t{}\t-".format(epoch, avg_loss_train))

        # Validation
        model.eval()
        with torch.no_grad():
            avg_loss_validation = loss_epoch(model, loss_criterion, validation_loader, args, constraint)
            save_num_append(avg_loss_validation, join(save_path, "validation.txt"))

        print("Train: {}".format(avg_loss_train))
        print("Validation: {}".format(avg_loss_validation))
        print()

    torch.save(full_model.state_dict(), join(save_path, "e2e_control_full.pt"))
    print("Finished Training")
    return model


# Initial Setup

demo_folder = "./demos/reach_blue_cube"
nn_joint_names = np.genfromtxt("config/nn_joint_names.txt", np.str)
constant_joint_names = np.loadtxt("config/constant_joint_names.txt", np.str, ndmin=1)

print(nn_joint_names)
print(constant_joint_names)

constant_param_map = load_constant_joint_vals(demo_folder, constant_joint_names)
# Take in the name of demos folder, and the joint names you want, output just a map for each of the values. Should be easy if you just follow the template
# for the get pose function

K_kinect = np.array([366.096588, 0 , 268, 0, 366.096588, 192,0,0,1]).reshape(3,3)
robot_model = RobotModel(urdf_path="config/pr2.xml",
                         base_frame='base_link',
                         end_effector_frame='r_gripper_tool_frame',
                         camera_model=K_kinect,
                         constant_params=constant_param_map)

args = {"constr_weight": 0.1}
batch_size = 32
num_epochs = 500

im_params = load_json("config/image_params.json")
im_trans = Compose([
    Crop(im_params["crop_top"], im_params["crop_bottom"],
         im_params["crop_left"], im_params["crop_right"]),
    Resize(im_params["resize_height"], im_params["resize_width"])])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set, train_loader = load_demos(
    demo_folder, im_params["file_glob"], batch_size, nn_joint_names, im_trans, True, device, from_demo=0, to_demo=80)
validation_set, validation_loader = load_demos(
    demo_folder, im_params["file_glob"], batch_size, nn_joint_names, im_trans, False, device, from_demo=80)

        
# Train the model
exp_name = "BlueCubeLevineNet"
log_path = "./logs/{}-{}".format(exp_name, t_stamp())
full_model = setup_model(device, im_params["resize_height"], im_params["resize_width"], nn_joint_names)

lower_bounds = joints_lower_limits(nn_joint_names, robot_model)
upper_bounds=  joints_upper_limits(nn_joint_names, robot_model)

print(lower_bounds)
print(upper_bounds)

constraint = StayInZone(full_model, mbs, maxbs, nn_joint_names, robot_model)
constraint = MoveSlowly(full_model, 2.0, nn_joint_names, robot_model)
constraint = MatchOrientation(full_model, target_orientation, nn_joint_names, robot_model)
constraint = SmoothMotion(full_model, 0.5, nn_joint_names, robot_model)

print(constraint)
full_model = train(full_model, train_loader, validation_loader,
                   num_epochs, constraint, args, log_path)
