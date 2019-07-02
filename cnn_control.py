import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from load_data import load_demos, save_nums, load_pose_demos
from helper_funcs.rm import RobotModel

from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize

from model import setup_model, load_model, setup_joints_model


import os
from os.path import join 
from helper_funcs.utils import t_stamp, temp_print

# matplotlib.rcParams['animation.embed_limit'] = 100.0

def loss_batch(model, loss_func, batch_id, num_batches, in_batch, targets, optimizer=None):
    # forward + backward + optimize
    loss = loss_func(model(*in_batch), targets)

    if optimizer is not None:
        loss.backward()
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()

    temp_print("Batch {}/{} loss: {}".format(batch_id + 1, num_batches, loss.item()))
    return loss.item()

def loss_epoch(model, loss_func, d_loader, optimizer=None):
    losses = [loss_batch(model, loss_func, i, len(d_loader), ins, controls, optimizer) for i, (ins, controls) in enumerate(d_loader)]
    return sum(losses) / len(losses)


def train(model, train_loader, validation_loader, epochs, save_path=None):

    optimizer = optim.Adam(model.parameters(), eps=1e-7)
    loss_criterion = nn.MSELoss()

    print("Beginning Training")
    print("Epoch\tTrain\tValidation")
    
    avg_losses_train = []
    avg_losses_validation = []
    for epoch in range(epochs):
        # Training
        model.train()
        avg_loss_train = loss_epoch(model, loss_criterion, train_loader, optimizer)
        avg_losses_train.append(avg_loss_train)

        temp_print("{}\t{}\t-".format(epoch, avg_loss_train))

        # Validation
        model.eval()
        with torch.no_grad():
            avg_loss_validation = loss_epoch(model, loss_criterion, validation_loader)
            avg_losses_validation.append(avg_loss_validation)

        print('\x1b[2K\r'),
        print("{}\t{}\t{}".format(epoch, avg_loss_train, avg_loss_validation))


    if save_path is not None:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_nums(avg_losses_train, join(save_path, "train.txt"))
        save_nums(avg_losses_validation, join(save_path, "validation.txt"))
        torch.save(full_model.state_dict(), join(save_path, "e2e_control_full.pt"))
    
    print("Finished Training")
    return model


# Initial Setup
arm_joint_names = np.genfromtxt("config/arm_joint_names.txt", np.str)

K_kinect = np.array([366.096588, 0 , 268, 0, 366.096588, 192,0,0,1]).reshape(3,3)
robot_model = RobotModel(urdf_path="config/pr2.xml",
                         base_frame='base_link',
                         ee_frame='r_gripper_tool_frame',
                         camera_model=K_kinect)

batch_size = 32
num_epochs = 200
image_glob = "kinect2_qhd_image_color_rect_*.jpg"
im_height = 224
im_width = 224
im_trans = Compose([Crop(150, 475, 50, 800), Resize(224, 224)]) # TODO change to be configurable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_set, train_loader = load_demos("./demos/reach_blue_cube", image_glob, batch_size, arm_joint_names, im_trans, True, device, from_demo=0, to_demo=80)
validation_set, validation_loader = load_demos("./demos/reach_blue_cube", image_glob, batch_size, arm_joint_names, im_trans, False, device, from_demo=80)


"""
# TODO: Move this into an easy-to-use function
# im_canvas = img_heatmap_ax.imshow(raw_ims[0])
processed_im = train_set[0][0][0].permute(1,2,0)
print(processed_im.shape)
plt.imshow(processed_im)
plt.show()
"""

# Train the model
exp_name = "BlueCubeSmallNet"
log_path = "./logs/{}-{}".format(exp_name, t_stamp())
full_model = setup_model(device, im_height, im_width, arm_joint_names)
full_model = train(full_model, train_loader, validation_loader, num_epochs, log_path)

# Load an existing model
# load_model(join(log_path, "e2e_control_full.pt"), device)