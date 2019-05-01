#%%
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from glob import glob

from IPython import display
import matplotlib.pyplot as plt

import re
import cv2

import os
from os.path import join, split

def get_pose_and_control(im_paths, idx, ordered_joints):

    folder_path, im_name = split(im_paths[idx])
    _, im_name_next = split(im_paths[idx + 1])

    timestamp_regex = re.compile(r'.*_(\d+\.\d+)\.jpg')

    name_path = join(folder_path, timestamp_regex.sub(r'joint_names_\1.txt', im_name))
    pose_path = join(folder_path, timestamp_regex.sub(r'joint_pos_\1.txt', im_name))
    vel_path = join(folder_path, timestamp_regex.sub(r'joint_vel_\1.txt', im_name_next))


    names = np.genfromtxt(name_path, dtype=np.str)
    pose = np.genfromtxt(pose_path)
    vels = np.genfromtxt(vel_path)

    # Sometimes the joint names in each file might arrive out of order due to message passing
    # This line ensures joint velocities / poses correspond to joints in a fixed order
    joint_idxs = np.array([names.tolist().index(a) for a in ordered_joints])
    return pose[joint_idxs], vels[joint_idxs]


class ImagePoseControlDataset(Dataset):

    def __init__(self, images_by_demo, arm_joint_names):
        self.images_by_demo = images_by_demo
        self.arm_joint_names = arm_joint_names

        # We don't want sampler to pick last image of each demo
        # because the velocities are taken from the current_image_id + 1
        self.demos_strides = zip(images_by_demo, strides(images_by_demo, -1))

    def __len__(self):
        return sum(len(demo) - 1 for demo in self.images_by_demo)

    def __getitem__(self, idx):
        demo, demo_stride = find_last(lambda ds: idx >= ds[1], self.demo_strides)
        img_id = idx - demo_stride
        np_pose, np_control = get_pose_and_control(demo, img_id, self.arm_joint_names)

        # Crop to only the useful area, then scale it up to standard dimensions
        cropped_img = cv2.imread(demo[img_id])[125:320, 360:450,:]
        resized_img = cv2.resize(cropped_img , (224, 224))
        np_img = rescale_pixel_intensities(resized_img)

        # Numpy -> Torch =  (Height x Width x Colour) ->  (Colour x Height x Width)
        img = torch.from_numpy(np_img.transpose(2, 0, 1)).to(dtype=torch.float)
        pose = torch.from_numpy(np_pose).to(dtype=torch.float)
        control = torch.from_numpy(np_control).to(dtype=torch.float)

        return {"image": img, "pose": pose, "control": control}


def rescale_pixel_intensities(img):
    scaled_img = img / 127.5 # Divide by half of full image range ([0, 255] -> [0, 2])
    return scaled_img - 1.0 # Translate by 1 [0, 2] -> [-1, 1]


def strides(lists, lengths_offset):
    return np.cumsum([0] + [len(l) + lengths_offset for l in lists[:-1]])


def find_last(pred, lst):
    return next(x for x in reversed(lst) if pred(x))


class ImageAndJointsNet(nn.Module):
    def __init__(self, image_height, image_width, joint_dim):
        super(ImageAndJointsNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 7)
        self.conv2 = nn.Conv2d(3, 3, 5)
        self.conv3 = nn.Conv2d(3, 3, 3)

        o_height, o_width = output_size(image_height, image_width, 7)
        o_height, o_width = output_size(o_height, o_width, 5)
        o_height, o_width = output_size(o_height, o_width, 3)

        linear_input_size = o_height * o_width * 3
        print("Linear Input Size:", linear_input_size)

        self.linear1 = nn.Linear(linear_input_size, 32)
        self.linear2 = nn.Linear(32, 32)
        self.linear3 = nn.Linear(32 + joint_dim, joint_dim)

    def forward(self, img_ins, pose_ins):
        img_ins = F.relu(self.conv1(img_ins))
        img_ins = F.relu(self.conv2(img_ins))
        img_ins = F.relu(self.conv3(img_ins))
        img_ins = torch.flatten(img_ins, 1)
        img_ins = F.relu(self.linear1(img_ins))
        img_ins = F.relu(self.linear2(img_ins))

        image_and_pos = torch.cat((img_ins, pose_ins), dim=1)
        output = self.linear3(image_and_pos)

        return output


def output_size(in_height, in_width, kernel_size, stride=1, padding=0):
    out_height = int((in_height - kernel_size + padding * 2) / stride) + 1
    out_width = int((in_width - kernel_size + padding * 2) / stride) + 1
    return (out_height, out_width)


#%%
arm_joint_names = np.genfromtxt("./arm_joint_names.txt", np.str)
demos_train_root = "./demos/train"
demos_test_root = "./demos/test"

train_demo_paths = join(demos_train_root, os.listdir(demos_train_root))
test_demo_paths = join(demos_test_root, os.listdir(demos_test_root))

train_demos = [sorted(glob(join(demo_path, "kinect_colour_*.jpg"))) for demo_path in train_demo_paths]
test_demos = [sorted(glob(join(demo_path, "kinect_colour_*.jpg"))) for demo_path in test_demo_paths]

batch_size = 64

train_set = ImagePoseControlDataset(train_demos, arm_joint_names)
test_set = ImagePoseControlDataset(test_demos, arm_joint_names)

train_loader = DataLoader(train_set, batch_size, shuffle=True)


# Train the model
full_model = ImageAndJointsNet(224, 224, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model.to(device)
print(device)
print(full_model)  # If this isn't enough info, try the "pytorch-summary" package

optimizer = optim.Adam(full_model.parameters(), eps=1e-7)
loss_criterion = nn.MSELoss()

#%%
print("Beginning Training")
full_model.train()
num_epochs = 30
for epoch in range(num_epochs):
    running_loss = 0.0
    print(f"Epoch:{epoch+1}/{num_epochs}")
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        img_ins, pose_ins, controls = data["image"], data["pose"], data["control"]
        img_ins, pose_ins, controls = img_ins.to(device, dtype=torch.float), pose_ins.to(device), controls.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = full_model(img_ins, pose_ins)
        loss = loss_criterion(outputs, controls)
        loss.backward()
        optimizer.step()

        # print statistics
        current_loss = loss.item()
        running_loss += current_loss
        print(f"Batch {i + 1} of {len(train_loader)}, loss: {current_loss}", end='\r', flush=True)
    print(f"Avg Loss: {running_loss / len(train_loader)}")

print("Finished Training")
torch.save(full_model.state_dict(), "./logs/e2e_control_full.pt")


#%%
# Load an existing model
full_model = ImageAndJointsNet(224, 224, 8)
full_model.load_state_dict(torch.load("./logs/e2e_control_full.pt"))
full_model.to(device)

#%%
# Show attribution video
fig = plt.figure(figsize=(15,5))
img_heatmap_ax = fig.add_subplot(1, 2, 1)
control_ax = fig.add_subplot(2, 2, 2)
control_est_ax = fig.add_subplot(2, 2, 4)

heatmap = np.zeros((224, 224))

c = []
cest = []

full_model.eval()
for idx in range(200):
    print(idx)
    data = train_set[idx]
    im_in, pose = data["image"], data["pose"]
    im_in, pose = im_in.to(device), pose.to(device)
    controls = data["control"]
    
    c.append(controls.numpy())

    im_original = cv2.imread(image_paths[idx])

    outputs = full_model(im_in.unsqueeze(0), pose.unsqueeze(0))
    cest.append(outputs.data.cpu().numpy())

    img_heatmap_ax.cla()  # cla = Clear axis
    img_heatmap_ax.imshow(im_original[125:320, 320:450, [2,1,0]])
    # img_heatmap_ax.imshow(Lxd[125:320, 360:450], alpha=0.5)

    # Plot the control velocities over time
    control_ax.cla()
    control_ax.plot(np.squeeze(np.array(c)))
    control_ax.grid() 

    # Plot the estimated velocities over time
    control_est_ax.cla()
    control_est_ax.plot(np.squeeze(np.array(cest)))
    control_est_ax.grid()

    # Get rid of previous image, then display the latest one
    display.clear_output(wait=True)
    display.display(plt.gcf())
