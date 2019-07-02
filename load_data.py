import torch
import numpy as np
import os
from os.path import join, split
import re
import cv2
from torch.utils.data import Dataset, DataLoader
from helper_funcs.utils import find_last
from glob import glob


def image_demo_paths(demos_root, image_glob):
    demo_paths = [join(demos_root, d) for d in os.listdir(demos_root)]
    demo_images = [sorted(glob(join(demo_path, image_glob))) for demo_path in demo_paths]
    return demo_images


def load_demos(demos_folder, image_glob, batch_size, joint_names, im_trans, shuffled, device, from_demo=None, to_demo=None, frame_limit=None):
    demo_paths = image_demo_paths(demos_folder, image_glob)

    demos = [d[0:frame_limit] for d in demo_paths[from_demo:to_demo]]
    d_set = ImagePoseControlDataset(demos, joint_names, im_trans)
    d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

    return d_set, d_loader


def load_pose_demos(demos_folder, image_glob, batch_size, joint_names, shuffled, device, from_demo=None, to_demo=None, frame_limit=None):
    demo_paths = image_demo_paths(demos_folder, image_glob)
    demos = [d[0:frame_limit] for d in demo_paths[from_demo:to_demo]]

    d_set = PoseControlDataset(demos, joint_names)
    d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

    return d_set, d_loader


def save_nums(lines, file_path):
    with open(file_path, "w") as save_file:
        for l in lines:
            save_file.write("{}\n".format(l))


class DeviceDataLoader:
    def __init__(self, data_loader, device):
        self.data_loader = data_loader
        self.device = device

    def __len__(self):
        return len(self.data_loader)

    def __iter__(self):
        batches = iter(self.data_loader)
        for b in batches:
            yield (send_to_device_rec(b, self.device))


def send_to_device_rec(xs, device):
    if isinstance(xs, torch.Tensor):
        return xs.to(device)

    return [send_to_device_rec(x,device) for x in xs]


class PoseControlDataset(Dataset):
    def __init__(self, images_by_demo, arm_joint_names):
        self.images_by_demo = images_by_demo
        self.arm_joint_names = arm_joint_names

        # We don't want sampler to pick last image of each demo
        # because the velocities are taken from the current_image_id + 1
        self.demo_strides = list(zip(images_by_demo, strides(images_by_demo, -1)))

    def __len__(self):
        return sum(len(demo) - 1 for demo in self.images_by_demo)

    def __getitem__(self, idx):
        demo, demo_stride = find_last(lambda ds: idx >= ds[1], self.demo_strides)
        img_id = idx - demo_stride
        np_pose, np_control = get_pose_and_control(demo, img_id, self.arm_joint_names)

        # Normalize joint angles by encoding with sin/cos
        wrapped_pose = np.arctan2(np.sin(np_pose), np.cos(np_pose))
        pose = torch.from_numpy(wrapped_pose).to(dtype=torch.float)

        control = torch.from_numpy(np_control).to(dtype=torch.float)

        return ((pose,), control) # {"raw_image":raw_img, "image": img, "pose": pose, "control": control}


class ImagePoseControlDataset(Dataset):

    def __init__(self, images_by_demo, arm_joint_names, im_transform = None):
        self.images_by_demo = images_by_demo
        self.arm_joint_names = arm_joint_names
        self.im_transform = im_transform

        # We don't want sampler to pick last image of each demo
        # because the velocities are taken from the current_image_id + 1
        self.demo_strides = list(zip(images_by_demo, strides(images_by_demo, -1)))

    def __len__(self):
        return sum(len(demo) - 1 for demo in self.images_by_demo)

    def __getitem__(self, idx):
        demo, demo_stride = find_last(lambda ds: idx >= ds[1], self.demo_strides)
        img_id = idx - demo_stride
        np_pose, np_control = get_pose_and_control(demo, img_id, self.arm_joint_names)

        
        # Convert image format
        raw_img = cv2.imread(demo[img_id])[:, :, [2, 1, 0]] # BGR -> RGB Colour Indexing
        if self.im_transform is None:
            transformed_im = raw_img
        else:
            transformed_im = self.im_transform(raw_img)

        np_img = cv_to_nn_input(transformed_im)

        # Numpy -> Torch =  (Height x Width x Colour) ->  (Colour x Height x Width)
        img = torch.from_numpy(np_img.transpose(2, 0, 1)).to(dtype=torch.float)

        # Normalize joint angles by encoding with sin/cos
        wrapped_pose = np.arctan2(np.sin(np_pose), np.cos(np_pose))
        pose = torch.from_numpy(wrapped_pose).to(dtype=torch.float)


        control = torch.from_numpy(np_control).to(dtype=torch.float)

        return ((img, pose), control) # {"raw_image":raw_img, "image": img, "pose": pose, "control": control}


def get_pose_and_control(im_paths, idx, ordered_joints):

    folder_path, im_name = split(im_paths[idx])
    _, im_name_next = split(im_paths[idx + 1])

    timestamp_regex = re.compile(r'.*_(\d+\.\d+)\.jpg')

    # Note, was changed from having a name file for every time-stamp to a name file for just each demo
    # name_path = join(folder_path, timestamp_regex.sub(r'joint_names_\1.txt', im_name))
    name_path = glob(join(folder_path, "joint_names_*.txt"))[0]
    pose_path = join(folder_path, timestamp_regex.sub(r'joint_position_\1.txt', im_name))
    vel_path = join(folder_path, timestamp_regex.sub(r'joint_vel_\1.txt', im_name_next))


    names = np.genfromtxt(name_path, dtype=np.str)
    pose = np.genfromtxt(pose_path)
    vels = np.genfromtxt(vel_path)

    # Sometimes the joint names in each file might arrive out of order due to message passing
    # This line ensures joint velocities / poses correspond to joints in a fixed order
    joint_idxs = np.array([names.tolist().index(a) for a in ordered_joints])
    return pose[joint_idxs], vels[joint_idxs]


def cv_to_nn_input(img):
    scaled_img = img / 127.5 # Divide by half of full image range ([0, 255] -> [0, 2])
    return scaled_img - 1.0 # Translate by 1 [0, 2] -> [-1, 1]


def nn_input_to_imshow(nn_img):
    raw_im = np.transpose(nn_img.numpy(), (1, 2, 0))
    return (raw_im + 1.0) / 2.0


def strides(lists, lengths_offset):
    return np.cumsum([0] + [len(l) + lengths_offset for l in lists[:-1]])