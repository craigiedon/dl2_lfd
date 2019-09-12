import torch
import numpy as np
import os
from os.path import join, split
import re
import cv2
import json
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from helper_funcs.utils import find_last
from glob import glob
from scipy.spatial.transform import Rotation as R

class PoseAndGoal(Dataset):
    def __init__(self, images_by_demo, active_ee_name, goal_ee_name, skip_count=5):
        self.images_by_demo = images_by_demo
        self.active_ee_name = active_ee_name
        self.goal_ee_name = goal_ee_name

        self.data_points = []
        self.skip_count = skip_count

        num_demos = len(images_by_demo)
        for demo_id in range(num_demos):
            for img_id in range(len(images_by_demo[demo_id]) - skip_count):
                self.data_points.append(self.load_data_point(demo_id, img_id))

    def __len__(self):
        return len(self.data_points)

    
    def __getitem__(self, idx):
        return self.data_points[idx]

    def load_data_point(self, demo_id, img_id):
        current_path = self.images_by_demo[demo_id][img_id]        
        next_path = self.images_by_demo[demo_id][img_id + self.skip_count]

        current_pose_q = get_pose(current_path, self.active_ee_name)
        goal_pose_q = get_pose(current_path, self.goal_ee_name)
        next_pose_q = get_pose(next_path, self.active_ee_name)

        current_pose_rpy = quat_pose_to_rpy(current_pose_q)
        goal_pose_rpy = quat_pose_to_rpy(goal_pose_q)
        next_pose_rpy = quat_pose_to_rpy(next_pose_q)

        current_pose = torch.from_numpy(current_pose_rpy).to(dtype=torch.float)
        goal_pose = torch.from_numpy(goal_pose_rpy).to(dtype=torch.float)
        next_pose = torch.from_numpy(next_pose_rpy).to(dtype=torch.float)

        return current_pose, goal_pose, next_pose

def quat_pose_to_rpy(quat_pose):
    pos, quat = quat_pose[0:3], quat_pose[3:]
    rpy = R.from_quat(quat).as_euler("xyz") / np.pi
    return np.concatenate((pos, rpy))
    

class ImageRGBDPoseHist(Dataset):

    def __init__(self,
                 images_by_demo,
                 ee_name,
                 rgb_trans=None,
                 depth_trans=None):
        # self.images_by_demo = images_by_demo
        self.images_by_demo = images_by_demo
        self.ee_name = ee_name
        self.rgb_trans = rgb_trans
        self.depth_trans = depth_trans

        print("Loading {} files".format(len(self.images_by_demo)))

        # We don't want the final image, because we need to predict future
        # And we dont want first 4, because we are using past history as an input
        self.data_points = []
        for demo_id in range(len(images_by_demo)):
            for img_id in range(5, len(images_by_demo[demo_id]) - 1):
                self.data_points.append(self.load_data_point(demo_id, img_id))

    def __len__(self):
        return len(self.data_points)

    def __getitem__(self, idx):
        return self.data_points[idx]

    def load_data_point(self, demo_id, img_id):

        past_im_paths = self.images_by_demo[demo_id][img_id - 4: img_id + 1]
        next_im_path = self.images_by_demo[demo_id][img_id + 1]

        img_path = self.images_by_demo[demo_id][img_id]
        depth_path = img_path.replace("_color_", "_depth_")
        raw_rgb = cv2.imread(img_path)
        raw_depth = np.expand_dims(cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE) , 2)

        if self.rgb_trans is not None:
            img_rgb = self.rgb_trans(raw_rgb)
        else:
            img_rgb = raw_rgb

        if self.depth_trans is not None:
            img_depth = self.depth_trans(raw_depth)
        else:
            img_depth = raw_depth

        past_poses_np = get_pose_hist(past_im_paths, self.ee_name)
        next_pose_np = get_pose_hist([next_im_path], self.ee_name)[0]

        # Convert to roll-pitch-yaw and normalize between -1 and 1
        next_pose_pos, next_pose_quat = next_pose_np[0:3], next_pose_np[3:]
        next_pose_rpy = R.from_quat(next_pose_quat).as_euler("xyz") / np.pi
        
        past_poses_pos, past_poses_quat = past_poses_np[:, 0:3], past_poses_np[:, 3:]
        past_poses_rpy = R.from_quat(past_poses_quat).as_euler("xyz") / np.pi

        next_pose_pos_rpy = np.concatenate((next_pose_pos, next_pose_rpy))
        past_poses_pos_rpy = np.concatenate((past_poses_pos, past_poses_rpy), 1)
        

        past_poses = torch.from_numpy(past_poses_pos_rpy).to(dtype=torch.float)
        next_pose = torch.from_numpy(next_pose_pos_rpy).to(dtype=torch.float)

        return img_rgb, img_depth, past_poses, next_pose

def get_pose_hist(img_paths, ee_name):
    return np.stack([get_pose(p, ee_name) for p in img_paths])

def get_pose(img_path, ee_name):
    folder_path, img_name = split(img_path)
    ts_reg = re.compile(r'.*_(\d+)\.jpg')
    repl_form = r'{}_\1.txt'.format(ee_name)
    pose_path = join(folder_path, ts_reg.sub(repl_form, img_name))
    return np.genfromtxt(pose_path)


def image_demo_paths(demos_root, image_glob, from_demo=None, to_demo=None, frame_limit=None):
    demo_paths = [join(demos_root, d) for d in sorted(os.listdir(demos_root))]
    demo_images = [sorted(glob(join(demo_path, image_glob))) for demo_path in demo_paths]
    return [d[:frame_limit] for d in demo_images[from_demo:to_demo]]


def load_demos(demos_folder, image_glob, batch_size, joint_names,
               im_trans, shuffled, device, from_demo=None, to_demo=None,
               frame_limit=None, skip_count=1):
    demo_paths = image_demo_paths(demos_folder, image_glob)
    print(demo_paths[from_demo][0])

    demos = [d[0:frame_limit] for d in demo_paths[from_demo:to_demo]]
    print("Demo Length - Raw: {}, Processed {}".format(len(demo_paths[from_demo]), [len(demos[0])]))


    d_set = ImagePoseFuturePoseDataSet(demos, joint_names, skip_count, im_trans)
    d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

    return d_set, d_loader

def load_rgbd_demos(demo_paths, batch_size, ee_name, rgb_trans, depth_trans, shuffled, device):
    d_set = ImageRGBDPoseHist(demo_paths, ee_name, rgb_trans, depth_trans)
    d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

    return d_set, d_loader

def load_pose_state_demos(demo_paths, batch_size, active_ee, goal_ee, shuffled, device):
    print("Loading {} demo paths".format(len(demo_paths)))
    d_set = PoseAndGoal(demo_paths, active_ee, goal_ee)
    d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

    return d_set, d_loader



def load_constant_joint_vals(demos_root, constant_joint_names):
    demo_folders = [join(demos_root, d) for d in os.listdir(demos_root)]
    name_path = glob(join(demo_folders[0], "joint_names_*.txt"))[0]
    pose_path = glob(join(demo_folders[0], "joint_position_*.txt"))[0]

    names = np.genfromtxt(name_path, dtype=np.str).tolist()
    pose = np.genfromtxt(pose_path)

    joint_ids = np.array([names.index(a) for a in constant_joint_names])
    relevant_poses = pose[joint_ids]
    return {n:p for n, p in zip(constant_joint_names, relevant_poses)}
    

# def load_pose_demos(demos_folder, image_glob, batch_size, joint_names, shuffled, device, from_demo=None, to_demo=None, frame_limit=None):
#     demo_paths = image_demo_paths(demos_folder, image_glob)
#     demos = [d[0:frame_limit] for d in demo_paths[from_demo:to_demo]]

#     d_set = PoseControlDataset(demos, joint_names)
#     d_loader = DeviceDataLoader(DataLoader(d_set, batch_size, shuffle=shuffled), device)

#     return d_set, d_loader


def save_dict_append(d, file_path):
    with open(file_path, "a") as save_file:
        save_file.write("{}\n".format(json.dumps(d)))


def save_list_append(xs, file_path):
    with open(file_path, "a") as save_file:
        for x in xs:
            save_file.write("{}\n".format(x))

def append_tensors_as_csv(nums, file_path, cols=None):
    # Add a column reference at top if required
    if not os.path.exists(file_path) and cols is not None:
        with open(file_path, 'a') as save_file:
            cols_csv = ",".join(cols)
            save_file.write("{}\n".format(cols_csv))

    csv_line = ",".join([str(x.item()) for x in nums])
    with open(file_path, 'a') as save_file:
        save_file.write("{}\n".format(csv_line))


def save_list_nums(lines, file_path):
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
        wrapped_pose = wrap_pose(np_pose)
        pose = torch.from_numpy(wrapped_pose).to(dtype=torch.float)


        control = torch.from_numpy(np_control).to(dtype=torch.float)

        return ((img, pose), control) # {"raw_image":raw_img, "image": img, "pose": pose, "control": control}


class ImagePoseFuturePoseDataSet(Dataset):

    def __init__(self, images_by_demo, arm_joint_names, skip_count=1, im_transform=None):
        # self.images_by_demo = images_by_demo
        self.images_by_demo = images_by_demo
        self.arm_joint_names = arm_joint_names
        self.im_transform = im_transform
        self.skip_count = skip_count

        # We don't want sampler to pick last x images of each demo
        # because we are predicting positions on current instance + x
        # self.skipped_demos = [d[::self.skip_count] for d in images_by_demo]

        # print("Full lengths {}, Skipped Lengths: {}".format([len(d) for d in images_by_demo], [len(d) for d in self.skipped_demos]))
        self.demo_strides = list(zip(self.images_by_demo, strides(self.images_by_demo, -skip_count)))
        print("Loading {} files".format(len(self)))
        self.data_points = [self.load_data_point(i) for i in range(len(self))]

    def __len__(self):
        return sum(len(demo) - self.skip_count for demo in self.images_by_demo)

    def __getitem__(self, idx):
        return self.data_points[idx]

    def load_data_point(self, idx):
        demo, demo_stride = find_last(lambda ds: idx >= ds[1], self.demo_strides)
        img_id = idx - demo_stride

        np_pose, np_next_pose = get_pose_next_pose(demo, img_id, self.arm_joint_names, self.skip_count)

        
        raw_img = cv2.imread(demo[img_id])

        if self.im_transform is None:
            transformed_im = raw_img
        else:
            transformed_im = self.im_transform(raw_img)

        # Normalize joint angles by encoding with sin/cos
        wrapped_pose = wrap_pose(np_pose)
        pose = torch.from_numpy(wrapped_pose).to(dtype=torch.float)
        # pose = torch.from_numpy(np_pose).to(dtype=torch.float)

        wrapped_next_pose = wrap_pose(np_next_pose)
        next_pose = torch.from_numpy(wrapped_next_pose).to(dtype=torch.float)
        # next_pose = torch.from_numpy(np_next_pose).to(dtype=torch.float)

        return ((transformed_im, pose), next_pose) # {"raw_image":raw_img, "image": img, "pose": pose, "control": control}

def wrap_pose(unbounded_rads):
    pi_bounded_rads = np.arctan2(np.sin(unbounded_rads), np.cos(unbounded_rads)) # bound between [-pi, pi]
    return (pi_bounded_rads + np.pi) / (np.pi * 2.0) # bound between [0, 1]

def unnorm_pose(normed_rads):
    return normed_rads * (np.pi * 2.0) - np.pi


def get_pose_and_control(im_paths, idx, ordered_joints):

    folder_path, im_name = split(im_paths[idx])
    _, im_name_next = split(im_paths[idx + 1])

    timestamp_regex = re.compile(r'.*_(\d+)\.jpg')

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


def get_pose_next_pose(im_paths, idx, ordered_joints, pred_interval):
    folder_path, im_name = split(im_paths[idx])
    _, im_name_next = split(im_paths[idx + pred_interval])

    timestamp_regex = re.compile(r'.*_(\d+)\.jpg')

    name_path = glob(join(folder_path, "joint_names_*.txt"))[0]
    pose_path = join(folder_path, timestamp_regex.sub(r'joint_position_\1.txt', im_name))
    next_pose_path = join(folder_path, timestamp_regex.sub(r'joint_position_\1.txt', im_name_next))

    names = np.genfromtxt(name_path, dtype=np.str)
    pose = np.genfromtxt(pose_path)
    next_pose = np.genfromtxt(next_pose_path)

    # Sometimes the joint names in each file might arrive out of order due to message passing
    # This line ensures joint poses correspond to joints in a fixed order
    joint_idxs = np.array([names.tolist().index(a) for a in ordered_joints])
    return pose[joint_idxs], next_pose[joint_idxs]


def cv_to_nn_input(img):
    # scaled_img = img / 127.5 # Divide by half of full image range ([0, 255] -> [0, 2])
    # offset_img = scaled_img - 1.0 # Translate by 1 [0, 2] -> [-1, 1]
    # return offset_img
    return img / 255.0


def nn_input_to_imshow(nn_img):
    #raw_im = np.transpose(nn_img.cpu().numpy(), (1, 2, 0))
    #return (raw_im + 1.0) / 2.0
    return np.transpose(nn_img.cpu().numpy(), (1,2,0))


def strides(lists, lengths_offset):
    return np.cumsum([0] + [len(l) + lengths_offset for l in lists[:-1]])


def show_torched_im(torched_im):
    processed_im = nn_input_to_imshow(torched_im)
    plt.imshow(processed_im)
    plt.show()
