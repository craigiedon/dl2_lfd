import torch
import numpy as np
from torchvision.transforms import Compose
from torchvision.transforms.functional import resize, to_pil_image
from helper_funcs.transforms import Crop, Resize
from glob import glob
import os
from os.path import join
from cnns import ImageAndJointsNet, ImagePoseControlDataset
import matplotlib.pylab as plt
from matplotlib import animation
import cv2
import numpy as np


def grad_cam_heatmap(model, device, in_data):
    model.eval()
    img_in, pose_in = in_data["image"].to(device), in_data["pose"].to(device)

    # zero the gradients
    model.zero_grad()

    outputs = model(img_in.unsqueeze(0), pose_in.unsqueeze(0))

    # Back prop the gradients through (maybe this needs to be done on the parameters themselves?)
    # Can you back-propagate multiple gradients at once though
    # intermediate output.register_hook(logging_function)
    conv_outs = outputs["conv3_out"]
    conv_gradients = torch.zeros(conv_outs.size()).to(device)

    def conv_grad_hook(grad, out_tensor):
        print("Hook activated!")
        print("Grads Size:", grad.size())
        out_tensor[0] = grad[0]

    
    hook = conv_outs.register_hook(lambda x: conv_grad_hook(x, conv_gradients))
    control_out = outputs["output"][0]
    print("Controls out size:", control_out.size())

    size_vec = torch.ones(control_out.size()).to(device)
    control_out.backward(size_vec)
    hook.remove()

    print("Outer conv grads size", conv_gradients.size())
    print("Conv outs:", conv_outs.size())

    # Importance weights: sum the gradients across height and width for each channel
    conv_gradients = normalize(conv_gradients)
    importance_weights = torch.mean(conv_gradients, (2,3))
    if torch.sum(importance_weights) == 0:
        print("Zero'd gradients!")

    print("Channel Importance Weights: ", importance_weights)

    # Sum the 3 channels together, weighted by their gradient importance
    weighted_channels = importance_weights.reshape(3,1,1) * conv_outs[0]

    # Unlike the ReLU used in Grad-CAM, we instead normalize and take absolute
    # because a large negative value is actually important for regression
    grad_scores = torch.sum(weighted_channels, 0)
    heatmap = torch.abs(grad_scores - torch.mean(grad_scores))
    if torch.sum(heatmap) != 0.0:
        heatmap = heatmap / torch.max(heatmap)

    
    print("Grad Scores Size:", grad_scores.size())
    heatmap = cv2.resize(heatmap.detach().cpu().numpy(), img_in.shape[1:])
    heatmap = cv2.GaussianBlur(heatmap, (15,15), 5)
    return heatmap


def normalize(x):
    return x / (torch.sqrt(torch.mean(torch.pow(x, 2))) + 1e-5)


def heatmap_animation(model, device, demo_dataset):
    fig = plt.figure(figsize=(15,5))
    imgs = [d["image"].permute(1,2,0) for d in demo_dataset]
    im_canvas = plt.imshow(imgs[0])
    heatmap_canvas = plt.imshow(grad_cam_heatmap(model, device, demo_dataset[1]), cmap="jet", alpha=0.5)

    def anim_step(i):
        heatmap = grad_cam_heatmap(model, device, demo_dataset[i])
        im_canvas.set_data(imgs[i])
        heatmap_canvas.set_data(heatmap)
        return [im_canvas, heatmap_canvas]

    ani = animation.FuncAnimation(fig, anim_step, interval=200, frames=len(demo_dataset), blit=True, repeat=False)
    plt.show()
    return ani




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_model = ImageAndJointsNet(224, 224, 8)
full_model.load_state_dict(torch.load("./logs/e2e_control_full.pt"))
full_model.to(device)


demos_train_root = "./demos/train"

train_demo_paths = [join(demos_train_root, d) for d in os.listdir(demos_train_root)]

arm_joint_names = np.genfromtxt("./arm_joint_names.txt", np.str)
train_demos = [sorted(glob(join(demo_path, "kinect_colour_*.jpg"))) for demo_path in train_demo_paths]
im_trans = Compose([Crop(115, 300, 0, 450), Resize(224, 224)])
train_set = ImagePoseControlDataset([train_demos[0]], arm_joint_names, im_trans)

# plt.imshow(grad_cam_heatmap(full_model, device, train_set[3]), cmap="jet", alpha=0.5)
# plt.show()

heatmap_animation(full_model, device, train_set)