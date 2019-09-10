import cv2
import torch
import numpy as np
from helper_funcs.utils_image import random_distort
from torchvision.transforms import Compose, Normalize


class Crop(object):
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return img[self.top:self.top + self.height, self.left:self.left + self.width, :]


class Resize(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        return cv2.resize(img, (self.width, self.height))


class ToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0).to(dtype=torch.float).permute((2,0,1))


class GrayToTensor(object):
    def __call__(self, img):
        if img.ndim == 2:
            np_img = np.expand_dims(img, 2)
        else:
            np_img = img

        return torch.from_numpy(np_img / 255.0).to(dtype=torch.float).permute((2,0,1))


class ColorJitter(object):
    def __call__(self, img):
        return random_distort(img)


def get_resnet_trans(im_params, distorted=False):
    ct, cl, ch, cw = im_params["crop_top"], im_params["crop_left"], im_params["crop_height"], im_params["crop_width"]
    transforms = [Crop(ct, cl, ch, cw), Resize(224, 224), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]

    if distorted:
        transforms = [ColorJitter()] + transforms
    
    return Compose(transforms)

def get_trans(im_params, distorted=False):
    ct, cl, ch, cw = im_params["crop_top"], im_params["crop_left"], im_params["crop_height"], im_params["crop_width"]
    rh, rw = im_params["resize_height"], im_params["resize_width"]

    transforms = [Crop(ct, cl, ch, cw), Resize(rh, rw), ToTensor()]
    if distorted:
        transforms = [ColorJitter()] + transforms

    return Compose(transforms)


def get_grayscale_trans(im_params):
    ct, cl, ch, cw = im_params["crop_top"], im_params["crop_left"], im_params["crop_height"], im_params["crop_width"]
    rh, rw = im_params["resize_height"], im_params["resize_width"]
    transforms = [Crop(ct, cl, ch, cw), Resize(rh, rw), GrayToTensor()]
    return Compose(transforms)