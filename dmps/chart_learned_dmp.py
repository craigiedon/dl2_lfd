import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
from scipy.interpolate import interp1d
from dmps.dmp import DMP, imitate_path, load_dmp, save_dmp
from dmps.fit_dmp_nn import dmp_nn
from helper_funcs.utils import t_stamp
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join
from glob import glob
from math import ceil
import pickle
from scipy.spatial.transform import Rotation as R

# Load up the dmp nn
# torch.load
# model = ImagePlusPoseNet((im_params["resize_height"], im_params["resize_width"]), 100)
model_path = "logs/synth-wave-2019-10-23-13-46-52/learned_model_epoch_250.pt"
model = dmp_nn(2, 100, 30)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda")))
model.to(torch.device("cuda"))
model.eval()
# Load in the single dim pose history
# Create a dmp with weights by running the start and end through the trained NN
# plot the rolled-out DMP versus a few of the actual rollouts