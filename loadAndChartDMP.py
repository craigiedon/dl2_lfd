import torch
from torch import nn, optim, autograd
from ltl_diff import constraints, oracle
import os
from os.path import join
from nns.dmp_nn import DMPNN
from dmps.dmp import load_dmp_demos, DMP
from helper_funcs.conversions import np_to_pgpu
from helper_funcs.utils import t_stamp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import sys

def chart_multidim_DMP(data_folder, model_folder):
    start_states, pose_hists = load_dmp_demos(data_folder)
    start_states, pose_hists = np_to_pgpu(start_states), np_to_pgpu(pose_hists)
    data_set = TensorDataset(start_states, pose_hists)
    basis_fs = 30
    dt = 0.01

    model = DMPNN(start_states[0].numel(), 1024, pose_hists.shape[2], basis_fs).cuda()
    model.load_state_dict(torch.load(join(model_folder, "learned_model_epoch_final.pt")))
    model.eval()

    i = 0
    inputs, rollout = data_set[i]
    T, dims = rollout.shape
    y0 = inputs[0].unsqueeze(0)
    goal = inputs[-1].unsqueeze(0)

    learned_weights = model(inputs.unsqueeze(0))
    # learned_weights = torch.zeros_like(learned_weights)

    dmp = DMP(basis_fs, dt, rollout.shape[1])
    learned_rollout = dmp.rollout_torch(y0, goal, learned_weights, 1.0)[0][0].detach().cpu()
    print(learned_rollout.shape)

    for d in range(dims):
        plt.subplot(3,3, d + 1)


        dmp_timescale = np.linspace(0, 1, rollout.shape[0])
        plt.scatter(dmp_timescale, rollout.detach().cpu()[:, d], label="Demo", c='orange', alpha=0.5)
        if(d < 3):
            plt.plot([0.0, 1.0], [inputs[2][d].cpu(), inputs[2][d].cpu()])
        plt.scatter(dmp_timescale, learned_rollout[:, d], label="DMP + LTL", alpha=0.5)

    plt.subplot(3,3,7)
    plt.scatter(rollout.detach().cpu()[:, 0], rollout.detach().cpu()[:, 1], label="Demo", c='orange', alpha=0.5)
    plt.scatter(learned_rollout[:, 0], learned_rollout[:, 1])
    # plt.scatter(inputs[1, 0].cpu(), inputs[1, 1].cpu())
    plt.scatter([0.5], [0.5])
    plt.gca().add_patch(plt.Circle([0.15, 0.5], radius=0.1, color="red", alpha=0.1))


    # plt.scatter(inputs[:, 0].detach().cpu(),
    #         inputs[:, 1].detach().cpu(),
    #         c='r', marker='x')

    # Steady Plotting Stuff
    # plt.plot([0.0, 1.0], [0.75, 0.75], c='r')
    # plt.plot([0.0, 1.0], [0.25, 0.25], c='r')

    # Avoid obstacle plotting stuff

    # plt.xlabel("X")
    # plt.xlim(0.0, 1.0)
    # plt.ylabel("Y")
    # plt.ylim(0.0, 1.0)
    # plt.legend(prop={"size": 14})
    plt.tight_layout()
    plt.show()

def chart_2D_DMP(data_folder, model_folder):
    start_states, pose_hists = load_dmp_demos(data_folder)
    start_states, pose_hists = np_to_pgpu(start_states), np_to_pgpu(pose_hists)
    data_set = TensorDataset(start_states, pose_hists)
    basis_fs = 30
    dt = 0.01

    print(pose_hists.shape)
    model = DMPNN(start_states[0].numel(), 1024, pose_hists.shape[2], basis_fs).cuda()
    model.load_state_dict(torch.load(join(model_folder, "learned_model_epoch_final.pt")))
    model.eval()

    for i in range(9):
        plt.subplot(3,3, i+1)
        inputs, rollout = data_set[i]
        y0 = inputs[0].unsqueeze(0)
        goal = inputs[-1].unsqueeze(0)

        learned_weights = model(inputs.unsqueeze(0))

        dmp = DMP(basis_fs, dt, rollout.shape[1])
        learned_rollout = dmp.rollout_torch(y0, goal, learned_weights, 1.0)[0][0].detach().cpu()

        dmp_timescale = np.linspace(0, 1, learned_rollout.shape[0])

        plt.scatter(rollout.detach().cpu()[:, 0], rollout.detach().cpu()[:, 1], label="Demo", c='orange', alpha=0.5)
        plt.scatter(learned_rollout[:, 0], learned_rollout[:, -1], label="DMP + LTL", alpha=0.5)

        plt.scatter(inputs[:, 0].detach().cpu(),
                inputs[:, 1].detach().cpu(),
                c='r', marker='x')

        # Steady Plotting Stuff
        plt.plot([0.0, 1.0], [0.75, 0.75], c='r')
        plt.plot([0.0, 1.0], [0.25, 0.25], c='r')

        # Avoid obstacle plotting stuff
        # plt.gca().add_patch(plt.Circle(inputs[1], radius=0.1, color="red", alpha=0.1))

    plt.xlabel("X")
    plt.xlim(0.0, 1.0)
    plt.ylabel("Y")
    plt.ylim(0.0, 1.0)
    plt.legend(prop={"size": 14})
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    data_folder = sys.argv[1]
    model_folder = sys.argv[2]
    # chart_2D_DMP(data_folder, model_folder)
    chart_multidim_DMP(data_folder, model_folder)