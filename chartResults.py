import pandas as pd
import matplotlib.pyplot as plt
import sys
from math import ceil
from model import load_model
from load_data import load_demos
import numpy as np
import torch

from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize

def chart_train_validation_error(train_results_path, validation_results_path):
    training_df = pd.read_csv(train_results_path, sep=" ", header=None, names=["error"])
    print(training_df)
    validation_df = pd.read_csv(validation_results_path, sep=" ", header=None, names=["error"])
    plt.plot(training_df.error, label="Train")
    plt.plot(validation_df.error, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.show()


def chart_demo_predictions(model_path, demo_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, device)
    joint_names = np.genfromtxt("config/arm_joint_names.txt", np.str)
    im_trans = Compose([Crop(115, 300, 0, 450), Resize(224, 224)])
    _, demo_loader = load_demos(demo_path, 32, joint_names, im_trans, True, device, to_demo=1)

    model.eval()
    ests = []
    trues = []
    with torch.no_grad():
        for (ins, true_controls) in demo_loader:
            est_controls = model(*ins)
            est_controls = [x.cpu().numpy() for x in est_controls]
            true_controls = [x.cpu().numpy() for x in true_controls]

            ests.extend(est_controls)
            trues.extend(true_controls)

    ests = np.array(ests).transpose()
    trues = np.array(trues).transpose()

    fig = plt.figure()


    for i in range(len(ests)):
        c_ax = fig.add_subplot(ceil(len(ests) / 3.0), 3, i + 1)
        c_ax.plot(ests[i], label="Estimated Vels")
        c_ax.plot(trues[i], label="True Vels")
        c_ax.legend()
        c_ax.title.set_text(joint_names[i])
        c_ax.set_xlabel("t")
        c_ax.set_ylabel("Velocity")
    
    plt.show()
    
    print(ests.shape)
    """
    fig = plt.figure()
    est_ax = fig.add_subplot(1,1,1)
    true_ax = fig.add_subplot(1,2,1)
    est_ax.plot(ests)
    true_ax.plot(trues)

    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()

    plt.show()
    """
    

    

# Run it through the device, get the results, and display them. This should already be in the animation thing
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: chartResults.py <folder-path>")
        sys.exit(0)

    log_path = sys.argv[1]
    chart_train_validation_error("{}/train.txt".format(log_path),
                                 "{}/validation.txt".format(log_path))
