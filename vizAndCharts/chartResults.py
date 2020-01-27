import numpy as np
import matplotlib.pyplot as plt
import sys
from os.path import join

def plot_train_val(train_path, val_path):
    train_losses = np.loadtxt(train_path)
    val_losses = np.loadtxt(val_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_metrics = train_losses.shape[1]
    for i in range(loss_metrics):
        plt.subplot(1, loss_metrics, i + 1)
        plt.plot(train_losses[:, i], label="train")
        plt.plot(val_losses[:, i], label="test")
        plt.legend()
    plt.show()

# Want Charting that compares
# validation results on constrained/unconstrained models
def plot_constrained_v_unconstrained(uncons_path, train_only_path, adv_path):
    cons_losses = np.loadtxt(cons_path)
    uncons_losses = np.loadtxt(uncons_path)
    adv_losses = np.loadtxt(adv_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_metrics = cons_losses.shape[1]
    loss_metric_names = ["Imitation", "Constraint"]
    for i in range(2):
        plt.subplot(1, len(loss_metric_names), i + 1)
        plt.gca().set_title(loss_metric_names[i])
        # plt.yscale("log")
        plt.plot(uncons_losses[:, i], label="Unconstrained")
        plt.plot(cons_losses[:, i], label="Train-Only")
        plt.plot(adv_losses[:, i], label="Adversarial" )
        if i == 0:
            plt.ylabel("Loss")
            plt.legend()
        plt.xlabel("Epochs")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logs_path = sys.argv[1]
    demo_type = sys.argv[2]
    uncons_path = join(logs_path, "{}-unconstrained/val_losses.txt".format(demo_type))
    cons_path = join(logs_path, "{}-train/val_losses.txt".format(demo_type))
    adv_path = join(logs_path, "{}-adversarial/val_losses.txt".format(demo_type))
    plot_constrained_v_unconstrained(uncons_path, cons_path, adv_path)