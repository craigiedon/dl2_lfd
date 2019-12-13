import numpy as np
import matplotlib.pyplot as plt

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
def plot_constrained_v_unconstrained(cons_path, uncons_path):
    cons_losses = np.loadtxt(cons_path)
    uncons_losses = np.loadtxt(uncons_path)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    loss_metrics = cons_losses.shape[1]
    for i in range(loss_metrics):
        plt.subplot(1, loss_metrics, i + 1)
        plt.plot(cons_losses[:, i], label="Constrained")
        plt.plot(uncons_losses[:, i], label="Unconstrained")
        plt.legend()
    plt.show()