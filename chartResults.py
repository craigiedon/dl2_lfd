import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import sys
from math import ceil
from model import load_model
from load_data import load_demos, nn_input_to_imshow, show_torched_im
import numpy as np
import torch
from helper_funcs.utils import zip_chunks, load_json, load_json_lines

from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize

"""
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
"""

def chart_train_validation_error(train_path, val_path):
    # Load in the states from both as json lists
    training_stats = load_json_lines(train_path)
    val_stats = load_json_lines(val_path)

    plt.plot(training_stats["training_loss"], label="Train")
    plt.plot(val_stats["training_loss"], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Average MSE")
    plt.legend()
    plt.show()


def animate_spatial_features(model_path, demos_folder, demo_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_names = np.genfromtxt("config/arm_joint_names.txt", np.str)

    # TODO: These need to come out as config i reckon...
    im_h, im_w = (240, 240)
    crop_t, crop_b, crop_l, crop_r = (150, 475, 50, 800)

    model = load_model(model_path, device, im_h, im_w, joint_names)
    im_trans = Compose([Crop(crop_t, crop_b, crop_l, crop_r), Resize(im_h, im_w)])

    demo_ds, demo_loader = load_demos(demos_folder, "kinect2_qhd_image_color_rect_*.jpg", 32, joint_names, im_trans, False, device, from_demo=demo_num, to_demo=demo_num + 1)

    model.eval()

    # Setup subplots and axes
    fig = plt.figure(figsize=(15,5))
    img_ax = fig.add_subplot(1,1,1)

    resizer = Resize(model.cout_h, model.cout_w)
    resized_im = resizer(nn_input_to_imshow(demo_ds[0][0][0]))
    im_canvas = img_ax.imshow(resized_im)

    feature_points = []
    display_ims = []

    with torch.no_grad():
        # For each image, pass it through, then plot it (except its not plotting it, it will be animating it...)
        for (ins, _) in demo_loader:
            img_ins, _ = ins
            model(*ins)
            fp_batch = model.aux_outputs["feature_points"]
            fp_batch = zip_chunks(fp_batch, 2, dim=1)

            # Append the current feature points to list
            feature_points.extend(fp_batch.tolist())
            # Append the current image to a list
            # TODO: Resize to the conv_out size, or just show the conv_outs!
            display_batch = [nn_input_to_imshow(x) for x in img_ins]
            display_batch = [resizer(x) for x in display_batch]
            display_ims.extend(display_batch)
            

    def anim_step(i):
        del img_ax.patches[:]
        ps = []
        for (fx, fy) in feature_points[i]:
            ps.append(img_ax.add_patch(patches.Circle((fx, fy), radius=1, color="red")))
        im_canvas.set_data(display_ims[i])
        return [im_canvas] + ps

    ani = animation.FuncAnimation(fig, anim_step, interval=200, frames=len(display_ims), blit=True, repeat=True)
    plt.show()
    return ani
            

def chart_demo_predictions(model_path, demo_path, demo_num):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_config = load_json("config/experiment_config.json")
    # TODO: This is in multiple places, so I think it needs to be config
    im_params = exp_config["image_config"]

    im_trans = Compose([Crop(im_params["crop_top"], im_params["crop_left"],
                             im_params["crop_height"], im_params["crop_width"]),
                        Resize(im_params["resize_height"], im_params["resize_width"])])
    model = load_model(model_path, device, im_params["resize_height"], im_params["resize_width"], exp_config["nn_joint_names"])

    demo_set, demo_loader = load_demos(
        demo_path,
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        False,
        device,
        from_demo=demo_num,
        to_demo=demo_num + 1
    )

    show_torched_im(demo_set[0][0][0])

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


    for i, _ in enumerate(ests):
        c_ax = fig.add_subplot(ceil(len(ests) / 3.0), 3, i + 1)
        c_ax.plot(ests[i], label="Estimated Vels")
        c_ax.plot(trues[i], label="True Vels")
        c_ax.legend()
        c_ax.title.set_text(exp_config["nn_joint_names"][i])
        c_ax.set_xlabel("t")
        c_ax.set_ylabel("Velocity")
    
    plt.show()
    

# Run it through the device, get the results, and display them. This should already be in the animation thing
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: chartResults.py <results-path> <demos-folder> <demo-num>")
        sys.exit(0)

    log_path = sys.argv[1]
    demos_folder = sys.argv[2]
    demo_num = int(sys.argv[3])
    model_path = "{}/e2e_control_full.pt".format(log_path)

    chart_train_validation_error("{}/train.txt".format(log_path),
                                 "{}/validation.txt".format(log_path))

    """
    for i in [43, 82, 101]:
        print(i)
        chart_demo_predictions(model_path, demos_folder, i)
    # animate_spatial_features(model_path, demos_folder, demo_num)
    """
