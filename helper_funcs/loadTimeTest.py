import time
import torch
# from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter
from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize, ToTensor, ColorJitter
from load_data import load_demos
from helper_funcs.utils import load_json

exp_config = load_json("config/experiment_config.json")
im_params = exp_config["image_config"]

im_trans_1 = Compose([ColorJitter(),
                    Crop(im_params["crop_top"], im_params["crop_left"],
                            im_params["crop_height"], im_params["crop_width"]),
                    Resize(im_params["resize_height"],
                            im_params["resize_width"]),
                    ToTensor()])

im_trans_2 = Compose([ColorJitter(),
                    Crop(im_params["crop_top"], im_params["crop_left"],
                            im_params["crop_height"], im_params["crop_width"]),
                    Resize(im_params["resize_height"],
                           im_params["resize_width"])])


im_trans_3 = Compose([ColorJitter(),
                    Crop(im_params["crop_top"], im_params["crop_left"],
                            im_params["crop_height"], im_params["crop_width"])])


im_trans_4 = Compose([ColorJitter()])

im_trans_5 = Compose([
    Crop(im_params["crop_top"], im_params["crop_left"],
            im_params["crop_height"], im_params["crop_width"]),
    Resize(im_params["resize_height"],
           im_params["resize_width"]),
    ToTensor()
])

trans_options = [im_trans_1, im_trans_2, im_trans_3, im_trans_4, im_trans_5, None]

for i, im_trans in enumerate(trans_options):
    start = time.time()
    load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        True,
        torch.device("cuda"),
        from_demo=0,
        to_demo=1,
        skip_count=1)
    end = time.time()
    print("Transform {}: {}".format(i, end - start))
