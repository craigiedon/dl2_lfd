from torchvision.transforms import Compose, Resize, ToTensor, ColorJitter

class Crop(object):
    def __init__(self, top, left, height, width):
        self.top = top
        self.left = left
        self.height = height
        self.width = width

    def __call__(self, img):
        return img.crop((self.left, self.top, self.left + self.width, self.top + self.height))


def get_trans(im_params, distorted=False):
    if distorted:
        return Compose([ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                        Crop(im_params["crop_top"], im_params["crop_left"],
                             im_params["crop_height"], im_params["crop_width"]),
                        Resize((im_params["resize_height"],
                                im_params["resize_width"])),
                        ToTensor()])
    return Compose([Crop(im_params["crop_top"], im_params["crop_left"],
                         im_params["crop_height"], im_params["crop_width"]),
                    Resize((im_params["resize_height"],
                            im_params["resize_width"])),
                    ToTensor()])
