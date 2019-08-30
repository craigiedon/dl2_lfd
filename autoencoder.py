import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from helper_funcs.utils import load_json, temp_print, t_stamp
from torchvision.transforms import Compose
from helper_funcs.transforms import Crop, Resize
from load_data import load_demos, show_torched_im, nn_input_to_imshow
import matplotlib.pyplot as plt
from os.path import join
import os


# Make an encoder module
class Encoder(nn.Module):
    def __init__(self, in_height, in_width, out_dim):
        super(Encoder, self).__init__()
        out_channels_first = 64
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=out_channels_first, kernel_size=4, stride=2, padding=1)
        self.cn2 = nn.Conv2d(in_channels=out_channels_first, out_channels=out_channels_first // 2, kernel_size=4, stride=2, padding=1)
        self.cn3 = nn.Conv2d(in_channels=out_channels_first // 2, out_channels=out_channels_first // 4, kernel_size=4, stride=2, padding=1)

        #TODO: Put in some batch norm?
        o_height, o_width = out_size_cnns((in_height, in_width), [self.cn1, self.cn2, self.cn3])
        flattened_dim = o_height * o_width * self.cn3.out_channels

        self.c_out_h = o_height
        self.c_out_w = o_width
        self.c_out_c = self.cn3.out_channels
        

        self.means = nn.Linear(flattened_dim, out_dim)
        self.ln_vars = nn.Linear(flattened_dim, out_dim)
        # self.eps = Normal(torch.zeros(out_dim), torch.ones(out_dim))

    def forward(self, img_batch_in):
        batch_size = img_batch_in.shape[0]
        conv_out = F.relu(self.cn1(img_batch_in))
        conv_out = F.relu(self.cn2(conv_out))
        conv_out = F.relu(self.cn3(conv_out))

        flattened_im = torch.flatten(conv_out, 1)

        means = self.means(flattened_im)
        ln_vars = self.ln_vars(flattened_im)
        stds = torch.exp(ln_vars / 2) 
        # eps = self.eps.sample_n(batch_size)
        eps = torch.randn_like(means)

        # Reparameterization Trick
        z = means + eps * stds

        return z, means, ln_vars


class Decoder(nn.Module):
    def __init__(self, z_dim, enc_out_c, enc_out_h, enc_out_w):
        super(Decoder, self).__init__()
        # The out features are enc_out_channel_dims, encoder_out_height, encoder_out_width
        self.enc_out_h = enc_out_w
        self.enc_out_w = enc_out_w

        self.initial_channels = 128

        self.lin1 = nn.Linear(z_dim, self.initial_channels * 4)

        self.ctn1 = nn.ConvTranspose2d(in_channels=self.initial_channels, out_channels=self.initial_channels, kernel_size=4, stride=2, padding=1)
        self.ctn2 = nn.ConvTranspose2d(in_channels=self.initial_channels, out_channels=self.initial_channels // 2, kernel_size=4, stride=2, padding=1)
        self.ctn3 = nn.ConvTranspose2d(in_channels=self.initial_channels // 2, out_channels= self.initial_channels // 4, kernel_size=4, stride=2, padding=1)
        self.ctn4 = nn.ConvTranspose2d(in_channels=self.initial_channels // 4, out_channels=self.initial_channels // 8, kernel_size=4, stride=2, padding=1)
        self.ctn5 = nn.ConvTranspose2d(in_channels=self.initial_channels // 8, out_channels=self.initial_channels // 16, kernel_size=4, stride=2, padding=1)
        self.ctn6 = nn.ConvTranspose2d(in_channels=self.initial_channels // 16, out_channels=3, kernel_size=4, stride=2, padding=1)


    def forward(self, z_batch):
        flattened_h = self.lin1(z_batch)
        # print("Flattened shape", flattened_h.shape)
        reshaped_h = flattened_h.reshape((-1, self.initial_channels, 2, 2))
        # print(reshaped_h.shape)
        ct_out = self.ctn1(reshaped_h)
        ct_out = self.ctn2(ct_out)
        ct_out = self.ctn3(ct_out)
        ct_out = self.ctn4(ct_out)
        ct_out = self.ctn5(ct_out)
        ct_out = self.ctn6(ct_out)

        result = torch.sigmoid(ct_out)

        return result


class EncoderDecoder(nn.Module):
    def __init__(self, in_height, in_width, z_dims):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(in_height, in_width, z_dims)
        self.decoder = Decoder(z_dims, self.encoder.c_out_c, self.encoder.c_out_h, self.encoder.c_out_w)
    

    def forward(self, im_batch):
        encoded_ims, mu, ln_var = self.encoder(im_batch)
        decoded = self.decoder(encoded_ims)
        return decoded, mu, ln_var


def vae_loss(recon_im_batch, orig_im_batch, mu, log_var):
    # E_q [log P(X | z)]
    # Why the binary CE? Think of the decoder as defining bernoulli parameters over a randomly binarized version of the image.
    # So the original image is sort of like an expected statistic over these binarized images.
    recon_loss = F.binary_cross_entropy(recon_im_batch, orig_im_batch)
    # KL(Q(z | X) || P(z))
    kl_to_prior = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + kl_to_prior), recon_loss, kl_to_prior


def out_size_cnns(img_dims, cnns):
    current_dims = img_dims
    for cnn in cnns:
        current_dims = output_size(current_dims[0], current_dims[1], cnn.kernel_size[0], cnn.stride[0], cnn.padding[0])
    return current_dims

def output_size(in_height, in_width, kernel_size, stride=1, padding=0):
    out_height = int((in_height - kernel_size + padding * 2) / stride) + 1
    out_width = int((in_width - kernel_size + padding * 2) / stride) + 1
    return (out_height, out_width)


exp_config = load_json("config/experiment_config.json")
im_params = exp_config["image_config"]
im_trans = Compose([
    Crop(im_params["crop_top"], im_params["crop_left"],
         im_params["crop_height"], im_params["crop_width"]),
    Resize(im_params["resize_height"], im_params["resize_width"])])

train_set, train_loader = load_demos(
    exp_config["demo_folder"],
    im_params["file_glob"],
    exp_config["batch_size"],
    exp_config["nn_joint_names"],
    im_trans,
    True,
    torch.device("cuda"),
    from_demo=0,
    to_demo=1)


IM_HEIGHT = 128
IM_WIDTH = 128
Z_DIMS = 512
EPOCHS = 300
vae_model = EncoderDecoder(IM_HEIGHT, IM_WIDTH, Z_DIMS)
vae_model.to(torch.device("cuda"))

optimizer = optim.Adam(vae_model.parameters())
#TODO: Does this loss function need a "requires gradient" line anywhere?
loss_criterion = vae_loss
print(vae_model)


results_folder = "logs/vae-test-{}".format(t_stamp())
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

for epoch in range(EPOCHS):
    # print("Epoch: {}".format(epoch))
    for i, in_batch in enumerate(train_loader):

        temp_print("Batch {}/{}".format(i, len(train_loader)))
        (img_ins, _), _ = in_batch
        decoded_ims, mu, ln_var = vae_model(img_ins)
        full_loss, recon_loss, kl_loss = loss_criterion(decoded_ims, img_ins, mu, ln_var)
        optimizer.zero_grad()
        full_loss.backward()
        optimizer.step()
        # print("Batch {}".format(i))

        # print("Full Loss: {}, Recon: {}, KL-part: {}".format(full_loss, recon_loss, kl_loss))

    print("For Epoch {}, Full Loss: {}, Recon: {}, KL-part: {} ".format(epoch, full_loss, recon_loss, kl_loss))
    # if epoch % 10 == 0:
    _, (orig_ax, decoded_ax) = plt.subplots(1, 2)
    orig_ax.imshow(nn_input_to_imshow(img_ins[0]))
    decoded_ax.imshow(nn_input_to_imshow(decoded_ims[0].detach()))
    plt.savefig(join(results_folder, "decodedIm-epoch-{}".format(epoch)))
    ## Write function to decode images, display them etc. using cv2 (use daniel's code for this if there is some? If not should be not too hard)

## Implements skip count function, image contrasts augmentation etc.