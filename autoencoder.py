import torch
import numpy as np
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from helper_funcs.utils import load_json, temp_print, t_stamp
from helper_funcs.transforms import get_trans
from load_data import load_demos, show_torched_im, nn_input_to_imshow, append_tensors_as_csv
import matplotlib.pyplot as plt
from os.path import join
import os
import pandas as pd


# Make an encoder module
class Encoder(nn.Module):
    def __init__(self, in_height, in_width, out_dim):
        super(Encoder, self).__init__()
        out_channels_first = 16
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=out_channels_first, kernel_size=4, stride=2, padding=1)
        self.cn2 = nn.Conv2d(in_channels=out_channels_first, out_channels=out_channels_first * 2, kernel_size=4, stride=2, padding=1)
        self.cn3 = nn.Conv2d(in_channels=out_channels_first * 2, out_channels=out_channels_first * 4, kernel_size=4, stride=2, padding=1)
        self.cn4 = nn.Conv2d(in_channels=out_channels_first * 4, out_channels=out_channels_first * 8, kernel_size=4, stride=2, padding=1)
        self.cn5 = nn.Conv2d(in_channels=out_channels_first * 8, out_channels=out_channels_first * 16, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.cn1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.cn2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.cn3.out_channels)
        self.bn4 = nn.BatchNorm2d(self.cn4.out_channels)
        self.bn5 = nn.BatchNorm2d(self.cn5.out_channels)

        o_height, o_width = out_size_cnns((in_height, in_width), [self.cn1, self.cn2, self.cn3, self.cn4, self.cn5])
        flattened_dim = o_height * o_width * self.cn5.out_channels

        self.c_out_h = o_height
        self.c_out_w = o_width
        self.c_out_c = self.cn5.out_channels
        

        self.means = nn.Linear(flattened_dim, out_dim)
        self.ln_vars = nn.Linear(flattened_dim, out_dim)
        # self.eps = Normal(torch.zeros(out_dim), torch.ones(out_dim))

    def forward(self, img_batch_in):
        batch_size = img_batch_in.shape[0]
        conv_out = F.leaky_relu(self.bn1(self.cn1(img_batch_in)), negative_slope=0.2)
        conv_out = F.leaky_relu(self.bn2(self.cn2(conv_out)), negative_slope=0.2)
        conv_out = F.leaky_relu(self.bn3(self.cn3(conv_out)), negative_slope=0.2)
        conv_out = F.leaky_relu(self.bn4(self.cn4(conv_out)), negative_slope=0.2)
        conv_out = F.leaky_relu(self.bn5(self.cn5(conv_out)), negative_slope=0.2)

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
    def __init__(self, z_dim, enc_out_h, enc_out_w):
        super(Decoder, self).__init__()
        # The out features are enc_out_channel_dims, encoder_out_height, encoder_out_width
        self.enc_out_h = enc_out_h
        self.enc_out_w = enc_out_w

        self.initial_channels = 128 * 16

        self.lin1 = nn.Linear(z_dim, self.initial_channels * 4)

        self.ctn1 = nn.ConvTranspose2d(in_channels=self.initial_channels, out_channels=self.initial_channels, kernel_size=4, stride=2, padding=1)
        self.ctn2 = nn.ConvTranspose2d(in_channels=self.initial_channels, out_channels=self.initial_channels // 2, kernel_size=4, stride=2, padding=1)
        self.ctn3 = nn.ConvTranspose2d(in_channels=self.initial_channels // 2, out_channels= self.initial_channels // 4, kernel_size=4, stride=2, padding=1)
        self.ctn4 = nn.ConvTranspose2d(in_channels=self.initial_channels // 4, out_channels=self.initial_channels // 8, kernel_size=4, stride=2, padding=1)
        self.ctn5 = nn.ConvTranspose2d(in_channels=self.initial_channels // 8, out_channels=self.initial_channels // 16, kernel_size=4, stride=2, padding=1)
        self.ctn6 = nn.ConvTranspose2d(in_channels=self.initial_channels // 16, out_channels=3, kernel_size=4, stride=2, padding=1)

        self.bn1 = nn.BatchNorm2d(self.ctn1.out_channels)
        self.bn2 = nn.BatchNorm2d(self.ctn2.out_channels)
        self.bn3 = nn.BatchNorm2d(self.ctn3.out_channels)
        self.bn4 = nn.BatchNorm2d(self.ctn4.out_channels)
        self.bn5 = nn.BatchNorm2d(self.ctn5.out_channels)
        self.bn6 = nn.BatchNorm2d(self.ctn6.out_channels)


    def forward(self, z_batch):
        flattened_h = self.lin1(z_batch)
        # print("Flattened shape", flattened_h.shape)
        reshaped_h = flattened_h.reshape((-1, self.initial_channels, 2, 2))
        # print(reshaped_h.shape)
        ct_out = F.relu(self.bn1(self.ctn1(reshaped_h)))
        ct_out = F.relu(self.bn2(self.ctn2(ct_out)))
        ct_out = F.relu(self.bn3(self.ctn3(ct_out)))
        ct_out = F.relu(self.bn4(self.ctn4(ct_out)))
        ct_out = F.relu(self.bn5(self.ctn5(ct_out)))
        ct_out = self.bn6(self.ctn6(ct_out))

        result = torch.sigmoid(ct_out)

        return result

class FFNet(nn.Module):
    def __init__(self, in_dims, hidden_dims, out_dims):
        super(FFNet, self).__init__()
        self.l1 = nn.Linear(in_dims, hidden_dims)
        self.l2 = nn.Linear(hidden_dims, hidden_dims)
        self.l3 = nn.Linear(hidden_dims, out_dims)

    def forward(self, in_batch):
        result = F.relu(self.l1(in_batch))
        result = F.relu(self.l2(result))
        result = self.l3(result)
        return result




class EncodeDecodePredictor(nn.Module):
    def __init__(self, in_height, in_width, z_dims, ff_h_dims, out_dims):
        super(EncodeDecodePredictor, self).__init__()
        self.encoder = Encoder(in_height, in_width, z_dims)
        self.decoder = Decoder(z_dims, self.encoder.c_out_h, self.encoder.c_out_w)
        self.ff_net = FFNet(z_dims, ff_h_dims, out_dims)
    

    def forward(self, im_batch):
        encoded_ims, mu, ln_var = self.encoder(im_batch)
        decoded = self.decoder(encoded_ims)
        prediction = self.ff_net(encoded_ims)
        return prediction, decoded, mu, ln_var


def full_loss(predicted_joints, target_joints, recon_im_batch, orig_im_batch, mu, log_var):
    # Mean squared error of joints
    pred_loss = F.mse_loss(predicted_joints, target_joints)
    combined_vae_loss, recon_loss, kl_to_prior = vae_loss(recon_im_batch, orig_im_batch, mu, log_var)
    full_loss = pred_loss + 1E-5 * combined_vae_loss
    return full_loss, pred_loss, combined_vae_loss, recon_loss, kl_to_prior


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

    

def plot_csv(csv_path, save_path=None, show_fig=False, col_subset=None):
    df = pd.read_csv(csv_path, sep=",")
    # print(training_df)
    # df.plot(subplots=True)

    if col_subset == None:
        display_cols = df.columns
    else:
        display_cols = col_subset

    for col in display_cols:
        plt.plot(df[col], label=col)

    # plt.plot(training_df.error, label="Results")
    # # plt.plot(validation_df.error, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path)

    if show_fig:
        plt.show()

    plt.close()

if __name__ == "__main__":
    exp_config = load_json("config/experiment_config.json")
    im_params = exp_config["image_config"]
    im_trans = get_trans(im_params, distorted=True)

    train_set, train_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        True,
        torch.device("cuda"),
        from_demo=0,
        to_demo=60,
        skip_count=5)

    # for i in range(12):
    #     plt.subplot(3,4,i + 1)
    #     plt.imshow(nn_input_to_imshow(train_set[i][0][0]))
    # plt.show()

    validation_set, validation_loader = load_demos(
        exp_config["demo_folder"],
        im_params["file_glob"],
        exp_config["batch_size"],
        exp_config["nn_joint_names"],
        im_trans,
        False,
        torch.device("cuda"),
        from_demo=60,
        to_demo=80,
        skip_count=5)

    IM_HEIGHT = 128
    IM_WIDTH = 128
    Z_DIMS = 512
    HIDDEN_LAYER = 100
    EPOCHS = 300
    print("Prediction Dims: {}".format(len(train_set[0][1])))

    vae_model = EncodeDecodePredictor(IM_HEIGHT, IM_WIDTH, Z_DIMS, HIDDEN_LAYER, len(train_set[0][1]))
    vae_model.to(torch.device("cuda"))

    optimizer = optim.Adam(vae_model.parameters())
    loss_criterion = full_loss
    #print(vae_model)


    results_folder = "logs/vae-test-{}".format(t_stamp())
    decoder_preview_folder = join(results_folder, "decoder-preview")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        os.makedirs(decoder_preview_folder)

    # df = pd.DataFrame(index=np.arange(0, EPOCHS), columns=["t-full", "t-recon", "t-kl", "v-full", "v-recon", "v-kl"])
    # print(df.columns)

    for epoch in range(EPOCHS):
        # print("Epoch: {}".format(epoch))
        vae_model.train()
        train_losses = []
        for i, in_batch in enumerate(train_loader):

            temp_print("T Batch {}/{}".format(i, len(train_loader)))
            (img_ins, _), target_joints = in_batch
            predicted_joints, decoded_ims, mu, ln_var = vae_model(img_ins)
            train_loss = loss_criterion(predicted_joints, target_joints, decoded_ims, img_ins, mu, ln_var)
            train_losses.append(train_loss)
            optimizer.zero_grad()
            train_loss[0].backward()
            optimizer.step()

        
        vae_model.eval()
        val_losses = []
        for i, in_batch in enumerate(validation_loader):
            temp_print("V Batch {}/{}".format(i, len(validation_loader)))
            with torch.no_grad():
                (img_ins, _), target_joints = in_batch
                predicted_joints, decoded_ims, mu, ln_var = vae_model(img_ins)
                val_loss = loss_criterion(predicted_joints, target_joints, decoded_ims, img_ins, mu, ln_var)
                val_losses.append(val_loss)

        t_loss_means = np.mean(train_losses, axis=0)
        v_loss_means = np.mean(val_losses, axis=0)

        # df.loc[epoch] = [train_full_loss.item(), train_recon_loss.item(), train_kl_loss.item(), val_full_loss.item(), val_recon_loss.item(), val_kl_loss.item()]
        # print(df.loc[epoch])
        print("{} T-Full: {}, T-MSE: {}, T-VAE: {} T-Recon: {}, T-KL: {}, V-Full {}, V-MSE {}, V-VAE: {}, V-Recon: {}, V-KL: {}"
            .format(epoch, t_loss_means[0], t_loss_means[1], t_loss_means[2], t_loss_means[3], t_loss_means[4],
                    v_loss_means[0], v_loss_means[1], v_loss_means[2], v_loss_means[3], v_loss_means[4]))
        metrics = ["T-Full", "T-MSE", "T-VAE", "T-Recon", "T-KL",
                "V-Full", "V-MSE", "V-VAE", "V-Recon", "V-KL"]
        append_tensors_as_csv(np.concatenate((t_loss_means, v_loss_means)),
        join(results_folder, "losses.csv"),
        cols = metrics)
        plot_csv(join(results_folder, "losses.csv"), join(results_folder, "losses.pdf"))

        if epoch % 10 == 0:
            torch.save(vae_model.state_dict(), join(results_folder, "learned_model_epoch_{}.pt".format(epoch)))

        preview_ids = [0, len(validation_loader) // 2, len(validation_loader) - 1]
        fig, axes = plt.subplots(len(preview_ids), 2)
        for i, p_id in enumerate(preview_ids):
            (preview_im, _), _ = validation_set[p_id]
            with torch.no_grad():
                decoded_preview = vae_model(preview_im.unsqueeze(0).to(torch.device("cuda")))[1].squeeze()
                axes[i,0].imshow(nn_input_to_imshow(preview_im))
                axes[i,1].imshow(nn_input_to_imshow(decoded_preview.detach()))

                axes[i,0].axis('off')
                axes[i,1].axis('off')
        plt.savefig(join(decoder_preview_folder, "decodedIm-epoch-{}".format(epoch)), bbox_inches=0)
        plt.close(fig)