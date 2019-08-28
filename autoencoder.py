import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

# Make an encoder module
class Encoder(nn.Module):
    def __init__(self, in_height, in_width, out_dim):
        super(Encoder, self).__init__()
        out_channels_first = 16
        self.cn1 = nn.Conv2d(in_channels=3, out_channels=out_channels_first, kernel_size=4, stride=2, padding=1)
        self.cn2 = nn.Conv2d(in_channels=out_channels_first, out_channels=out_channels_first * 2, kernel_size=4, stride=2, padding=1)
        self.cn3 = nn.Conv2d(in_channels=out_channels_first * 2, out_channels=out_channels_first * 4, kernel_size=4, stride=2, padding=1)

        #TODO: Put in some batch norm?
        o_height, o_width = out_size_cnns((in_height, in_width), [self.cn1, self.cn2, self.cn3])
        flattened_dim = o_height * o_width * self.cn3.out_channels
        

        self.means = nn.Linear(flattened_dim, out_dim)
        self.ln_vars = nn.Linear(flattened_dim, out_dim)
        self.eps = Normal(torch.zeros(out_dim), torch.ones(out_dim))

    def forward(self, img_batch_in):
        batch_size = img_batch_in.shape[0]
        conv_out = F.relu(self.cn1(img_batch_in))
        conv_out = F.relu(self.cn2(conv_out))
        conv_out = F.relu(self.cn3(conv_out))

        flattened_im = torch.flatten(conv_out, 1)

        means = self.means(flattened_im)
        stds = torch.exp(self.log_vars(flattened_im) / 2) 
        eps = self.eps.sample_n(batch_size)

        # Reparameterization Trick
        z = means + eps * stds

        return z


class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        # The out features are enc_out_channel_dims, encoder_out_height, encoder_out_width
        self.lin1 = nn.Linear(z_dim, enc_out_channel_dims * e_out_height * e_out_width)
        self.ctn1 = nn.ConvTranspose2d(in_channels=enc_out_channel_dims, out_channels=enc_out_channel_dims / 2.0, kernel_size=4, stride=2, padding=1)
        self.ctn2 = nn.ConvTranspose2d(in_channels=enc_out_channel_dims / 2.0, out_channels=enc_out_channel_dims / 4.0, kernel_size=4, stride=2, padding=1)
        self.ctn3 = nn.ConvTranspose2d(in_channels=enc_out_channel_dims / 4.0, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, z_batch):
        # Push it back through a linear layer till we are back at the flattened_im dimensions
        # Reshape the flattened im back into unflattened form. (However, the examples seem to shape it in a strange way, so not 100 percent here.)
        # Pass through multiple deconvolution layers (or transposed convolutions) and relus to get back up to desired image shape (Read transpose 2d manual, the guide to convolutional arithmetic etc.)
        # outpout this image


def out_size_cnns(img_dims, cnns):
    current_dims = img_dims
    for cnn in cnns:
        current_dims = output_size(current_dims[0], current_dims[1], cnn.kernel_size, cnn.stride, cnn.padding)
    return current_dims

def output_size(in_height, in_width, kernel_size, stride=1, padding=0):
    out_height = int((in_height - kernel_size + padding * 2) / stride) + 1
    out_width = int((in_width - kernel_size + padding * 2) / stride) + 1
    return (out_height, out_width)




# Make a decoder module

## Make a VAE module?

## Make the cost function

## Construct a training regiment which loads in images as a dataset, implements skip count (as a function?), etc...
## Implement the various image augmentation / contrasts.