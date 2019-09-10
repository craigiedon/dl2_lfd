import torch
import math
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.utils.data import TensorDataset, DataLoader

from torch import autograd
import matplotlib.pyplot as plt
import numpy as np


class MDN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, mix_num):
        super(MDN, self).__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.mix_num = mix_num

        self.ff_1 = nn.Linear(in_dim, hidden_dim)
        self.ff_1_drop = nn.Dropout()

        self.ff_2 = nn.Linear(hidden_dim, hidden_dim)
        self.ff_2_drop = nn.Dropout()

        self.ff_mu = nn.Linear(hidden_dim, out_dim * mix_num)
        self.ff_ln_var = nn.Linear(hidden_dim, mix_num)
        self.ff_pi = nn.Linear(hidden_dim, mix_num)

    def forward(self, ins):
        enc_out = F.relu(self.ff_1(ins))
        # enc_out = self.ff_1_drop(enc_out)
        enc_out = F.relu(self.ff_2(enc_out))
        # enc_out = self.ff_2_drop(enc_out)

        mu = self.ff_mu(enc_out).reshape((-1, self.mix_num, self.out_dim))
        std = torch.exp(self.ff_ln_var(enc_out))
        pi = F.softmax(self.ff_pi(enc_out), dim=1)
        return mu, std, pi

def gaussian_spherical_pdf(mu, std, targets):
    # Shape mu: N x C X D, # Std: N X C, targets: N X D
    n_samples, n_comps, n_dims = mu.shape[0], mu.shape[1], mu.shape[2]
    pi_const = -0.5 * n_comps * np.log(2 * np.pi) # dim: 1
    ln_stds = -n_comps * torch.log(std) # dim: N X C
    # dim1: C X N X D - dim2: N X D = C X N X D
    # summed out: N X C
    mean_dist = (targets.unsqueeze(1).expand_as(mu) - mu).sum(dim=2) * torch.reciprocal(std) 
    return pi_const + ln_stds - 0.5 * torch.pow(mean_dist, 2) # out: N X C
    
def mdn_loss(mu, std, pi, targets):
    ln_probs_mix = gaussian_spherical_pdf(mu, std, targets) # dim: N X C
    weighted_mix = ln_probs_mix + torch.log(pi)
    ln_probs = torch.logsumexp(weighted_mix, dim=1)
    return -ln_probs.mean()

def gumbel_sample(x, axis=1):
    z = np.random.gumbel(loc=0, scale=1, size=x.shape)
    return (np.log(x.detach().cpu().numpy()) + z).argmax(axis=axis)

def mdn_sample(mu, std, pi):
    mix_ids = gumbel_sample(pi) # dim: N
    # mus : N X C X D
    selected_mus = mu[range(mu.shape[0]), mix_ids] # dims: N X D?
    selected_stds = std[range(std.shape[0]), mix_ids] # dims: N X D?
    eps = torch.randn_like(selected_stds)
    print("selected_mus : {}, eps {}, std: {}, stdselected: {}".format(selected_mus.shape, eps.shape, std.shape, selected_stds.shape))
    # (N X D) + N * N
    return selected_mus + (eps * selected_stds).unsqueeze(1).expand_as(selected_mus)


if __name__ == "__main__":
    # Example problem: Setup some train input and train targets
    n_samples = 1000
    device = torch.device("cuda")
    targets_train = torch.linspace(-1, 1, n_samples)
    inputs_train = (torch.sin(7.5*targets_train) + 0.05*targets_train + 0.1 * torch.randn(n_samples))

    plt.scatter(inputs_train, targets_train, alpha=0.2)
    plt.show()

    targets_train = targets_train.to(device).unsqueeze(1)
    inputs_train = inputs_train.to(device).unsqueeze(1)

    # print(targets_train.shape)
    # print(inputs_train.shape)


    model = MDN(1, 20, 1, 5)
    model.to(device)
    optimizer = optim.Adam(model.parameters())
    loss_fn = mdn_loss

    num_epochs = 1000
    data_set = TensorDataset(inputs_train, targets_train)
    data_loader = DataLoader(data_set, batch_size=1000, shuffle=True)
    for epoch in range(num_epochs):
        train_losses = []
        for i, (batch_ins, batch_targets) in enumerate(data_loader):
            with autograd.detect_anomaly():
                mu, std, pi = model(batch_ins)
                train_loss = loss_fn(mu, std, pi, batch_targets)
                train_losses.append(train_loss.item())
                if math.isnan(train_loss.item()):
                    print("Found a NAN!")
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        print("Epoch: {}, Loss: {}".format(epoch, np.mean(train_losses)))

    # With no gradients
    model.eval()
    with torch.no_grad():
        inputs_test = torch.linspace(-1, 1, n_samples).unsqueeze(1).to(device)
        print(inputs_test.shape)
        test_mu, test_std, test_pi = model(inputs_test)

    # mix_components = gumbel_sample(test_pi)
    test_samples = mdn_sample(test_mu, test_std, test_pi)
    print("inp test: {}, test samples {}".format(inputs_test.shape, test_samples.shape))
    plt.scatter(inputs_train.detach().cpu().numpy(), targets_train.detach().cpu().numpy(), alpha=0.2)
    plt.scatter(inputs_test.cpu().numpy(), test_samples.cpu().numpy(), color="orange", alpha=0.2)
    plt.show()
    print("Test shape:{}".format(test_samples.shape))
    print(test_mu.shape, test_std.shape, test_pi.shape)