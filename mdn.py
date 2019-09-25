"""A module for a mixture density network layer

For more info on MDNs, see _Mixture Desity Networks_ by Bishop, 1994.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical
import math


ONEOVERSQRT2PI = 1.0 / math.sqrt(2*math.pi)

class MDN(nn.Module):
    """A mixture density network layer

    The input maps to the parameters of a MoG probability distribution, where
    each Gaussian has O dimensions and diagonal covariance.

    Arguments:
        in_features (int): the number of dimensions in the input
        out_features (int): the number of dimensions in the output
        num_gaussians (int): the number of Gaussians per output dimensions

    Input:
        minibatch (BxD): B is the batch size and D is the number of input
            dimensions.

    Output:
        (pi, sigma, mu) (BxG, BxGxO, BxGxO): B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions for each
            Gaussian. Pi is a multinomial distribution of the Gaussians. Sigma
            is the standard deviation of each Gaussian. Mu is the mean of each
            Gaussian.
    """
    def __init__(self, in_features, out_features, num_gaussians):
        super(MDN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_gaussians = num_gaussians
        self.pi = nn.Sequential(
            nn.Linear(in_features, num_gaussians),
            nn.Softmax(dim=1)
        )
        self.sigma = nn.Linear(in_features, out_features*num_gaussians)
        self.mu = nn.Linear(in_features, out_features*num_gaussians)

    def forward(self, minibatch):
        pi = self.pi(minibatch)
        sigma = torch.exp(self.sigma(minibatch))
        sigma = sigma.view(-1, self.num_gaussians, self.out_features)
        mu = self.mu(minibatch)
        mu = mu.view(-1, self.num_gaussians, self.out_features)
        return pi, sigma, mu


def gaussian_probability(sigma, mu, target):
    """Returns the probability of `data` given MoG parameters `sigma` and `mu`.
    
    Arguments:
        sigma (BxGxO): The standard deviation of the Gaussians. B is the batch
            size, G is the number of Gaussians, and O is the number of
            dimensions per Gaussian.
        mu (BxGxO): The means of the Gaussians. B is the batch size, G is the
            number of Gaussians, and O is the number of dimensions per Gaussian.
        data (BxI): A batch of data. B is the batch size and I is the number of
            input dimensions.

    Returns:
        probabilities (BxG): The probability of each point in the probability
            of the distribution in the corresponding sigma/mu index.
    """
    target = target.unsqueeze(1).expand_as(sigma)
    ret = ONEOVERSQRT2PI * torch.exp(-0.5 * ((target - mu) / sigma)**2) / sigma
    return torch.prod(ret, 2)


def mdn_loss(pi, sigma, mu, target):
    """Calculates the error, given the MoG parameters and the target

    The loss is the negative log likelihood of the data given the MoG
    parameters.
    """
    prob = pi * gaussian_probability(sigma, mu, target)
    nll = -torch.log(torch.sum(prob, dim=1))
    return torch.mean(nll)


def sample(pi, sigma, mu, num_samples=1):
    """Draw samples from a MoG. pi : M, sigma: M X D, mu: M X D
    """
    categorical = Categorical(pi)
    sampled_pis = categorical.sample_n(num_samples)
    selected_mus = mu[sampled_pis]
    selected_sigmas = sigma[sampled_pis]
    eps = torch.randn_like(selected_mus)

    # print("sel mus {}, eps {}, sel sigs {}".format(selected_mus.shape, eps.shape, selected_sigmas.shape))

    return selected_mus + eps * selected_sigmas

def approx_ml(pi, sigma, mu, num_samples=100):
    samples = sample(pi, sigma, mu, num_samples)
    likelihoods = (pi * gaussian_probability(sigma.expand(num_samples, -1, -1), mu.expand(num_samples, -1, -1), samples)).sum(dim=1)
    # print(likelihoods.shape)
    most_likely_id = torch.argmax(likelihoods)
    return samples[most_likely_id]
    

# import torch
# import math
# from torch import nn, optim
# import torch.nn.functional as F
# from torch.distributions.normal import Normal
# from torch.utils.data import TensorDataset, DataLoader

# from torch import autograd
# import matplotlib.pyplot as plt
# import numpy as np


# class MDN(nn.Module):
#     def __init__(self, in_dim, out_dim, mix_num):
#         super(MDN, self).__init__()

#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.mix_num = mix_num

#         self.ff_mu = nn.Linear(in_dim, out_dim * mix_num)
#         self.ff_ln_var = nn.Linear(in_dim, mix_num)
#         self.ff_pi = nn.Linear(in_dim, mix_num)

#     def forward(self, ins):
#         mu = self.ff_mu(ins).reshape((-1, self.mix_num, self.out_dim))
#         std = torch.exp(self.ff_ln_var(ins))
#         pi = F.softmax(self.ff_pi(ins), dim=1)
#         return mu,std,pi

# def gaussian_spherical_pdf(mu, std, targets):
#     # Shape mu: N x C X D, # Std: N X C, targets: N X D
#     n_samples, n_comps, n_dims = mu.shape[0], mu.shape[1], mu.shape[2]
#     pi_const = -0.5 * n_comps * np.log(2 * np.pi) # dim: 1
#     ln_stds = -n_comps * torch.log(std) # dim: N X C
#     # dim1: C X N X D - dim2: N X D = C X N X D
#     # summed out: N X C
#     mean_dist = (targets.unsqueeze(1).expand_as(mu) - mu).sum(dim=2) * torch.reciprocal(std) 
#     return pi_const + ln_stds - 0.5 * torch.pow(mean_dist, 2) # out: N X C
    
# def mdn_loss(mu, std, pi, targets):
#     ln_probs_mix = gaussian_spherical_pdf(mu, std, targets) # dim: N X C
#     weighted_mix = ln_probs_mix + torch.log(pi)
#     ln_probs = torch.logsumexp(weighted_mix, dim=1)
#     return -ln_probs.mean()

# def gumbel_sample(x, axis=1):
#     z = np.random.gumbel(loc=0, scale=1, size=x.shape)
#     return (np.log(x.detach().cpu().numpy()) + z).argmax(axis=axis)

# def mdn_sample(mu, std, pi):
#     mix_ids = gumbel_sample(pi) # dim: N
#     # mus : N X C X D
#     selected_mus = mu[range(mu.shape[0]), mix_ids] # dims: N X D?
#     selected_stds = std[range(std.shape[0]), mix_ids] # dims: N X D?
#     eps = torch.randn_like(selected_stds)
#     print("selected_mus : {}, eps {}, std: {}, stdselected: {}".format(selected_mus.shape, eps.shape, std.shape, selected_stds.shape))
#     # (N X D) + N * N
#     return selected_mus + (eps * selected_stds).unsqueeze(1).expand_as(selected_mus)


# if __name__ == "__main__":
#     # Example problem: Setup some train input and train targets
#     n_samples = 1000
#     device = torch.device("cuda")
#     targets_train = torch.linspace(-1, 1, n_samples)
#     inputs_train = (torch.sin(7.5*targets_train) + 0.05*targets_train + 0.1 * torch.randn(n_samples))

#     plt.scatter(inputs_train, targets_train, alpha=0.2)
#     plt.show()

#     targets_train = targets_train.to(device).unsqueeze(1)
#     inputs_train = inputs_train.to(device).unsqueeze(1)

#     # print(targets_train.shape)
#     # print(inputs_train.shape)


#     model = MDN(1, 20, 1, 5)
#     model.to(device)
#     optimizer = optim.Adam(model.parameters())
#     loss_fn = mdn_loss

#     num_epochs = 1000
#     data_set = TensorDataset(inputs_train, targets_train)
#     data_loader = DataLoader(data_set, batch_size=1000, shuffle=True)
#     for epoch in range(num_epochs):
#         train_losses = []
#         for i, (batch_ins, batch_targets) in enumerate(data_loader):
#             with autograd.detect_anomaly():
#                 mu, std, pi = model(batch_ins)
#                 train_loss = loss_fn(mu, std, pi, batch_targets)
#                 train_losses.append(train_loss.item())
#                 if math.isnan(train_loss.item()):
#                     print("Found a NAN!")
#                 optimizer.zero_grad()
#                 train_loss.backward()
#                 optimizer.step()
#         print("Epoch: {}, Loss: {}".format(epoch, np.mean(train_losses)))

#     # With no gradients
#     model.eval()
#     with torch.no_grad():
#         inputs_test = torch.linspace(-1, 1, n_samples).unsqueeze(1).to(device)
#         print(inputs_test.shape)
#         test_mu, test_std, test_pi = model(inputs_test)

#     # mix_components = gumbel_sample(test_pi)
#     test_samples = mdn_sample(test_mu, test_std, test_pi)
#     print("inp test: {}, test samples {}".format(inputs_test.shape, test_samples.shape))
#     plt.scatter(inputs_train.detach().cpu().numpy(), targets_train.detach().cpu().numpy(), alpha=0.2)
#     plt.scatter(inputs_test.cpu().numpy(), test_samples.cpu().numpy(), color="orange", alpha=0.2)
#     plt.show()
#     print("Test shape:{}".format(test_samples.shape))
#     print(test_mu.shape, test_std.shape, test_pi.shape)