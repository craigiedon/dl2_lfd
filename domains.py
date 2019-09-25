import numpy as np
import torch

class Box:

    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.name = 'box'

    def is_empty(self):
        return torch.any(self.a > self.b)
        
    def project(self, x):
        return torch.min(torch.max(x, self.a), self.b)

    def sample(self):
        s = (self.b - self.a) * torch.rand_like(self.a) + self.a
        return s

class Segment:

    def __init__(self, p1, p2):
        self.p1_n = p1.cpu().numpy()
        self.p2_n = p2.cpu().numpy()
        self.d = self.p2_n - self.p1_n
        self.d_norm = self.d / np.linalg.norm(self.d)
        self.name = 'segment'

    def is_empty(self):
        return False

    def project(self, x):
        dp = np.sum((x - self.p1_n) * self.d_norm)
        if dp < 0:
            return self.p1_n
        elif dp > np.linalg.norm(self.d):
            return self.p2_n
        else:
            return self.p1_n + dp * self.d_norm

    def sample(self):
        return self.p1_n + (self.p2_n - self.p1_n) * np.random.random()

# class L2_Ball(Domain):

#     def __init__(self, r):
#         self.r = r

#     def project(self, x_batch):
#         norms = np.linalg.norm(x_batch, axis=1)
#         p_batch = x_batch.clone()
#         p_batch[norms > self.r] = x_batch / norms * self.r
#         return p_batch



