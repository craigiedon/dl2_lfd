from torch import nn

class DMPNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, dmp_dims, n_basis_fs):
        super(DMPNN, self).__init__()
        self.dmp_dims = dmp_dims
        self.n_basis_fs = n_basis_fs

        self.ff = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dmp_dims * n_basis_fs)
        )

    def forward(self, ins):
        flat_in = ins.view(ins.shape[0], -1)
        flat_out = self.ff(flat_in)
        return flat_out.view(-1, self.dmp_dims, self.n_basis_fs)

# Example builder
def dmp_nn(dmp_dims, n_basis_fs):
    return DMPNN(5, 100, dmp_dims, n_basis_fs)