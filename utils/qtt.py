import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def contract_cores(cores, L):
    """Contract a list of (r,2,2,r) torch cores into the (2^L, 2^L) image matrix."""
    X = cores[0].squeeze(0)
    for k in range(1, L):
        X = torch.einsum("...a,aijb->...ijb", X, cores[k])
    X = X.squeeze(-1)
    row_axes = list(range(0, 2 * L, 2))
    col_axes = list(range(1, 2 * L, 2))
    X = X.permute(*(row_axes + col_axes)).contiguous()
    return X.view(2 ** L, 2 ** L)


def contract_cores_np(cores):
    """Same as ``contract_cores`` for numpy cores (used after decoding circuits)."""
    cores_t = [torch.as_tensor(np.asarray(c)) for c in cores]
    return contract_cores(cores_t, len(cores_t)).numpy()


class QTT(nn.Module):
    """Learnable QTT: ``L`` cores of shape (r_{k-1}, 2, 2, r_k)."""

    def __init__(self, L=5, rank=4, init_std=0.1):
        super().__init__()
        self.L = int(L)
        self.side = 2 ** self.L
        if isinstance(rank, int):
            ranks = [1] + [rank] * (self.L - 1) + [1]
        else:
            ranks = list(rank)
            assert len(ranks) == self.L + 1 and ranks[0] == 1 and ranks[-1] == 1
        self.ranks = ranks
        self.cores = nn.ParameterList([
            nn.Parameter(init_std * torch.randn(ranks[k], 2, 2, ranks[k + 1]))
            for k in range(self.L)
        ])

    def forward(self):
        return contract_cores(list(self.cores), self.L)

    def count_params(self):
        return sum(c.numel() for c in self.cores)

    def cores_numpy(self):
        return [c.detach().cpu().numpy().astype(np.float64) for c in self.cores]


def train_qtt(target, rank=4, steps=2000, lr=1e-2, init_std=0.1, device=None, seed=None,
              verbose=False):
    """Fit a QTT to ``target`` ((2^L,2^L) array in [0,1]) with Adam + MSE.

    Returns (model, recon) where ``recon`` is the (2^L,2^L) numpy reconstruction.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if seed is not None:
        torch.manual_seed(int(seed))
    target_t = torch.as_tensor(np.asarray(target, dtype=np.float32), device=device)
    L = int(round(np.log2(target_t.shape[0])))
    model = QTT(L=L, rank=rank, init_std=init_std).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(model(), target_t)
        loss.backward()
        opt.step()
        if verbose and (step % max(1, steps // 10) == 0 or step == 1):
            print(f"  step={step:5d}  mse={loss.item():.6e}")
    with torch.no_grad():
        recon = model().detach().cpu().numpy().astype(np.float64)
    return model, recon
