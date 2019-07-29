

def setup(N, device):
    import torch
    from torch import nn
    network = nn.Sequential(
        nn.Linear(N, N),
        nn.ReLU(),
        nn.Linear(N, N),
        nn.Tanh(),
        nn.Linear(N, N)
        ).to(device)
    a = torch.rand(100, 1000, N, dtype=torch.float32, device=device)

def run(network, a):
    network(a)
