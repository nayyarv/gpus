
def setup(N, device):
    import torch
    a = torch.rand(N, N, dtype=torch.float32, device=device)
    b = torch.rand(N, N, dtype=torch.float32, device=device)

def run(a, b):
    a @ b

