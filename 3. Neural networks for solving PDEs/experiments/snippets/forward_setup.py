# Imports: numpy as np, torch, and torch.nn as nn
initial = nn.Sequential(
    nn.Linear(1, 32), nn.Tanh(),
    nn.Linear(32, 32), nn.Tanh(),
    nn.Linear(32, 1)
)
nodes, quadrature_weights = np.polynomial.legendre.leggauss(QUADRATURE_NODES)
x = torch.tensor(((nodes + 1.0) / 2.0)[:, None], requires_grad=True)
weights = torch.tensor((quadrature_weights / 2.0)[:, None])

hard_model = Solution(initial, hard_boundary=True)
