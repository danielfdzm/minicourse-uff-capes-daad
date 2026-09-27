def derivative(y, x):
    return torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True)[0]

def pinn_loss(model, x, weights):
    u = model(x)
    u_x = derivative(u, x)
    u_xx = derivative(u_x, x)
    residual = (-u_xx - forcing(x)) / math.pi**2
    return (weights * residual.square()).sum()
optimizer = torch.optim.Adam(hard_model.parameters(), lr=1e-3)
optimizer.zero_grad()
loss = pinn_loss(hard_model, x, weights)
loss.backward()
optimizer.step()
