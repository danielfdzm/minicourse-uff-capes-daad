def soft_loss(model, x, weights, penalty):
    boundary = torch.tensor([[0.0], [1.0]])
    boundary_loss = model(boundary).square().mean()
    return pinn_loss(model, x, weights) + penalty * boundary_loss
