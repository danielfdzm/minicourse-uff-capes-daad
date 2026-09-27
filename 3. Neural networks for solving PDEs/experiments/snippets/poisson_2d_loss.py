def normalized_residual(model, xy):
    """Return (-Delta u_theta - f)/(2*pi**2)."""
    u = model(xy)
    grad_u = torch.autograd.grad(
        u.sum(), xy, create_graph=True)[0]
    u_xx = torch.autograd.grad(
        grad_u[:, 0].sum(), xy, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(
        grad_u[:, 1].sum(), xy, create_graph=True)[0][:, 1:2]
    x, y = xy[:, 0:1], xy[:, 1:2]
    f = 2 * math.pi**2 * torch.sin(math.pi*x) * torch.sin(math.pi*y)
    return (-u_xx - u_yy - f) / (2 * math.pi**2)
