def ritz_loss(model, x, weights):
    u = model(x)
    u_x = derivative(u, x)
    return (weights * (0.5 * u_x.square() - forcing(x) * u)).sum()
