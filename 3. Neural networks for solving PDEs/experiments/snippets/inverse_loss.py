# Inside the model; raw_k is a trainable nn.Parameter.
@property
def k(self):
    return F.softplus(self.raw_k) + 1e-6

# r = (-k * u_xx - f) / pi**2; data_weight = 5.0
r = normalized_residual(model, x_r)
loss_physics = r.square().mean()
loss_data = (model(x_obs) - y_obs).square().mean()
loss = loss_physics + data_weight * loss_data
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
