import torch
import torch.nn as nn

# Example network
class SimpleNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim)
        )
    def forward(self, x):
        return self.layers(x)

def residual_jacobian(model, x, y):
    """
    Compute Jacobian of residuals (f(x)-y) w.r.t. parameters.
    
    Args:
        model: nn.Module
        x: input tensor (batch, input_dim)
        y: target tensor (batch, output_dim)
    
    Returns:
        jacobian: tensor of shape (batch, output_dim, n_params)
                  entry [b,i,p] = ∂ residual_i(b) / ∂ θ_p
    """
    # Flatten parameters into a single vector
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)

    # Make sure we track grads
    for p in params:
        p.requires_grad_(True)

    # Compute residuals
    pred = model(x)  # (batch, output_dim)
    residuals = pred - y

    batch_size, output_dim = residuals.shape
    J = torch.zeros(batch_size, output_dim, n_params, device=x.device)

    # Loop over outputs to extract ∂r_i / ∂θ
    for i in range(output_dim):
        grad_outputs = torch.zeros_like(residuals)
        grad_outputs[:, i] = 1.0
        grads = torch.autograd.grad(
            outputs=residuals,
            inputs=params,
            grad_outputs=grad_outputs,
            retain_graph=True,
            create_graph=False,
            allow_unused=False
        )
        # Flatten into parameter vector
        grads_flat = torch.cat([g.reshape(batch_size, -1) for g in grads], dim=1)  # (batch, n_params)
        J[:, i, :] = grads_flat

    return J

# Example usage
model = SimpleNet(input_dim=3, output_dim=2)
x = torch.randn(4, 3)
y = torch.randn(4, 2)

J = residual_jacobian(model, x, y)
print("Jacobian shape:", J.shape)  # (batch=4, output_dim=2, n_params)