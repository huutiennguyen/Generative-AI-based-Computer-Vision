import torch

def solve_ode(model, cond_img: torch.Tensor, t_steps, device):
    """
    Solves the ODE using the Euler method to transform random noise Gaussian into a HR image, conditioned on the low-field image.

    Args:
        - model (torch.nn.Module): Pre-trained UNet model.
        - cond_img (torch.Tensor): Conditional LR image, shape (B, C, H, W).
        - t_steps (int): Number of integration steps to solve the ODE.

    Returns:
        torch.Tensor: The generated HR image.
    """
    x_t = (torch.rand_like(cond_img) * 2) - 1 # normalized to [-1, 1]

    dt = 1.0 / t_steps
    with torch.no_grad():
        for i in range(t_steps):
            t_val = i * dt
            t = torch.tensor([t_val] * x_t.shape[0], device=device)
            predicted_u_t = model(x_t=x_t, time=t, x_cond=cond_img)
            x_t = x_t + predicted_u_t * dt

    return x_t