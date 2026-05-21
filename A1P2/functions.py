"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - functions

Magda Gregorová, April 2026

Implement the functions marked with TODO.
Do not change function signatures.
Use only PyTorch tensor operations — no numpy, no torch.nn.

For MSE, the local and global gradients are always identical since MSE is the
root of the computation graph — there is no upstream gradient to multiply by.
We therefore provide a single backward function for MSE.

For Linear and ReLU, the backward pass is split into two functions:
  _lgrad: computes local gradients (partial derivatives of this node's output
          w.r.t. its inputs), independent of the rest of the graph
  _ggrad: computes global gradients by applying the chain rule with upstream
          gout, and returns the results
"""

import torch


# ==============================================================================
# Section 1 — MSE backward
# ==============================================================================

def mse_backward_scalar(y_pred, y):
    """Gradient of scalar MSE loss w.r.t. prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: float - predicted value
        y:      float - true value

    Returns:
        float - gradient dL/d(y_pred)
    """
    # TODO: implement
    return 2 * (y_pred - y)
    pass


def mse_backward(y_pred, y):
    """Gradient of batch MSE loss w.r.t. each prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: torch.tensor of shape (N, 1)
        y:      torch.tensor of shape (N, 1)

    Returns:
        torch.tensor of shape (N, 1)
    """
    # TODO: implement
    N = y_pred.shape[0]
    # print("N is like", N)
    return (2 / N) * (y_pred - y)
    pass


# ==============================================================================
# Section 2 — Linear backward
# ==============================================================================

def linear_lgrad_scalar(x, theta):
    """Local gradients for scalar linear function z = theta_1 * x + theta_0.

    Args:
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (lgrad_theta_0, lgrad_theta_1, lgrad_x)
               lgrad_theta_0: torch.tensor of shape ()
               lgrad_theta_1: torch.tensor of shape ()
               lgrad_x:       torch.tensor of shape ()

    Hint: use .clone() when returning parts of existing tensors.
    """
    # TODO: implement
    lgrad_theta_0 = torch.tensor(1.0)  # dZ/dtheta_0 = 1
    lgrad_theta_1 = x.clone()          # dZ/dtheta_1 = x
    lgrad_x       = theta[1].clone()   # dZ/dx = theta_1
    # print("theta", lgrad_theta_0, "theta1", lgrad_theta_1, "x", lgrad_x)
    return lgrad_theta_0, lgrad_theta_1, lgrad_x
    pass


def linear_ggrad_scalar(gout, x, theta):
    """Global gradients for scalar linear function z = theta_1 * x + theta_0.

    Applies the chain rule with upstream gout.
    In the scalar case this is elementwise: gout * lgrad.

    Args:
        gout:  torch.tensor of shape () - upstream gradient dL/dz
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (ggrad_theta_0, ggrad_theta_1, ggrad_x)
               each a torch.tensor of shape ()
    """
    # TODO: implement using linear_lgrad_scalar
    lgrad_theta_0, lgrad_theta_1, lgrad_x = linear_lgrad_scalar(x, theta)
    
    # then multiply each by gout (chain rule!)
    ggrad_theta_0 = gout * lgrad_theta_0  # gout × 1
    ggrad_theta_1 = gout * lgrad_theta_1  # gout × x
    ggrad_x       = gout * lgrad_x        # gout × theta_1
    
    return ggrad_theta_0, ggrad_theta_1, ggrad_x
    pass


def linear_lgrad(ins, theta_1, theta_0):
    """Local gradient factors for batched linear layer Z = X @ theta_1.T + theta_0.

    Args:
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (lgrad_theta_1_factor, lgrad_theta_0_factor, lgrad_ins)
               lgrad_theta_1_factor: torch.tensor of shape (N, in_features)
               lgrad_theta_0_factor: torch.tensor of shape (N, out_features)
               lgrad_ins:            torch.tensor of shape (out_features, in_features)
    """
    # TODO: implement
    lgrad_theta_1_factor = ins.clone()                    # (N, in_features)
    lgrad_theta_0_factor = torch.ones_like(ins @ theta_1.T)  # (N, out_features)
    lgrad_ins            = theta_1.clone()                # (out_features, in_features)
    return lgrad_theta_1_factor, lgrad_theta_0_factor, lgrad_ins
    pass


def linear_ggrad(gout, ins, theta_1, theta_0):
    """Global gradients for batched linear layer Z = X @ theta_1.T + theta_0.

    Applies the chain rule with upstream gout.
    In the matrix case this involves matrix products — think carefully about
    how the dimensions work out.

    Args:
        gout:    torch.tensor of shape (N, out_features) - upstream gradient dL/dZ
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (ggrad_theta_1, ggrad_theta_0, ggrad_ins)
               ggrad_theta_1: torch.tensor of shape (out_features, in_features)
               ggrad_theta_0: torch.tensor of shape (1, out_features)
               ggrad_ins:     torch.tensor of shape (N, in_features)
    """
    # TODO: implement using linear_lgrad
    ggrad_theta_1 = gout.T @ ins      # (out, in)
    ggrad_theta_0 = gout.sum(axis=0, keepdim=True)
    ggrad_ins     = gout @ theta_1
    return ggrad_theta_1, ggrad_theta_0, ggrad_ins
    pass


# ==============================================================================
# Section 3 — ReLU backward
# ==============================================================================

def relu_lgrad_scalar(z):
    """Local gradient for scalar ReLU: a = relu(z) = max(0, z).

    Args:
        z: torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - local gradient da/dz
    """
    # TODO: implement
    if z > 0:
        return 1.0
    else:
        return 0.0
    pass


def relu_ggrad_scalar(gout, z):
    """Global gradient for scalar ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of shape () - upstream gradient dL/da
        z:    torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - global gradient dL/dz
    """
    # TODO: implement using relu_lgrad_scalar
    return gout * relu_lgrad_scalar(z)
    pass


def relu_lgrad(ins):
    """Local gradient for batch ReLU: A = relu(Z), element-wise.

    Args:
        ins: torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - local gradient
    """
    # TODO: implement
    return (ins > 0)
    pass


def relu_ggrad(gout, ins):
    """Global gradient for batch ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of same shape as ins - upstream gradient dL/dA
        ins:  torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - global gradient dL/dZ
    """
    # TODO: implement using relu_lgrad
    return gout * relu_lgrad(ins)
    pass
