"""THWS/MAI/Introduction to Deep Learning - Assignment 2 - layers

Magda Gregorová, April 2026

Implement the functions and classes marked with TODO.
Do not change function signatures or class interfaces.
Use only PyTorch tensor operations — no numpy, no torch.nn.
"""

import torch


# ==============================================================================
# Section 1 — Scalar forward pass
# ==============================================================================

def linear_scalar(x, theta):
    """Linear function for a single scalar input.

    Computes: f(x) = theta[1] * x + theta[0]

    Args:
        x:     float - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        float - scalar output
    """
    # TODO: implement

    return theta[1] * x + theta[0]
    pass


def squared_error(y_pred, y):
    """Squared error loss for a single prediction.

    Computes: L = (y_pred - y) ** 2

    Args:
        y_pred: float - predicted value
        y:      float - true value

    Returns:
        float - squared error
    """
    # TODO: implement
    return (y_pred - y) ** 2
    pass


# ==============================================================================
# Section 2 — Batch forward pass
# ==============================================================================

def linear_forward(X, theta_1, theta_0):
    """Linear forward pass for a batch of inputs.

    Computes: Y = X @ theta_1.T + theta_0

    Args:
        X:       torch.tensor of shape (N, in_features) - input batch
        theta_1: torch.tensor of shape (out_features, in_features) - weight matrix
        theta_0: torch.tensor of shape (1, out_features) - bias vector

    Returns:
        torch.tensor of shape (N, out_features) - output batch
    """
    return X @ theta_1.T + theta_0
    # TODO: implement
    pass


def mse_forward(y_pred, y):
    """Mean squared error loss for a batch of predictions.

    Computes: L = (1/N) * sum((y_pred - y) ** 2)

    Args:
        y_pred: torch.tensor of shape (N, 1) - predictions
        y:      torch.tensor of shape (N, 1) - true values

    Returns:
        torch.tensor of shape () - scalar MSE loss
    """
    # TODO: implement
    return ((y_pred - y) ** 2).mean()
    pass


# ==============================================================================
# Section 3 — Linear class
# ==============================================================================

class Linear:
    """Linear layer: applies affine transformation y = X @ theta_1.T + theta_0.

    Attributes:
        theta_1: torch.tensor of shape (out_features, in_features) - weight matrix
        theta_0: torch.tensor of shape (1, out_features) - bias vector
        ins:     torch.tensor of shape (N, in_features) - stored input (set in forward)
        outs:    torch.tensor of shape (N, out_features) - stored output (set in forward)
    """

    def __init__(self, theta_1, theta_0):
        """Initialise the layer with weight matrix and bias vector.

        Args:
            theta_1: torch.tensor of shape (out_features, in_features)
            theta_0: torch.tensor of shape (1, out_features)
        """
        self.theta_1 = theta_1
        self.theta_0 = theta_0
        # TODO: store theta_1 and theta_0 as attributes
        
        pass

    def forward(self, ins):
        """Forward pass: compute and return the affine transformation.

        Store the input in self.ins and the output in self.outs before returning.

        Args:
            ins: torch.tensor of shape (N, in_features)

        Returns:
            torch.tensor of shape (N, out_features)
        """
        # TODO: implement
        self.ins = ins
        self.outs = ins @ self.theta_1.T + self.theta_0
        return self.outs
        pass


# ==============================================================================
# Section 4 — ReLU class
# ==============================================================================

class ReLU:
    """ReLU non-linearity: applies a(z) = max(0, z) element-wise.

    Attributes:
        ins:  torch.tensor - stored pre-activation input z (set in forward)
        outs: torch.tensor - stored activation a (set in forward)
    """

    def forward(self, ins):
        """Forward pass: apply ReLU element-wise.

        Store the input in self.ins and the output in self.outs before returning.

        Args:
            ins: torch.tensor of any shape - pre-activations z

        Returns:
            torch.tensor of same shape - activations a = relu(z)
        """
        # TODO: implement
        self.ins = ins
        self.outs = torch.clamp(ins, min=0)
        return self.outs
        pass


# ==============================================================================
# Section 5 — Model class
# ==============================================================================

class Model:
    """Neural network model: a sequence of layers applied in order.

    Attributes:
        layers: list of layer objects (Linear, ReLU, ...) in forward order
    """

    def __init__(self, layers):
        """Initialise with a list of layers.

        Args:
            layers: list of layer instances in the order of the forward pass
        """
        self.layers = layers

    def forward(self, ins):
        """Forward pass through all layers in order.

        Args:
            ins: torch.tensor of shape (N, in_features) - network input

        Returns:
            torch.tensor of shape (N, out_features) - network output
        """
        # TODO: implement — pass ins through each layer in self.layers in order

        x = ins
        for layer in self.layers:
            x = layer.forward(x)
        return x
        pass
