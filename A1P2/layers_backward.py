"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - layers with backward pass

Magda Gregorová, April 2026

This file extends the classes you implemented in Part 1.

Instructions:
  1. Copy your Linear, ReLU, and Model classes from layers.py (Part 1) into this file.
  2. Add the backward methods to Linear, ReLU, and Model as described below.
  3. Do not modify the forward methods.

Use only PyTorch tensor operations — no numpy, no torch.nn.
"""

import torch
from functions import linear_ggrad, relu_ggrad


# TODO:
class Linear:
    def __init__(self, theta_1, theta_0):
        self.theta_1 = theta_1
        self.theta_0 = theta_0

    def forward(self, ins):
        self.ins  = ins
        self.outs = ins @ self.theta_1.T + self.theta_0
        return self.outs

    def backward(self, gout):
        # use stored self.ins from forward pass!
        ggrad_theta_1, ggrad_theta_0, ggrad_ins = linear_ggrad(
            gout,
            self.ins,      # ← stored during forward!
            self.theta_1,
            self.theta_0
        )
        
        # store gradients inside layer
        self.theta_1.g = ggrad_theta_1
        self.theta_0.g = ggrad_theta_0
        
        # return gradient for previous layer
        return ggrad_ins

# TODO: Copy your ReLU class from Part 1 here and add the backward method below.
#
class ReLU:

    def forward(self, ins):
        self.ins  = ins
        self.outs = torch.clamp(ins, min=0)
        return self.outs

    def backward(self, gout):
        """Backward pass: compute and store global gradient.

        Stores gradient as attribute on the tensor:
            self.ins.g of same shape as self.ins

        Args:
            gout: torch.tensor of same shape as self.ins - upstream gradient dL/dA

        Returns:
            torch.tensor of same shape - gradient w.r.t. input
        """
        # global gradient = gout × local gradient
        # local gradient  = 1 where ins > 0, else 0
        self.ins.g = relu_ggrad(gout, self.ins)
        return self.ins.g


class Model:

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
        x = ins
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gout):
        """Backward pass through all layers in reverse order.

        Args:
            gout: torch.tensor - upstream gradient from the loss

        Returns:
            torch.tensor - gradient w.r.t. network input
        """
        # go through layers in REVERSE order!
        for layer in reversed(self.layers):
            gout = layer.backward(gout)
        return gout
