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
from layers import linear_forward, mse_forward


class Linear:
    def __init__(self, theta_1, theta_0):
        self.theta_1 = theta_1
        self.theta_0 = theta_0

    def forward(self, ins):
        self.ins  = ins
        self.outs = linear_forward(ins, self.theta_1, self.theta_0)  # ← from layers.py
        return self.outs

    def backward(self, gout):
        ggrad_theta_1, ggrad_theta_0, ggrad_ins = linear_ggrad(
            gout,
            self.ins,
            self.theta_1,
            self.theta_0
        )
        self.theta_1.g = ggrad_theta_1
        self.theta_0.g = ggrad_theta_0
        return ggrad_ins


class ReLU:

    def forward(self, ins):
        self.ins  = ins
        self.outs = torch.clamp(ins, min=0)
        return self.outs

    def backward(self, gout):
        self.ins.g = relu_ggrad(gout, self.ins)  # ← from functions.py
        return self.ins.g


class Model:

    def __init__(self, layers):
        self.layers = layers

    def forward(self, ins):
        x = ins
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, gout):
        for layer in reversed(self.layers):
            gout = layer.backward(gout)
        return gout