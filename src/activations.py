"""
Activation Functions implemented from scratch in pure NumPy.

Intuition Behind GELU (Gaussian Error Linear Unit):
--------------------------------------------------
In standard ReLU(x) = max(0, x), inputs are deterministically multiplied by 1
if positive, and completely zeroed out if negative. This causes the "dying ReLU"
problem where neurons can become permanently inactive if their weights push
inputs strictly below zero.

GELU smoothly weights inputs by their probability under a standard Gaussian:
  GELU(x) = x * P(X <= x) where X ~ Normal(0, 1)

For large positive x, P(X <= x) approaches 1, so GELU(x) -> x (like ReLU).
For large negative x, P(X <= x) approaches 0, so GELU(x) -> 0.
Importantly, for moderately negative x (e.g. x = -0.5), GELU is slightly negative
and has a smooth non-zero curvature and gradient, allowing gradients to flow
even when activations are slightly negative.
"""

import numpy as np
from typing import Optional


class GELU:
    """
    Gaussian Error Linear Unit (GELU) with the standard tanh approximation.

    Mathematical Equation:
    ----------------------
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))

    Constants:
      sqrt(2 / pi) approx 0.7978845608028654
      cubic coeff  = 0.044715
    """

    SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
    COEFF = 0.044715

    def __init__(self):
        self.cached_x: Optional[np.ndarray] = None
        self.cached_tanh: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for GELU.
        
        Input x: shape (..., D)
        Output:  shape (..., D)
        """
        self.cached_x = x

        # u = sqrt(2/pi) * (x + 0.044715 * x^3)
        inner = self.SQRT_2_OVER_PI * (x + self.COEFF * (x ** 3))
        tanh_inner = np.tanh(inner)
        self.cached_tanh = tanh_inner

        # GELU(x) = 0.5 * x * (1 + tanh(inner))
        out = 0.5 * x * (1.0 + tanh_inner)
        return out

    def backward(self, d_out: np.ndarray) -> np.ndarray:
        """
        Analytical backward pass for GELU.

        Derivation:
        -----------
        Let u = sqrt(2/pi) * (x + 0.044715 * x^3)
        Let t = tanh(u)
        GELU(x) = 0.5 * x * (1 + t)

        d(GELU)/dx = 0.5 * (1 + t) + 0.5 * x * (1 - t^2) * du/dx
        where du/dx = sqrt(2/pi) * (1 + 3 * 0.044715 * x^2)
        """
        assert self.cached_x is not None and self.cached_tanh is not None, "Call forward first."

        x = self.cached_x
        t = self.cached_tanh

        # Derivative of the inner term with respect to x
        du_dx = self.SQRT_2_OVER_PI * (1.0 + 3.0 * self.COEFF * (x ** 2))

        # Full derivative of GELU w.r.t x
        dgelu_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t ** 2) * du_dx

        # Chain rule: dL/dx = d_out * dgelu_dx
        dx = d_out * dgelu_dx
        return dx
