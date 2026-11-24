"""
Neural Network Optimizers in Pure NumPy: SGD and AdamW with Gradient Clipping.

Adam / AdamW Optimizer Algorithm:
--------------------------------
Adam (Adaptive Moment Estimation) maintains running exponential moving averages
of both the past gradients (first moment m) and the squared gradients (second moment v).

Algorithm steps at time step t:
1. First moment vector (estimates gradient direction / momentum):
     m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
     
2. Second moment vector (estimates uncentered variance / curvature):
     v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
     
3. Bias correction (compensates for initialization of m and v at zero):
     m_hat_t = m_t / (1 - beta1 ** t)
     v_hat_t = v_t / (1 - beta2 ** t)

4. Decoupled Weight Decay (AdamW):
   Unlike L2 regularization which adds wd * theta to the gradient, AdamW directly
   scales down parameters independently of the gradient adaptivity:
     theta = theta - lr * weight_decay * theta  (for 2D weight matrices)

5. Parameter Update:
     theta = theta - (lr / (sqrt(v_hat_t) + eps)) * m_hat_t
"""

import numpy as np
from typing import Dict, Tuple, Optional


def get_cosine_schedule_with_warmup_lr(
    step: int,
    max_iters: int,
    warmup_iters: int = 50,
    lr_max: float = 3e-4,
    lr_min: float = 3e-5
) -> float:
    """
    Computes learning rate with linear warmup followed by cosine annealing decay.

    Mathematical Formulation:
    --------------------------
    1. Linear Warmup Phase (step < warmup_iters):
         The learning rate starts near 0.0 and ramps linearly up to lr_max:
           lr_t = lr_max * (t / max(1, warmup_iters))
         Rationale: Stabilizes initial gradients when random weights produce large updates.

    2. Cosine Annealing Decay (warmup_iters <= step <= max_iters):
         Progress ratio:
           decay_ratio = (t - warmup_iters) / max(1, max_iters - warmup_iters)
         Cosine factor:
           coeff = 0.5 * (1.0 + cos(pi * decay_ratio))
         Learning rate:
           lr_t = lr_min + coeff * (lr_max - lr_min)
         Rationale: Smooth, continuous monotonic descent from lr_max to lr_min without
         abrupt step decay discontinuities.

    3. Post-Schedule Cooldown (step > max_iters):
         lr_t = lr_min

    Args:
        step (int): Current optimization step (1-indexed or 0-indexed).
        max_iters (int): Total expected training steps.
        warmup_iters (int): Number of initial warmup steps.
        lr_max (float): Peak learning rate after warmup.
        lr_min (float): Minimum terminal learning rate.

    Returns:
        float: Computed learning rate for the current step.
    """
    if step < warmup_iters:
        return float(lr_max * float(step) / float(max(1, warmup_iters)))
    if step > max_iters:
        return float(lr_min)
    decay_ratio = float(step - warmup_iters) / float(max(1, max_iters - warmup_iters))
    decay_ratio = min(1.0, max(0.0, decay_ratio))
    coeff = 0.5 * (1.0 + float(np.cos(np.pi * decay_ratio)))
    return float(lr_min + coeff * (lr_max - lr_min))


def clip_grad_norm(
    params_and_grads: Dict[str, Tuple[np.ndarray, np.ndarray]],
    max_norm: float = 1.0
) -> float:
    """
    Clips the global L2 norm of all gradients to max_norm.
    Prevents gradient explosion during training.
    """
    total_sq_norm = 0.0
    for name, (param, grad) in params_and_grads.items():
        total_sq_norm += np.sum(grad ** 2)
    total_norm = float(np.sqrt(total_sq_norm))

    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for name, (param, grad) in params_and_grads.items():
            grad *= scale

    return total_norm


class SGD:
    """
    Stochastic Gradient Descent optimizer.
    theta = theta - lr * grad
    """

    def __init__(
        self,
        params_and_grads: Dict[str, Tuple[np.ndarray, np.ndarray]],
        lr: float = 1e-3
    ):
        self.params_and_grads = params_and_grads
        self.lr = lr

    def step(self) -> None:
        for name, (param, grad) in self.params_and_grads.items():
            param -= self.lr * grad


class AdamW:
    """
    AdamW (Adam with Decoupled Weight Decay) optimizer implemented in pure NumPy.
    """

    def __init__(
        self,
        params_and_grads: Dict[str, Tuple[np.ndarray, np.ndarray]],
        lr: float = 3e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        self.params_and_grads = params_and_grads
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.t = 0  # Timestep counter

        # First and second moment accumulators
        self.m: Dict[str, np.ndarray] = {}
        self.v: Dict[str, np.ndarray] = {}

        for name, (param, _) in self.params_and_grads.items():
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

    def step(self) -> None:
        """
        Performs a single AdamW optimization step across all parameters.
        """
        self.t += 1
        lr = self.lr
        beta1, beta2 = self.beta1, self.beta2
        eps = self.eps

        # Bias correction factors for timestep t
        bias_correction1 = 1.0 - (beta1 ** self.t)
        bias_correction2 = 1.0 - (beta2 ** self.t)

        for name, (param, grad) in self.params_and_grads.items():
            # Update biased 1st moment estimate
            # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
            m = self.m[name]
            m *= beta1
            m += (1.0 - beta1) * grad

            # Update biased 2nd moment estimate
            # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
            v = self.v[name]
            v *= beta2
            v += (1.0 - beta2) * (grad * grad)

            # Compute bias-corrected moments
            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            # Apply decoupled weight decay only to 2D weight matrices (not 1D biases or layer norm parameters)
            if self.weight_decay > 0.0 and param.ndim >= 2:
                param -= lr * self.weight_decay * param

            # Parameter update
            step_update = (lr / (np.sqrt(v_hat) + eps)) * m_hat
            param -= step_update

    def set_lr(self, lr: float) -> None:
        """Sets current learning rate."""
        self.lr = float(lr)

    def state_dict(self) -> dict:
        """Serializes optimizer state."""
        return {
            "t": self.t,
            "lr": self.lr,
            "beta1": self.beta1,
            "beta2": self.beta2,
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": {k: v.copy() for k, v in self.m.items()},
            "v": {k: v.copy() for k, v in self.v.items()},
        }

    def load_state_dict(self, state: dict) -> None:
        """Loads optimizer state."""
        self.t = state["t"]
        self.lr = state["lr"]
        self.beta1 = state["beta1"]
        self.beta2 = state["beta2"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]
        self.m = {k: v.copy() for k, v in state["m"].items()}
        self.v = {k: v.copy() for k, v in state["v"].items()}
