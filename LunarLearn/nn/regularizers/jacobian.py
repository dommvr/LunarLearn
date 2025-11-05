import LunarLearn.backend as backend
from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer
from LunarLearn.tensor import ops
from LunarLearn.tensor import Tensor

xp = backend.xp


class JacobianRegularizer(BaseRegularizer):
    """
    Penalizes the Jacobian norm of the model output w.r.t. its input.

    This encourages smoother mappings and robustness to input perturbations.
    It's similar to the contractive regularizer but computed via autograd.

    Loss = lam * ||∂y/∂x||_F^2

    Example:
        model.regularizer = JacobianRegularizer(lam=1e-3)
    """

    def __init__(self, lam=1e-3, num_samples=None, combine_mode="override"):
        """
        Args:
            lam (float): regularization strength.
            num_samples (int, optional): if provided, samples subset of output dims to reduce compute.
        """
        super().__init__(combine_mode=combine_mode)
        self.lam = lam
        self.num_samples = num_samples

    def loss(self, param):
        """No direct parameter-level penalty."""
        return Tensor(0.0)

    def __call__(self, model):
        """
        Compute Jacobian regularization loss for the given model.

        Requires that model keeps track of the last (input, output) pair:
            model.last_input  (Tensor)
            model.last_output (Tensor)
        """
        if not hasattr(model, "last_input") or not hasattr(model, "last_output"):
            return Tensor(0.0)

        x = model.last_input
        y = model.last_output

        if not (x.requires_grad and y.requires_grad):
            return Tensor(0.0)

        # Optionally subsample output dimensions for efficiency
        if self.num_samples is not None and y.shape[1] > self.num_samples:
            idx = xp.random.choice(y.shape[1], self.num_samples, replace=False)
            y = y[:, idx]

        # Compute gradients dy/dx for each output dimension
        batch_size, out_dim = y.shape
        total_norm = 0.0

        for i in range(out_dim):
            grad_y_i = ops.grad(y[:, i].sum(), x, retain_graph=True)
            total_norm += ops.sum(grad_y_i ** 2)

        total_norm /= batch_size

        return self.lam * total_norm
