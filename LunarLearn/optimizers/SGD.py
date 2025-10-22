import LunarLearn.backend as backend
from LunarLearn.optimizers.BaseOptimizer import BaseOptimizer
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE

class SGD(BaseOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    This is the simplest optimizer which updates parameters in the opposite 
    direction of the gradient, scaled by the learning rate. It does not use 
    momentum or adaptive learning rates.

    Args:
        learning_rate (float, optional): 
            Step size for parameter updates. Default is 0.01.

    Methods:
        step(params: List[Union[Tensor, dict]]):
            Perform one optimization step over a list of parameters. Supports
            both raw Tensors and dictionaries containing {"param": Tensor, "layer": Layer}.
    """
    def __init__(self, learning_rate=0.01):
        super().__init__(learning_rate)

    def step(self, params):
        for param_desc in params:
            # --- Extract param & layer if available ---
            if isinstance(param_desc, dict):
                p = param_desc["param"]
                layer = param_desc.get("layer", None)
            else:
                p = param_desc
                layer = None

            if not isinstance(p, Tensor) or not p.requires_grad:
                continue
            if p.grad is None:
                continue

            grad = p.grad
            lr = self._get_lr(param_desc)

            # Parameter update
            p.data -= lr * grad
