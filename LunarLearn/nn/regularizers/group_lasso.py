from LunarLearn.nn.regularizers import BaseRegularizer

class GroupLasso(BaseRegularizer):
    """
    Apply a custom user-defined penalty function to gradients or parameters.

    Example:
        reg = GradientHookRegularizer(lambda p: 1e-4 * ops.sum(p.grad ** 2))
        total_loss = loss + reg(param)
    """
    def __init__(self, func, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.func = func

    def loss(self, param):
        # If user-defined func expects full tensor
        return self.func(param)