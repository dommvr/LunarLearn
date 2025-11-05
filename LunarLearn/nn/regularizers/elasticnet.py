from LunarLearn.regularizers.BaseRegularizer import BaseRegularizer
from LunarLearn.regularizers import L1, L2

class ElasticNet(BaseRegularizer):
    """
    Combined L1 + L2 regularization (Elastic Net).

    Loss = λ₁ * ||W||₁ + 0.5 * λ₂ * ||W||₂²
    """
    def __init__(self, lam1=1e-4, lam2=1e-4, combine_mode="override"):
        super().__init__(combine_mode=combine_mode)
        self.l1 = L1(lam1)
        self.l2 = L2(lam2)

    def loss(self, param):
        return self.l1.loss(param) + self.l2.loss(param)