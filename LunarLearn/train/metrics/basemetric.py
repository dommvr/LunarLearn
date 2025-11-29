from LunarLearn.core import Tensor
from typing import Any
from abc import ABC, abstractmethod


class BaseMetric(ABC):
    def __init__(self, expect_vector=False):
        self.expect_vector = expect_vector

    def __call__(self, *args, **kwargs) -> Any:
        return self.compute(*args, **kwargs)

    @abstractmethod
    def compute(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    def update(self, preds: Tensor, targets: Tensor, **kwargs):
        result = self.compute(preds, targets, **kwargs)

        if self.expect_vector:
            micro_val, macro_val, weighted_val, per_class_val = result
            self.micro_total += micro_val
            self.macro_total += macro_val
            self.weighted_total += weighted_val

            if self.per_class_total is None:
                self.per_class_total = per_class_val.astype(float)
            else:
                self.per_class_total += per_class_val

        else:
            self.scalar_total += result

        self.count += 1
        return result
    
    def value(self):
        if self.count == 0:
            return 0.0
        
        if self.expect_vector:
            return (
                self.micro_total / self.count,
                self.macro_total / self.count,
                self.weighted_total / self.count,
                self.per_class_total / self.count
            )
        else:
            return self.scalar_total / self.count

    def reset(self):
        self.micro_total = 0.0
        self.macro_total = 0.0
        self.weighted_total = 0.0
        self.scalar_total = 0.0
        self.per_class_total = None
        self.count = 0
