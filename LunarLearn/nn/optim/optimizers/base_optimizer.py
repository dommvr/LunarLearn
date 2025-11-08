import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful
from LunarLearn.core import Parameter
import copy

xp = backend.xp
DTYPE = backend.DTYPE

class BaseOptimizer(Stateful):
    def __init__(self, learning_rate):
        self.learning_rate = xp.array(learning_rate, dtype=DTYPE)

        self.scheduler = None
        self.enable_weight_decay = False

    def state_dict(self):
        out = {
            "learning_rate": self.learning_rate,
            "enable_weight_decay": self.enable_weight_decay
        }

        if self.scheduler is not None:
            out["scheduler"] = self.scheduler.state_dict()

        if getattr(self, "weight_decay", None) is not None:
            out["weight_decay"] = self.weight_decay
        if getattr(self, "t", None) is not None:
            out["t"] = self.t
        if getattr(self, "state", None) is not None:
            out["state"] = copy.deepcopy(self.state)

        return out

    def load_state_dict(self, state):
        if "learning_rate" in state:
            self.learning_rate = xp.array(state["learning_rate"], dtype=self.learning_rate.dtype)
        if "weight_decay" in state:
            if hasattr(self, "weight_decay"):
                self.weight_decay = state["weight_decay"]
        if "enable_weight_decay" in state:
            self.enable_weight_decay = state["enable_weight_decay"]
        if "t" in state:
            if hasattr(self, "t"):
                self.t = state["t"]
        if "state" in state:
            if hasattr(self, "state"):
                self.state = copy.deepcopy(state["state"])
        if "scheduler" in state and self.scheduler is not None:
            self.scheduler.load_state_dict(state["scheduler"])
    
    def _iter_params(self, params):
        """Yield trainable Parameter objects from a params list."""
        for desc in params:
            if isinstance(desc, dict):
                param = desc["param"]
                layer = desc.get("layer", None)
            else:
                param = desc
                layer = None

            if not isinstance(param, Parameter):
                continue
            if param.grad is None:
                continue

            # Skip frozen parameters
            if getattr(param, "frozen", False):
                continue
            if layer is not None and getattr(layer, "frozen", False):
                continue

            # Priority: per-param optimizer > per-layer optimizer > global optimizer
            custom_optim = param.optimizer or (layer.optimizer if layer else None)

            yield param, layer, custom_optim
    
    def _get_lr(self, param, layer=None):
        """
        Compute the effective learning rate for a parameter.

        Priority:
        1. param.base_lr > layer.base_lr > global LR
        2. param.lr_scale > layer.lr_scale > 1.0
        """
        # 1. Base LR priority chain
        if getattr(param, "base_lr", None) is not None:
            lr = param.base_lr
        elif layer is not None and getattr(layer, "base_lr", None) is not None:
            lr = layer.base_lr
        else:
            lr = self.learning_rate

        # 2. Scaling (multiplicative)
        lr *= getattr(param, "lr_scale", 1.0)
        if layer is not None:
            lr *= getattr(layer, "lr_scale", 1.0)

        return lr
    
    def _apply_weight_decay(self, param, layer=None, lr=None):
        """
        Applies decoupled weight decay to a parameter if configured.
        Priority: param.weight_decay > layer.weight_decay > optimizer.weight_decay
        """
        wd = 0.0

        if getattr(param, "decay_exempt", False):
            return
        if getattr(param, "frozen", False):
            return
        if layer is not None:
            if getattr(layer, "decay_exempt", False):
                return
            if getattr(layer, "frozen", False):
                return
        
        if not self.enable_weight_decay:
            return

        # 1. Per-param setting
        if getattr(param, "weight_decay", None) is not None:
            wd = param.weight_decay

        # 2. Per-layer setting
        elif layer is not None and getattr(layer, "weight_decay", None) is not None:
            wd = layer.weight_decay

        # 3. Global optimizer default
        elif hasattr(self, "weight_decay"):
            wd = self.weight_decay

        # If no weight decay, bail out
        if wd == 0.0:
            return
        
        wd *= getattr(param, "weight_decay_scale", 1.0)
        if layer is not None:
            wd *= getattr(layer, "weight_decay_scale", 1.0)

        # Resolve LR if not passed
        if lr is None:
            lr = self._get_lr(param, layer)

        # Decoupled weight decay update
        param.data -= lr * wd * param.data

    def step(self, params):
        """Update parameters based on their gradients."""
        raise NotImplementedError

    def zero_grad(self, params):
        """Set gradients to zero."""
        for p in params:
            if isinstance(p, Parameter) and p.requires_grad:
                p.grad = None