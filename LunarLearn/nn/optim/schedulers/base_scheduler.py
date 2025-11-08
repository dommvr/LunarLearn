import LunarLearn.core.backend.backend as backend
from LunarLearn.nn import Stateful

xp = backend.xp
DTYPE = backend.DTYPE

class BaseScheduler(Stateful):
    """
        Universal base scheduler for optimizers, layers, etc.

        Args:
            target: Any object whose attribute should be scheduled
                    (e.g. optimizer, DropPath layer, etc.)
            attr_name (str): Name of the attribute to update (e.g. "learning_rate", "keep_prob")
            mode (str): 'epoch' or 'step'
    """
    def __init__(self, target, attr_name: str, mode='epoch'):
        self.target = target
        self.attr_name = attr_name
        self.mode = mode
        self.epoch = 0
        self.last_step = 0

        self.initial_value = getattr(target, attr_name)

    def state_dict(self):
        out = {
            "epoch": self.epoch,
            "last_step": self.last_step,
            "initial_value": self.initial_value
        }

        for k, v in self.__dict__.items():
            if k in ("target", "attr_name", "mode"):
                continue
            if isinstance(v, (int, float, str, bool, xp.ndarray)):
                out[k] = v

        return out
    
    def load_state_dict(self, state):
        for name, value in state.items():
            if hasattr(self, name):
                setattr(self, name, value)
    
    def get_value(self):
        return getattr(self.target, self.attr_name)
    
    def step(self, step=None):
        if self.mode == "epoch":
            self.epoch = step if step is not None else self.epoch + 1
            return self.epoch
        else:
            self.last_step = step if step is not None else self.last_step + 1
            return self.last_step

    def set_new_value(self, new_value):
        setattr(self.target, self.attr_name, xp.array(new_value, dtype=DTYPE))