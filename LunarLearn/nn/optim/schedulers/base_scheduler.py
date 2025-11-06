import LunarLearn.core.backend.backend as backend

xp = backend.xp
DTYPE = backend.DTYPE

class BaseScheduler:
    """
        Universal base scheduler for optimizers, layers, etc.

        Args:
            target: Any object whose attribute should be scheduled
                    (e.g. optimizer, DropPath layer, etc.)
            attr_name (str): Name of the attribute to update (e.g. "learning_rate", "keep_prob")
            mode (str): 'epoch' or 'step'
    """
    def __init__(self, target, attr_name: str, mode='epoch'):
        self.target =target
        self.attr_name = attr_name
        self.mode = mode
        self.epoch = 0
        self.last_step = 0

        self.initial_value = getattr(target, attr_name)

    def get_config(self):
        from LunarLearn.engine import serialize_value

        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "params": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k != "optimizer"
            }
        }
    
    @classmethod
    def from_config(cls, config, optimizer=None):
        from LunarLearn.engine import object_from_config
        import importlib

        module = importlib.import_module(config["module"])
        klass = getattr(module, config["class"])

        # Merge init params and optimizer if needed
        init_args = dict(config.get("params", {}))
        if optimizer is not None:
            init_args["optimizer"] = optimizer

        # Initialize without optimizer
        obj = klass(**init_args)

        # Set extra attributes
        for k, v in config.get("extra", {}).items():
            setattr(obj, k, v)

        return obj
    
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