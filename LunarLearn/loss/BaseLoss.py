class BaseLoss:
    def __init__(self, trainable=False):
        self.trainable=trainable

    def get_config(self):
        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "params": {}
        }

    @classmethod
    def from_config(cls, config):
        import importlib

        # Import the module and class
        module = importlib.import_module(config["module"])
        klass = getattr(module, config["class"])

        # Initialize object with params
        obj = klass(**config.get("params", {}))

        # Set extra attributes after init
        for k, v in config.get("extra", {}).items():
            setattr(obj, k, v)

        return obj
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        raise NotImplementedError