class Stateful:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass

    def get_config(self):
        return {}

    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)