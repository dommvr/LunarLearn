import LunarLearn.backend as backend
from LunarLearn.models import Sequential

xp = backend.xp
DTYPE = backend.DTYPE

class AuxiliaryClassifier:
    def __init__(self, level, head, loss_fn, weight=1.0, scheduler=None):
        self.level = level
        if isinstance(head, list):
            head = Sequential(*head)
        self.head = head
        self.loss_fn = loss_fn
        self.weight = xp.array(weight, dtype=DTYPE)
        self.scheduler = scheduler
    
    def forward(self, activations, y):
        # Get the activation
        if isinstance(activations, dict):
            if isinstance(self.level, int):
                # allow numeric indexing into dict (convert to list)
                key = list(activations.keys())[self.level]
                act = activations[key]
            else:
                act = activations[self.level]
        elif isinstance(activations, list):
            act = activations[self.level]
        else:
            raise TypeError("activations must be a dict or list.")

        # Forward through head
        out = self.head(act)

        # Compute weighted loss
        loss = self.loss_fn(out, y)
        loss *= self.weight
        return loss