class LRFinder:
    def __init__(self, model, optimizer, loader, lr_min=1e-8, lr_max=1):
        self.model = model
        self.optimizer = optimizer
        self.loader = loader
        self.lr_min = lr_min
        self.lr_max = lr_max

    def find(self):
        setattr(self.optimizer, "learning_rate", self.lr_min)

        for i, (X, y) in enumerate(self.loader):
            