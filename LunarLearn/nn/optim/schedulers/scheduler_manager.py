class SchedulerManager:
    def __init__(self, model=None, optimizer=None, deep_supervision=None):
        self.schedulers = []
        self.model = model
        self.optimizer = optimizer
        self.deep_supervision = deep_supervision

    def register(self, scheduler):
        """Register a scheduler manually."""
        self.schedulers.append(scheduler)

    def auto_register_layers(self):
        """Automatically find layers with .scheduler attribute and register them."""
        for layer in self.model.layers():
            if hasattr(layer, "scheduler") and layer.scheduler is not None:
                self.schedulers.append(layer.scheduler)

    def auto_register_layers_optim(self):
        for layer in self.model.layers():
            if hasattr(layer, "optimizer") and layer.optimizer.scheduler is not None:
                self.schedulers.append(layer.optimizer.scheduler)

    def auto_register_optimizer(self):
        """Automatically register optimizer scheduler if attached."""
        if hasattr(self.optimizer, "scheduler") and self.optimizer.scheduler is not None:
            self.schedulers.append(self.optimizer.scheduler)

    def auto_register_deep_supervision(self):
        for clas in self.deep_supervision.classifiers:
            if hasattr(clas, "scheduler") and clas.scheduler is not None:
                self.schedulers.append(clas.scheduler)

    def auto_register_params(self):
        for param in self.model.parameters():
            if getattr(param, "scheduler") and param.scheduler is not None:
                self.schedulers.append(param.scheduler)

    def auto_register_params_optim(self):
        for param in self.model.parameters():
            if getattr(param, "optimizer") and param.optimizer.scheduler is not None:
                self.schedulers.append(param.optimizer.scheduler)

    def auto_register_all(self):
        if self.model is not None:
            self.auto_register_layers()
            self.auto_register_layers_optim()
            self.auto_register_params()
            self.auto_register_params_optim()
        if self.optimizer is not None:
            self.auto_register_optimizer()
        if self.deep_supervision is not None:
            self.auto_register_deep_supervision()

    def step(self, epoch=None, batch=None):
        """
        Advance schedulers for the current training step.

        - Call with epoch != None to update only epoch-based schedulers.
        - Call with batch != None to update only batch-based schedulers.
        """
        if epoch is not None:
            # Step only epoch-based schedulers
            for sched in self.schedulers:
                if sched.mode == "epoch":
                    sched.step(epoch)

        if batch is not None:
            # Step only batch-based schedulers
            for sched in self.schedulers:
                if sched.mode == "batch":
                    sched.step(batch)