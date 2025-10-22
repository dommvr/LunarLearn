class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', patience=5, min_delta=0.0, restore_best_weights=True):
        """
        monitor: str, metric to monitor (e.g., 'val_loss')
        mode: 'min' or 'max' (lower is better for loss, higher is better for accuracy)
        patience: int, number of epochs with no improvement after which training stops
        min_delta: float, minimum change to qualify as improvement
        restore_best_weights: bool, whether to restore the best model at the end
        """
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights

        self.best = None
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.best_weights_path = None

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor_op = (lambda a, b: a < b - self.min_delta) if mode == 'min' else (lambda a, b: a > b + self.min_delta)

    def on_epoch_end(self, epoch, logs, model):
        current = logs.get(self.monitor)
        if current is None:
            return  # nothing to monitor

        if self.best is None or self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
            if self.restore_best_weights:
                # Save current model as best
                self.best_weights_path = model.save("_early_stopping_best.pkl")  # temporary save
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stop_training = True
                print(f"EarlyStopping: stopping at epoch {epoch} (no improvement in {self.patience} epochs)")
                if self.restore_best_weights and self.best_weights_path:
                    model.load(self.best_weights_path)
                    print("EarlyStopping: restored best model weights")

    def on_train_end(self, epoch, logs, model):
        if self.stop_training:
            return True  # signal to stop training
        return False
