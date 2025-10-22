import os

class ModelCheckpoint:
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True, period=1):
        """
        filepath: str, path to save the checkpoint
        monitor: str, metric name to monitor
        mode: 'min' or 'max', whether lower or higher is better
        save_best_only: save only when metric improves
        period: save every 'period' epochs regardless of improvement
        """
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.period = period
        self.best = None
        self.epochs_since_last_save = 0

        if mode not in ['min', 'max']:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor_op = (lambda a, b: a < b) if mode == 'min' else (lambda a, b: a > b)

    def on_epoch_end(self, epoch, logs, model):
        self.epochs_since_last_save += 1
        save_model = False

        current = logs.get(self.monitor)
        if current is None:
            # fallback: always save if metric not found
            save_model = True
        else:
            if self.best is None:
                self.best = current
                save_model = True
            elif self.monitor_op(current, self.best):
                self.best = current
                save_model = True

        if self.epochs_since_last_save >= self.period:
            save_model = True
            self.epochs_since_last_save = 0

        if save_model:
            #os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
            model.save(self.filepath)  # Delegate saving to the model
            print(f"ModelCheckpoint: saved model at epoch {epoch} to {self.filepath}")

    @staticmethod
    def load(filepath, model):
        """Load checkpoint into model using model.load()"""
        epoch = model.load(filepath)  # Assume load() returns the epoch number
        print(f"ModelCheckpoint: loaded checkpoint from {filepath}, epoch {epoch}")
        return epoch
