import LunarLearn.backend as backend

MIXED_PRECISION = backend.MIXED_PRECISION
SCALING_FACTOR = backend.SCALING_FACTOR

class Trainer:
    def __init__(self, model, optimizer, loss_fn):
        from LunarLearn.schedulers import SchedulerManager
        from LunarLearn.GradientProcessor import GradientProcessor
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.deep_supervision = None
        self.regularizer = None
        self.normalizer = None
        self.history = None
        self.grads_logger = None

        self.scheduler_manager = SchedulerManager(model, optimizer)
        self.scheduler_manager.auto_register_all()

        # Initialize hooks dictionary
        self.hooks = {
            "on_epoch_start": [],
            "on_epoch_end": [],
            "on_batch_start": [],
            "on_batch_end": []
        }

        self.gradient_options = {"clip_norm": None,
                                 "clip_norm_per_layer": None,
                                 "clip_value": None,
                                 "centralize": False,
                                 "accumulation_steps": 1,
                                 "nan_inf_policy": "skip",
                                 "grad_noise_std": None,
                                 "grad_noise_decay": None,
                                 "grad_ema_decay": None}
        self.grad_processor = GradientProcessor(self.gradient_options)
        self.callbacks = []

        self.trained_epochs = 0
        
    def register_hook(self, event, func):
        """
        Register a hook function to a specific event.
        Args:
            event (str): one of 'on_epoch_start', 'on_epoch_end', 'on_batch_start', 'on_batch_end'
            func (callable): function to call. Signature: func(model, **kwargs)
        """
        if event not in self.hooks:
            raise ValueError(f"Unknown hook event '{event}'")
        self.hooks[event].append(func)

    def _run_hooks(self, event, **kwargs):
        for func in self.hooks.get(event, []):
            func(self, **kwargs)

    def register_callbacks(self, callbacks):
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        self.callbacks = callbacks

    def regularization(self, reg):
        """Attach a global regularizer (e.g., model.regularization(L1(1e-4)))."""
        self.regularizer = reg

    def normalization(self, norm):
        """Attach a global normalizer"""
        self.normalizer = norm

    def train(self, train_loader, val_loader=None, epochs=1, train_amp=True, eval_amp=True):
        from LunarLearn.loggers import History
        from LunarLearn.bar import bar
        from LunarLearn.engine import accuracy
        import LunarLearn.amp as amp
        
        self.model.train()
        if not hasattr(self, "history") or self.history is None:
            self.history = History()

        start_epoch = self.trained_epochs

        for epoch in range(start_epoch, epochs):
            # ---------- Training ----------
            self._run_hooks("on_epoch_start", epoch=epoch)
            epoch_loss = 0.0
            progress_bar = bar(len(train_loader), epoch, epochs)

            # Epoch-level scheduler step
            self.scheduler_manager.step(epoch=epoch)

            for i, (X, y) in enumerate(train_loader):
                self._run_hooks("on_batch_start", epoch=epoch, batch=i, X=X, y=y)

                # Batch-level scheduler step
                self.scheduler_manager.step(batch=i)

                # Forward
                with amp.autocast(enabled=train_amp):
                    preds, activations = self.model(X, return_activations=True, activations_mode='dict')
                    main_loss = self.loss_fn(preds, y)

                if self.deep_supervision is not None:
                    # Compute deep supervision loss (already weighted inside)
                    ds_loss, ds_details = self.deep_supervision(activations, y, return_each_loss=True)
                    loss = main_loss + ds_loss
                else:
                    loss = main_loss

                acc = accuracy(preds, y)

                batch_loss = loss.data.item()

                # Regularization
                if self.regularizer is not None:
                    loss += self.regularizer(self.model)

                loss = amp.scale_loss(loss)

                loss.backward()

                grads_ok = self.grad_processor.process(self.model)

                # Gradient accumulation
                if grads_ok and self.grad_processor.step_ready():
                    # Optimizer step
                    amp.step_if_ready(self.optimizer, self.model)

                    # Reset gradients
                    self.optimizer.zero_grad(self.model.parameters())

                epoch_loss += batch_loss

                progress_bar.update(i, acc, batch_loss)

                for cb in self.callbacks:
                    cb.on_batch_end(epoch, i, self.model.parameters(), batch_loss, acc)

                self._run_hooks("on_batch_end", epoch=epoch, batch=i, loss=batch_loss, acc=acc)

            # ---------- Validation ----------
            val_loss, val_acc = (None, None)
            if val_loader is not None:
                val_loss, val_acc = self.evaluate(val_loader, enable_amp=eval_amp)

            self.trained_epochs += 1
            progress_bar.finish(val_acc, val_loss)

            # History logging
            self.history.add(
                epoch=epoch + 1,
                loss=progress_bar.epoch_avg_loss,
                acc=progress_bar.epoch_avg_acc,
                val_loss=val_loss,
                val_acc=val_acc,
                lr=self.optimizer.learning_rate,
                time=progress_bar.epoch_time,
                scale=self.scaler.scale
            )

            self._run_hooks("on_epoch_end", epoch=epoch, val_loss=val_loss, val_acc=val_acc)

            # ---------- Callbacks ----------
            for cb in self.callbacks:
                    cb.on_epoch_end(epoch, i, self.model.parameters(), epoch_loss, acc)

    def evaluate(self, data_loader, enable_amp=True):
        """
        Evaluate the model on a dataset.

        Computes both mean loss and mean accuracy across the data_loader.

        Args:
            data_loader: Iterable of (X, y) batches.

        Returns:
            tuple: (mean_loss, mean_accuracy)
        """
        from LunarLearn.engine import accuracy
        import LunarLearn.amp as amp

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        self.model.eval()
        with backend.no_grad():  # disable grad tracking
            for X, y in data_loader:
                with amp.autocast(enabled=enable_amp):
                    preds = self.model(X)
                    loss = self.loss_fn(preds, y)
                total_loss += loss.data.item()

                # Compute batch accuracy
                acc_batch = accuracy(preds, y)
                total_correct += acc_batch * len(y)
                total_samples += len(y)

        mean_loss = total_loss / len(data_loader)
        mean_acc = total_correct / total_samples
        return mean_loss, mean_acc