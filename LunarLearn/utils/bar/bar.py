import sys
import time

class bar:
    """
    Lightweight, dynamic console progress bar for training visualization.

    Displays per-batch progress, training metrics, and timing information
    during each epoch. Automatically adapts to datasets with or without
    known length (e.g., generator-based datasets).

    Attributes:
        total_batches (int or None): Total number of batches in the epoch.
            If None, progress visualization (bar/percent) is disabled, and
            only batch-level info is printed.
        epoch (int): Current epoch (1-indexed).
        epochs (int): Total number of epochs in the training run.
        start_time (float): Timestamp marking the start of the current epoch.
        train_acc_sum (float): Cumulative training accuracy over processed batches.
        train_loss_sum (float): Cumulative training loss over processed batches.
        BAR_LEN (int): Length of the visual progress bar (number of symbols).
        has_progress (bool): Whether to display a progress bar (depends on total_batches).

    Methods:
        update(batch_n, acc, loss):
            Updates and redraws the progress bar for the current batch.
            Displays training accuracy, loss, speed, and ETA if possible.

        finish(val_acc, val_loss):
            Finalizes the epoch summary, clears the bar line, and prints
            aggregated training and validation metrics with timing stats.
    """
    def __init__(self, total_batches, epoch, epochs):
        self.total_batches = total_batches
        self.epoch = epoch + 1
        self.epochs = epochs
        self.start_time = time.time()
        self.train_acc_sum = 0
        self.train_loss_sum = 0
        self.BAR_LEN = 20
        self.has_progress = total_batches is not None  # <-- key flag

    def update(self, batch_n, acc, loss):
        if hasattr(acc, "item"):
            acc = float(acc.item())
        if hasattr(loss, "item"):
            loss = float(loss.item())

        self.train_acc_sum += acc
        self.train_loss_sum += loss
        inv_denom = 1.0 / (batch_n + 1)
        avg_acc = self.train_acc_sum * inv_denom
        avg_loss = self.train_loss_sum * inv_denom

        elapsed = time.time() - self.start_time
        steps_per_sec = (batch_n + 1) / elapsed
        ms_per_step = 1000 / steps_per_sec if steps_per_sec > 0 else 0

        # Skip progress bar if total_batches unknown
        if not self.has_progress:
            sys.stdout.write(
                f"\rEpoch {self.epoch}/{self.epochs} "
                f"Batch {batch_n + 1} - {elapsed:.1f}s {ms_per_step:.1f}ms/step - "
                f"acc: {avg_acc:.4f} - loss: {avg_loss:.4f}"
            )
            sys.stdout.flush()
            return

        progress = (batch_n + 1) / self.total_batches
        filled = int(self.BAR_LEN * progress)
        progress_bar = '#' * filled + '=' * (self.BAR_LEN - filled)

        sys.stdout.write(
            f"\rEpoch {self.epoch}/{self.epochs} "
            f"{batch_n + 1}/{self.total_batches} "
            f"[{progress_bar}] - {elapsed:.1f}s {ms_per_step:.1f}ms/step - "
            f"acc: {avg_acc:.4f} - loss: {avg_loss:.4f}"
        )
        sys.stdout.flush()

    def finish(self, val_acc, val_loss):
        if hasattr(val_acc, "item"):
            val_acc = float(val_acc.item())
        if hasattr(val_loss, "item"):
            val_loss = float(val_loss.item())

        self.epoch_time = time.time() - self.start_time

        # Compute safe averages (avoid div by None)
        if self.total_batches:
            self.epoch_avg_acc = self.train_acc_sum / self.total_batches
            self.epoch_avg_loss = self.train_loss_sum / self.total_batches
        else:
            self.epoch_avg_acc = self.train_acc_sum
            self.epoch_avg_loss = self.train_loss_sum

        val_info = ""
        if val_acc is not None and val_loss is not None:
            val_info = f"val_acc: {val_acc:.4f} - val_loss: {val_loss:.4f}"

        sys.stdout.write('\r' + ' ' * 120 + '\r')
        sys.stdout.flush()

        print(
            f"Epoch {self.epoch}/{self.epochs} - "
            f"epoch_time: {self.epoch_time:.1f}s - "
            f"acc: {self.epoch_avg_acc:.4f} - loss: {self.epoch_avg_loss:.4f} - {val_info}\n"
        )
        