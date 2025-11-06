import os
import csv
import json
import threading
import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor

xp = backend.xp
MIXED_PRECISION = backend.MIXED_PRECISION


class GradsLogger:
    """
    Gradient statistics logger for tracking gradient stability, scale, and ratio metrics.

    Supports both standard and mixed-precision (AMP) training. Computes per-layer and
    global gradient metrics such as mean, std, norm, and grad/weight ratio. Can automatically
    save logs to JSON or CSV at epoch or batch intervals.

    Attributes:
        grad_log_mode (str): Logging mode, either "batch" or "epoch".
        grad_log_every (int): Frequency of batch logging (if mode="batch").
        per_layer (bool): Whether to record per-layer gradient statistics.
        autosave (str or None): Format for autosaving logs ("json" or "csv").
        save_path (str): Directory path for saving logs.
        scaler (DynamicLossScaler or None): Optional mixed-precision loss scaler reference.
        records (dict): Logged gradient statistics.
        last_epoch (int): Tracks last epoch index for aggregation.
    """
    def __init__(self, grad_log_mode="epoch", grad_log_every=1,
                 per_layer=True, autosave=None, save_path="grads_logs", scaler=None):
        self.grad_log_mode = grad_log_mode
        self.grad_log_every = grad_log_every
        self.per_layer = per_layer

        self.records = {}
        self.last_epoch = -1

        self.autosave = autosave
        self.save_path = save_path
        self.scaler = scaler  # Optional AMP loss scaler
        if autosave:
            os.makedirs(save_path, exist_ok=True)

    # -------------------------------
    # Core Computations
    # -------------------------------
    def _collect_param_accums(self, param, layer=None):
        """Collect accumulators for a single parameter tensor."""
        g = param.grad
        if g is None:
            return None

        grad_sum = xp.sum(g)
        grad_sq_sum = xp.sum(g * g, dtype=xp.float32)
        count = g.size
        grad_norm_sq = grad_sq_sum
        weight_norm = xp.linalg.norm(param.data)
        ratio = xp.sqrt(grad_norm_sq) / (weight_norm + 1e-12)

        # Handle scaled gradients (for AMP)
        scaled_stats = {}
        if MIXED_PRECISION and self.scaler is not None:
            scaled_grad_norm = float(xp.sqrt(grad_sq_sum)) * self.scaler.scale
            scaled_stats = {
                "scaled_grad_norm": scaled_grad_norm,
                "loss_scale": float(self.scaler.scale)
            }

        return {
            "grad_sum": grad_sum,
            "grad_sq_sum": grad_sq_sum,
            "count": count,
            "grad_norm_sq": grad_norm_sq,
            "weight_norm": weight_norm,
            "ratio": ratio,
            **scaled_stats
        }

    def _finalize_stats(self, acc):
        """Convert accumulators into scalar statistics."""
        grad_mean = acc["grad_sum"] / acc["count"]
        grad_var = acc["grad_sq_sum"] / acc["count"] - grad_mean**2
        grad_std = xp.sqrt(xp.maximum(grad_var, 0.0))
        grad_norm = xp.sqrt(acc["grad_norm_sq"])
        ratio = acc["ratio"]

        stats = {
            "grad_norm": float(grad_norm),
            "grad_mean": float(grad_mean),
            "grad_std": float(grad_std),
            "grad/weight": float(ratio),
        }

        # Include scaled gradient info if available
        if "scaled_grad_norm" in acc:
            stats["scaled_grad_norm"] = acc["scaled_grad_norm"]
            stats["loss_scale"] = acc["loss_scale"]

        return stats

    def _collect_global_accums(self, model):
        """Aggregate accumulators across all parameters for global stats."""
        total_norm_sq = 0.0
        grad_sums, grad_sq_sums, counts, ratios = [], [], [], []
        scaled_norms, loss_scales = [], []

        for param_desc in model.parameters(with_layer=True):
            p = param_desc["param"]
            if p.grad is None:
                continue
            acc = self._collect_param_accums(p)
            if acc is None:
                continue

            grad_sums.append(acc["grad_sum"])
            grad_sq_sums.append(acc["grad_sq_sum"])
            counts.append(acc["count"])
            total_norm_sq += acc["grad_norm_sq"]
            ratios.append(acc["ratio"])

            if "scaled_grad_norm" in acc:
                scaled_norms.append(acc["scaled_grad_norm"])
                loss_scales.append(acc["loss_scale"])

        return {
            "grad_sums": grad_sums,
            "grad_sq_sums": grad_sq_sums,
            "counts": counts,
            "total_norm_sq": total_norm_sq,
            "ratios": ratios,
            "scaled_norms": scaled_norms,
            "loss_scales": loss_scales,
        }

    def _finalize_global_stats(self, acc):
        """Compute final global metrics."""
        if not acc["grad_sums"]:
            return {
                "total_grad_norm": 0.0,
                "grad_mean": 0.0,
                "grad_std": 0.0,
                "grad/weight": 0.0,
                "scaled_grad_norm": 0.0,
                "loss_scale": float(self.scaler.scale) if self.scaler else 0.0,
            }

        grad_sum = xp.sum(xp.array(acc["grad_sums"]))
        grad_sq_sum = xp.sum(xp.array(acc["grad_sq_sums"]))
        total_count = xp.sum(xp.array(acc["counts"]))

        grad_mean = grad_sum / total_count
        grad_var = grad_sq_sum / total_count - grad_mean**2
        grad_std = xp.sqrt(xp.maximum(grad_var, 0.0))

        stats = {
            "total_grad_norm": float(xp.sqrt(acc["total_norm_sq"])),
            "grad_mean": float(grad_mean),
            "grad_std": float(grad_std),
            "grad/weight": float(xp.mean(xp.array(acc["ratios"]))),
        }

        # Add AMP metrics if available
        if acc["scaled_norms"]:
            stats["scaled_grad_norm"] = float(xp.mean(xp.array(acc["scaled_norms"])))
            stats["loss_scale"] = float(xp.mean(xp.array(acc["loss_scales"])))

        return stats

    # -------------------------------
    # Logging
    # -------------------------------
    def add(self, model, epoch, batch, n_batches):
        """
        Collect and log gradient statistics for the given model.

        Args:
            model: Model exposing `parameters(with_layer=True)`.
            epoch (int): Current epoch number.
            batch (int): Current batch index.
            n_batches (int): Total batches per epoch.
        """
        params = model.parameters(with_layer=True)

        if self.grad_log_mode == "epoch":
            if epoch > self.last_epoch:
                self.epoch_accums = {"params": [], "global": None}
                self.last_epoch = epoch

            if self.per_layer:
                if not self.epoch_accums["params"]:
                    self.epoch_accums["params"] = [{} for _ in params]

                for i, param_desc in enumerate(params):
                    p = param_desc["param"]
                    acc = self._collect_param_accums(p)
                    if acc is None:
                        continue
                    for k, v in acc.items():
                        self.epoch_accums["params"][i][k] = (
                            self.epoch_accums["params"][i].get(k, 0.0) + v
                        )

            global_acc = self._collect_global_accums(model)
            if self.epoch_accums["global"] is None:
                self.epoch_accums["global"] = global_acc
            else:
                for k, v in global_acc.items():
                    if isinstance(v, list):
                        self.epoch_accums["global"][k].extend(v)
                    else:
                        self.epoch_accums["global"][k] += v

            if batch + 1 == n_batches:
                averaged = {}
                if self.per_layer:
                    for i, acc in enumerate(self.epoch_accums["params"]):
                        if acc:
                            stats = self._finalize_stats(acc)
                            for k, v in stats.items():
                                averaged[f"param_{i}_{k}"] = v

                global_stats = self._finalize_global_stats(self.epoch_accums["global"])
                averaged.update(global_stats)

                self.records[f"Epoch_{epoch}"] = averaged
                self._autosave_async()

        elif self.grad_log_mode == "batch":
            if self.grad_log_every is None or batch % self.grad_log_every == 0:
                batch_records = {}
                if self.per_layer:
                    for i, param_desc in enumerate(params):
                        p = param_desc["param"]
                        acc = self._collect_param_accums(p)
                        if acc is None:
                            continue
                        stats = self._finalize_stats(acc)
                        for k, v in stats.items():
                            batch_records[f"param_{i}_{k}"] = v

                global_acc = self._collect_global_accums(model)
                global_stats = self._finalize_global_stats(global_acc)
                batch_records.update(global_stats)

                if f"Epoch_{epoch}" not in self.records:
                    self.records[f"Epoch_{epoch}"] = {}
                self.records[f"Epoch_{epoch}"][f"batch_{batch}"] = batch_records
                self._autosave_async()

    # -------------------------------
    # Export
    # -------------------------------
    def to_json(self, filepath):
        """Save all logged statistics to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.records, f, indent=4)

    def to_csv(self, filepath):
        """Save flattened log data to CSV."""
        flat_records = []
        for epoch, data in self.records.items():
            if isinstance(data, dict) and any("batch_" in k for k in data):
                for batch, vals in data.items():
                    row = {"epoch": epoch, "batch": batch}
                    row.update(vals)
                    flat_records.append(row)
            else:
                row = {"epoch": epoch}
                row.update(data)
                flat_records.append(row)

        keys = sorted({k for row in flat_records for k in row.keys()})
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(flat_records)

    def _autosave(self):
        """Automatically save logs in background if autosave is enabled."""
        if self.autosave == "json":
            filepath = os.path.join(self.save_path, "grads_logs.json")
            self.to_json(filepath)
        elif self.autosave == "csv":
            filepath = os.path.join(self.save_path, "grads_logs.csv")
            self.to_csv(filepath)

    def _autosave_async(self):
        """Run autosave in a non-blocking background thread."""
        threading.Thread(target=self._autosave, daemon=True).start()