import copy
import math
import LunarLearn.amp as amp
import numpy as np
import LunarLearn.backend as backend

xp = backend.xp

class LRFinder:
    def __init__(self, model, optimizer, loss_fn, loader,
                 lr_min=1e-8, lr_max=1, num_iters=None, with_amp=True,
                 beta=0.98, increase_threshold=5.0,
                 check_grad=True, grad_increase_threshold=10.0):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.loader = loader
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_iters = num_iters or len(loader)
        self.with_amp = with_amp
        self.beta = float(beta)
        self.increase_threshold = float(increase_threshold)
        self.check_grad = check_grad
        self.grad_increase_threshold = grad_increase_threshold

        self.model_state = copy.deepcopy(model.state_dict())
        self.opt_state = copy.deepcopy(optimizer.state_dict())

        self.reset_state()

    def reset_state(self):
        self.smooth_loss = 0.0
        self.best_loss = float("inf")
        self.grad_norm = 0.0
        self.history = {"lr": [], "loss": []}

    def _check_loss(self, loss, step):
        # Compute smoothed loss (EMA)
        cur_smooth = (self.beta * self.smooth_loss + (1 - self.beta) * loss) / (1 - self.beta**(step+1)) if step > 0 else loss
        self.smooth_loss = cur_smooth

        # Update best loss
        if cur_smooth < self.best_loss:
            self.best_loss = cur_smooth
        
        # Check for numerical or explosive growth
        if (
            not math.isfinite(cur_smooth)
            or (cur_smooth / self.best_loss > self.increase_threshold and cur_smooth > 1.5 * self.smooth_loss)
        ):
            return False
        
        # Optional: check for gradient explosion
        if self.check_grad:
            g_norm = math.sqrt(sum(float(xp.sum(p.grad ** 2)) for p in self.model.parameters() if p.grad is not None))
            if self.grad_norm == 0:
                self.grad_norm = g_norm
            elif g_norm / self.grad_norm > self.grad_increase_threshold:
                return False
        
        return True

    def find(self):
        self.reset_state()
        decay = math.log(self.lr_max / self.lr_min) / self.num_iters
        lr = self.lr_min

        for i, (X, y) in enumerate(self.loader):
            if i >= self.num_iters:
                break

            self.optimizer.learning_rate = lr

            with amp.autocast(enabled=self.with_amp):
                preds = self.model(X)
                loss = self.loss_fn(preds, y)

            batch_loss = loss.data.item()
            loss = amp.scale_loss(loss)

            if not self._check_loss(batch_loss, i):
                print(f"Loss exploded at step {i}, lr={lr:.2e}")
                break

            loss.backward()
            amp.step_if_ready(self.optimizer, self.model)
            self.optimizer.zero_grad(self.model.parameters())

            self.history["lr"].append(lr)
            self.history["loss"].append(batch_loss)

            lr *= math.exp(decay)

        self.model.load_state_dict(self.model_state)
        self.optimizer.load_state_dict(self.opt_state)

    def suggested(self, multiplier=0.1):
        losses = np.array(self.history["loss"])
        lrs = np.array(self.history["lr"])
        best_idx = np.argmin(losses)
        best_lr = lrs[best_idx]
        return best_lr * multiplier
    
    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(self.history["lr"], self.history["loss"])
        plt.xscale("log")
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        plt.title("LR Finder Curve")
        plt.show()