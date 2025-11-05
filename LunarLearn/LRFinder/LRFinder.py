import copy
import math
import LunarLearn.amp as amp
import LunarLearn.backend as backend

xp = backend.xp

class LRFinder:
    def __init__(self, model, optimizer, loss_fn, loader,
                 lr_min=1e-8, lr_max=1, num_iters=None, with_amp=True):
        self.model = copy.deepcopy(model)
        self.optimizer = copy.deepcopy(optimizer)
        self.loss_fn = loss_fn
        self.loader = copy.deepcopy(loader)
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.num_iters = num_iters or len(loader)
        self.with_amp = with_amp
        self.history = {"lr": [] , "loss": []}

        self.model_state = copy.deepcopy(model.state_dict())
        self.opt_state = copy.deepcopy(optimizer.state_dict())

    def find(self):
        setattr(self.optimizer, "learning_rate", self.lr_min)
        prev_loss = float("inf")
        for i, (X, y) in enumerate(self.loader):
            with amp.autocast(enabled=self.with_amp):
                preds = self.model(X, return_activations=False)
                loss = self.loss_fn(preds, y)
                batch_loss = loss.data.item()
                if batch_loss * 4 > prev_loss:
                    break
                loss = amp.scale_loss(loss)
                loss.backward()
                amp.step_if_ready(self.optimizer, self.model)
                self.optimizer.zero_grad()

                self.stats[batch_loss] = self.optimizer.learning_rate

                new_lr = self.lr_min * xp.exp(-self.decay_constant * i)
                setattr(self.optimizer, "learning_rate", new_lr)
                prev_loss = batch_loss

    def suggested(self, multiplyer=0.1):
        best_lr = self.stats[min(self.stats.keys())]
        return best_lr * multiplyer