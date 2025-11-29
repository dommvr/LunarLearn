import LunarLearn.core.backend.backend as backend
from LunarLearn.train.metrics import BaseMetric
from LunarLearn.train.metrics import _inception_score_split
from LunarLearn.core import Tensor

xp = backend.xp


class InceptionScoreSplit(BaseMetric):
    def __init__(self, num_splits: int = 10, eps: float = 1e-12):
        # we don't actually use micro/macro/weighted/per_class here
        super().__init__(expect_vector=False)
        self.num_splits = num_splits
        self.eps = eps
        self._logits_list = []

    def reset(self):
        super().reset()
        self._logits_list = []

    def update(self, preds: Tensor, targets: Tensor = None, **kwargs):
        # here preds are actually logits from classifier, targets unused
        self._logits_list.append(preds)
        self.count += 1

    def compute(self, logits: Tensor, **kwargs):
        # direct one-shot compute if user passes all logits manually
        return _inception_score_split(logits, num_splits=self.num_splits, eps=self.eps)

    def value(self):
        if not self._logits_list:
            return 0.0, 0.0, xp.array([], dtype=xp.float32)

        # concatenate along batch dimension
        logits = Tensor(xp.concatenate([t.data for t in self._logits_list], axis=0))
        mean_is, std_is, split_scores = _inception_score_split(
            logits, num_splits=self.num_splits, eps=self.eps
        )
        return mean_is, std_is, split_scores