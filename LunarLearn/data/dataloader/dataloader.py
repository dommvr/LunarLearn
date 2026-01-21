import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader import IterableDataset
from LunarLearn.data.dataloader.utils import _to_tensor_tree, compute_channel_mean_std, to_backend
from LunarLearn.data.dataloader.collate import _collate

xp = backend.xp
DTYPE = backend.DTYPE


class DataLoader:
    """
    Unified mini-batch data loader supporting both map-style and iterable datasets,
    plus arbitrary sample structures via collate_fn.
    """
    def __init__(self,
                 dataset,
                 batch_size=32,
                 shuffle=True,
                 to_tensor=True,
                 collate_fn=None,
                 normalize=False,
                 stats_from=None,
                 mean=None,
                 std=None,
                 channel_dim=1,
                 eps=1e-8,
                 normalize_key=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.to_tensor = to_tensor
        self.collate_fn = collate_fn if collate_fn is not None else _collate

        self.normalize = normalize
        self.mean, self.std = mean, std
        self.channel_dim = channel_dim
        self.eps = eps
        self.normalize_key = normalize_key

        if normalize:
            if (self.mean is None) or (self.std is None):
                if stats_from is not None:
                    self.mean, self.std = compute_channel_mean_std(stats_from)
                else:
                    self.mean, self.std = compute_channel_mean_std(self.dataset)

        self.is_iterable = isinstance(dataset, IterableDataset)

        if not self.is_iterable:
            self.m = len(dataset)
            self.n_batches = (self.m + self.batch_size - 1) // self.batch_size

    def __len__(self):
        if self.is_iterable:
            # Only valid if dataset provides a real length
            try:
                m = len(self.dataset)
            except Exception:
                raise TypeError("Iterable dataset length is unknown; __len__ is not available.")
            return (m + self.batch_size - 1) // self.batch_size

        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def _get_indices(self):
        if self.shuffle:
            return xp.random.permutation(self.m)
        return xp.arange(self.m, dtype=xp.int64)
    
    def _normalize_array(self, x):
        # Only normalize if requested
        if self.mean is None or self.std is None:
            return x

        # Ignore non-array-ish stuff
        if not hasattr(x, "shape"):
            return x

        # Cast to float for sane normalization
        x = x.astype(xp.float32, copy=False)

        mean = xp.asarray(self.mean, dtype=xp.float32)
        std = xp.asarray(self.std, dtype=xp.float32)

        # Broadcast mean/std along all dims except channel_dim
        nd = x.ndim
        cd = self.channel_dim if self.channel_dim >= 0 else (nd + self.channel_dim)
        shape = [1] * nd
        shape[cd] = mean.size

        mean = mean.reshape(shape)
        std = std.reshape(shape)

        return (x - mean) / (std + self.eps)

    def _normalize_batch(self, out):
        """
        Apply normalization to the "input" part of the batch.
        Common cases:
          - out is just x
          - out is (x, y)
          - out is dict with a known key (normalize_key)
        """
        if not self.normalize:
            return out
        
        if self.mean is None or self.std is None:
            return out

        # dict-like
        if self.normalize_key is not None and isinstance(out, dict) and self.normalize_key in out:
            out = dict(out)
            out[self.normalize_key] = self._normalize_array(out[self.normalize_key])
            return out

        # tuple/list (assume first item is input)
        if isinstance(out, (tuple, list)) and len(out) >= 1:
            x0 = self._normalize_array(out[0])
            if isinstance(out, tuple):
                return (x0, *out[1:])
            out = list(out)
            out[0] = x0
            return out

        # single tensor/array
        return self._normalize_array(out)

    def __iter__(self):
        if self.is_iterable:
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    out = self.collate_fn(batch)
                    out = to_backend(out)
                    out = self._normalize_batch(out)
                    if self.to_tensor:
                        out = _to_tensor_tree(out)
                    yield out
                    batch = []
            if batch:
                out = self.collate_fn(batch)
                out = to_backend(out)
                out = self._normalize_batch(out)
                if self.to_tensor:
                    out = _to_tensor_tree(out)
                yield out

        else:
            indices = self._get_indices()
            for start in range(0, self.m, self.batch_size):
                end = min(start + self.batch_size, self.m)
                batch_idx = indices[start:end]

                # IMPORTANT: cupy scalars aren't python ints
                batch = [self.dataset[int(i)] for i in batch_idx]

                out = self.collate_fn(batch)
                out = to_backend(out)
                out = self._normalize_batch(out)
                if self.to_tensor:
                    out = _to_tensor_tree(out)
                yield out