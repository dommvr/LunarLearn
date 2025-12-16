import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader import IterableDataset
from LunarLearn.data.dataloader.utils import _to_tensor_tree
from LunarLearn.data.dataloader.collate import _collate

xp = backend.xp
DTYPE = backend.DTYPE


class DataLoader:
    """
    Unified mini-batch data loader supporting both map-style and iterable datasets,
    plus arbitrary sample structures via collate_fn.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, to_tensor=True, collate_fn=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = shuffle
        self.to_tensor = to_tensor
        self.collate_fn = collate_fn if collate_fn is not None else _collate

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

    def __iter__(self):
        if self.is_iterable:
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    out = self.collate_fn(batch)
                    if self.to_tensor:
                        out = _to_tensor_tree(out)
                    yield out
                    batch = []
            if batch:
                out = self.collate_fn(batch)
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
                if self.to_tensor:
                    out = _to_tensor_tree(out)
                yield out