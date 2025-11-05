import LunarLearn.backend as backend
from LunarLearn.dataloader.datasets import IterableDataset
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE


class DataLoader:
    """
    Unified mini-batch data loader supporting both static and iterable datasets.

    Works seamlessly with standard array-based datasets or custom IterableDatasets,
    automatically batching, shuffling, and optionally converting data to Tensors.
    Designed for compatibility with both CPU and GPU backends.

    Attributes:
        dataset (Dataset or IterableDataset): The dataset providing (X, y) pairs.
        batch_size (int): Number of samples per mini-batch.
        shuffle (bool): Whether to shuffle dataset indices each epoch.
        to_tensor (bool): Whether to automatically wrap outputs as `Tensor` objects.
        is_iterable (bool): True if dataset is an instance of `IterableDataset`.
        m (int): Total number of samples (only for non-iterable datasets).
        n_batches (int): Number of mini-batches per epoch (only for non-iterable datasets).

    Methods:
        __len__():
            Returns the total number of batches per epoch (if defined).
            Raises an error if dataset length is undefined.

        _get_indices():
            Generates shuffled or sequential sample indices for the current epoch.

        __iter__():
            Iterates over dataset samples in mini-batches.
            Supports both standard and streaming datasets.
            Yields (X_batch, Y_batch) tuples — optionally as Tensors.
    """
    def __init__(self, dataset, batch_size=32, shuffle=True, to_tensor=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.to_tensor = to_tensor

        # Detect dataset type
        self.is_iterable = isinstance(dataset, IterableDataset)

        if not self.is_iterable:
            self.m = len(dataset)
            if self.m % batch_size != 0:
                self.n_batches = (self.m // batch_size) + 1
            else:
                self.n_batches = self.m // batch_size

    def __len__(self):
        """
        Return the total number of batches per epoch.

        Returns:
            int: Number of batches in the dataset.
        """
        if hasattr(self, "dataset") and hasattr(self.dataset, "__len__"):
            # Dataset-based loader
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size
        elif hasattr(self, "X"):
            # Array-based loader (old style)
            return (self.X.shape[0] + self.batch_size - 1) // self.batch_size
        else:
            raise AttributeError("Cannot determine dataset length — no dataset or X attribute found.")

    def _get_indices(self):        
        if self.shuffle:
            return xp.random.permutation(self.m)
        return xp.arange(self.m, dtype=xp.int64)

    def __iter__(self):
        if self.is_iterable:
            # Stream samples until exhausted
            batch = []
            for sample in self.dataset:
                batch.append(sample)
                if len(batch) == self.batch_size:
                    x, y = zip(*batch)
                    if self.to_tensor:
                        x = Tensor(x, requires_grad=False)
                        y = Tensor(y, requires_grad=False)
                    yield x, y
                    batch = []
            if batch:  # leftover
                x, y = zip(*batch)
                if self.to_tensor:
                    x = Tensor(x, requires_grad=False)
                    y = Tensor(y, requires_grad=False)
                yield x, y
        else:
            indices = self._get_indices()
            for start in range(0, self.m, self.batch_size):
                end = min(start + self.batch_size, self.m)
                batch_idx = indices[start:end]
                batch = [self.dataset[i] for i in batch_idx]
                x, y = zip(*batch)
                if self.to_tensor:
                    x = Tensor(x, requires_grad=False)
                    y = Tensor(y, requires_grad=False)
                yield x, y