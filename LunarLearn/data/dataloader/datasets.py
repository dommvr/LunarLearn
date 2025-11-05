import os
import numpy as np
from PIL import Image
import csv
import random

import LunarLearn.backend as backend
from LunarLearn.tensor import Tensor

xp = backend.xp
DTYPE = backend.DTYPE
USING = backend.USING


class Dataset:
    """
    Base Dataset (map-style).
    Must implement __len__ and __getitem__.
    """
    def __init__(self, to_tensor=True):
        self.to_tensor = to_tensor

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError


class IterableDataset:
    """
    Base class for streaming datasets (iterable-style).
    Must implement __iter__, returning samples one by one.
    """
    def __init__(self, to_tensor=True):
        self.to_tensor = to_tensor

    def __iter__(self):
        raise NotImplementedError


class ArrayDataset(Dataset):
    """
    Simple Dataset for in-memory arrays.
    """
    def __init__(self, X, Y, dtype=DTYPE, to_tensor=True):
        super().__init__(to_tensor)
        self.X = xp.asarray(X, dtype=dtype)
        self.Y = xp.asarray(Y, dtype=dtype)
        self.m = self.X.shape[0]

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)
        return x, y
    

class GeneratorDataset(IterableDataset):
    """
    Wraps a Python generator as an iterable dataset.

    Example:
        def gen():
            for i in range(100):
                yield xp.array([i]), xp.array([i % 2])
        ds = GeneratorDataset(gen)
    """
    def __init__(self, generator_fn, length=None, to_tensor=True):
        super().__init__(to_tensor)
        self.generator_fn = generator_fn
        self.length = length

    def __len__(self):
        """
        Return dataset length if known, otherwise None for unknown-length generators.
        """
        return getattr(self, "length", None)

    def __iter__(self):
        x, y = self.generator_fn()
        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)
        return x, y
    

class ImageDataset:
    def __init__(self, path, size=(64, 64), to_tensor=True, one_hot=True):
        """
        Image dataset loader compatible with DataLoader.

        Args:
            path (str): Root folder containing class subfolders.
            size (tuple): (H, W) resize target for images.
            to_tensor (bool): If True, return Tensors. If False, return xp.arrays.
            one_hot (bool): If True, labels are one-hot encoded.
        """
        self.folder_path = path
        self.size = size
        self.to_tensor = to_tensor
        self.one_hot = one_hot

        # Collect class names and map to indices
        self.class_names = sorted([
            d for d in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, d))
        ])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}

        # Collect samples
        self.samples = []
        for cls_name in self.class_names:
            cls_folder = os.path.join(self.folder_path, cls_name)
            files = [f for f in os.listdir(cls_folder)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f in files:
                self.samples.append((os.path.join(cls_folder, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load + preprocess image
        img = Image.open(img_path).convert('RGB')
        img = img.resize(self.size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)

        # Convert to xp
        X = xp.asarray(img_array, dtype=DTYPE)
        if self.one_hot:
            Y = xp.eye(len(self.class_to_idx), dtype=DTYPE)[label]
        else:
            Y = xp.array(label, dtype=DTYPE)

        # Wrap in Tensor if requested
        if self.to_tensor:
            X = Tensor(X, requires_grad=False)
            Y = Tensor(Y, requires_grad=False)

        return X, Y
    

class CSVImageDataset:
    def __init__(self, csv_file, img_root="", size=(64, 64),
                 to_tensor=True, one_hot=True, delimiter=","):
        """
        Dataset that loads images from paths listed in a CSV file.

        Args:
            csv_file (str): Path to CSV file with columns [path,label].
            img_root (str): Optional root folder prefix for image paths.
            size (tuple): (H, W) resize target for images.
            to_tensor (bool): If True, return Tensors. If False, return xp.arrays.
            one_hot (bool): If True, labels are one-hot encoded.
            delimiter (str): CSV delimiter (default ',').
        """
        self.csv_file = csv_file
        self.img_root = img_root
        self.size = size
        self.to_tensor = to_tensor
        self.one_hot = one_hot

        # Read CSV rows
        self.samples = []
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                path = os.path.join(img_root, row["path"])
                label = int(row["label"])
                self.samples.append((path, label))

        # Infer number of classes
        labels = [lbl for _, lbl in self.samples]
        self.num_classes = max(labels) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # Load + preprocess
        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.size, Image.Resampling.LANCZOS)
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.transpose(img_array, (2, 0, 1))  # (C, H, W)

        # Convert to xp
        X = xp.asarray(img_array, dtype=DTYPE)
        if self.one_hot:
            Y = xp.eye(self.num_classes, dtype=DTYPE)[label]
        else:
            Y = xp.array(label, dtype=DTYPE)

        # Wrap in Tensor if requested
        if self.to_tensor:
            X = Tensor(X, requires_grad=False)
            Y = Tensor(Y, requires_grad=False)

        return X, Y
    
class SyntheticDataset:
    """
    Generate synthetic data for debugging/training toy models.
    Can create Gaussian blobs, noise, or classification spirals.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Dimensionality of features.
        n_classes (int): Number of classes.
        mode (str): ['gaussian', 'spiral', 'noise'].
        to_tensor (bool): Whether to return Tensors instead of arrays.
    """
    def __init__(self, n_samples=1000, n_features=2, n_classes=2,
                 mode="gaussian", to_tensor=True):
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.mode = mode
        self.to_tensor = to_tensor

        self.X, self.Y = self._generate()

    def _generate(self):
        if self.mode == "gaussian":
            X = []
            Y = []
            for i in range(self.n_classes):
                mean = xp.random.randn(self.n_features) * 2
                cov = xp.eye(self.n_features)
                samples = xp.random.multivariate_normal(mean, cov,
                                                        size=self.n_samples // self.n_classes)
                X.append(samples)
                Y.append(xp.full(samples.shape[0], i))
            X = xp.vstack(X)
            Y = xp.concatenate(Y)

        elif self.mode == "spiral":
            X = xp.zeros((self.n_samples, self.n_features))
            Y = xp.zeros(self.n_samples, dtype=int)
            n_per_class = self.n_samples // self.n_classes
            for j in range(self.n_classes):
                ix = range(j * n_per_class, (j + 1) * n_per_class)
                r = xp.linspace(0.0, 1, n_per_class)
                t = xp.linspace(j * 4, (j + 1) * 4, n_per_class) + xp.random.randn(n_per_class) * 0.2
                X[ix] = xp.c_[r * xp.sin(t), r * xp.cos(t)]
                Y[ix] = j

        elif self.mode == "noise":
            X = xp.random.randn(self.n_samples, self.n_features)
            Y = xp.random.randint(0, self.n_classes, self.n_samples)

        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        return X.astype(DTYPE), Y.astype(int)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.to_tensor:
            return Tensor(x, requires_grad=False), Tensor(y, requires_grad=False)
        return x, y


class AugmentedDataset:
    """
    Wraps another dataset and applies a transform to each sample.

    Args:
        dataset (Dataset): Base dataset.
        transform (callable): Function applied to (x, y).
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.transform:
            x, y = self.transform(x, y)
        return x, y


class LazyDataset:
    """
    Lazily loads data only when indexed (no preloading).

    Args:
        loader_fn (callable): Function that loads a single item given index.
        length (int): Total number of samples.
        to_tensor (bool): Convert output to Tensors.
    """
    def __init__(self, loader_fn, length, to_tensor=True):
        self.loader_fn = loader_fn
        self.length = length
        self.to_tensor = to_tensor

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x, y = self.loader_fn(idx)
        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)
        return x, y


class ConcatDataset:
    """
    Concatenate multiple datasets into one.

    Args:
        datasets (list): List of Dataset objects.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(d) for d in datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side="right")
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]
    

class PairedDataset:
    """
    Dataset for paired inputs (e.g., (src, tgt), (image, mask)).

    Args:
        X (array-like or list): Source samples.
        Y (array-like or list): Target samples (must be same length as X).
        to_tensor (bool): Whether to convert outputs to Tensors.
    """
    def __init__(self, X, Y, to_tensor=True):
        assert len(X) == len(Y), "X and Y must have the same length"
        self.X = X
        self.Y = Y
        self.to_tensor = to_tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)

        return x, y
    

class MixedDataset:
    """
    Dataset that randomly samples from multiple datasets.

    Useful for multitask training or mixing domains.

    Args:
        datasets (list): List of datasets to mix.
        sampling_probs (list, optional): Probabilities for sampling from each dataset.
                                         If None, uniform sampling is used.
    """
    def __init__(self, datasets, sampling_probs=None):
        self.datasets = datasets
        self.n = sum(len(ds) for ds in datasets)

        if sampling_probs is None:
            self.sampling_probs = [1 / len(datasets)] * len(datasets)
        else:
            assert len(sampling_probs) == len(datasets), "Mismatch in number of datasets"
            total = sum(sampling_probs)
            self.sampling_probs = [p / total for p in sampling_probs]

    def __len__(self):
        return self.n  # approximate size

    def __getitem__(self, idx):
        # Randomly pick a dataset according to probabilities
        ds_idx = random.choices(range(len(self.datasets)), weights=self.sampling_probs, k=1)[0]
        dataset = self.datasets[ds_idx]

        # Pick random sample from that dataset
        sample_idx = random.randint(0, len(dataset) - 1)
        return dataset[sample_idx]