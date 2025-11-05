import numpy as np
from urllib.request import urlretrieve
from pathlib import Path

class CIFAR10:
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    CACHE_DIR = Path.home() / ".lunarlearn" / "datasets"
    FILE = CACHE_DIR / "cifar10.npz"

    @classmethod
    def _download(cls):
        import pickle, tarfile
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not cls.FILE.exists():
            print("Downloading CIFAR-10...")
            tgz_path = cls.CACHE_DIR / "cifar10.tgz"
            urlretrieve(cls.URL, tgz_path)
            with tarfile.open(tgz_path, "r:gz") as tar:
                batches = []
                labels  = []
                for member in tar.getmembers():
                    if "data_batch" in member.name or "test_batch" in member.name:
                        data_dict = pickle.load(tar.extractfile(member), encoding="bytes")
                        batches.append(data_dict[b'data'])
                        labels.extend(data_dict[b'labels'])
                x = np.vstack(batches).reshape(-1, 3, 32, 32)
                y = np.array(labels, dtype=np.int64)
                np.savez(cls.FILE, x=x, y=y)
            tgz_path.unlink()

    @classmethod
    def load(cls, normalize=True, one_hot=False, num_classes=10):
        """
        Args:
            normalize (bool): scale pixel values to [0,1]
            one_hot (bool): if True, returns one-hot encoded labels
            num_classes (int): number of classes for one-hot encoding
        Returns:
            (x_train, y_train), (x_test, y_test)
        """
        cls._download()
        data = np.load(cls.FILE)
        x, y = data["x"], data["y"]
        if normalize:
            x = x.astype(np.float32) / 255.0

        if one_hot:
            y_onehot = np.zeros((y.shape[0], num_classes), dtype=np.float32)
            y_onehot[np.arange(y.shape[0]), y.astype(int)] = 1.0
            y = y_onehot

        # Split manually (first 50000 train, last 10000 test)
        return (x[:50000], y[:50000]), (x[50000:], y[50000:])

    @classmethod
    def load_as_dataloaders(cls, batch_size, shuffle=True, one_hot=False, num_classes=10):
        from LunarLearn.dataloader import DataLoader
        (x_train, y_train), (x_test, y_test) = cls.load(one_hot=one_hot, num_classes=num_classes)
        return (
            DataLoader(x_train, y_train, batch_size=batch_size, shuffle=shuffle),
            DataLoader(x_test,  y_test,  batch_size=batch_size, shuffle=False)
        )