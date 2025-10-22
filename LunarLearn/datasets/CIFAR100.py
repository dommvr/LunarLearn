import numpy as np
from urllib.request import urlretrieve
from pathlib import Path

class CIFAR100:
    URL  = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    CACHE_DIR = Path.home() / ".lunarlearn" / "datasets"
    FILE = CACHE_DIR / "cifar100.npz"

    @classmethod
    def _download(cls):
        import pickle, tarfile
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not cls.FILE.exists():
            print("Downloading CIFAR-100...")
            tgz_path = cls.CACHE_DIR / "cifar100.tgz"
            urlretrieve(cls.URL, tgz_path)
            with tarfile.open(tgz_path, "r:gz") as tar:
                train_dict = pickle.load(tar.extractfile("cifar-100-python/train"), encoding="bytes")
                test_dict  = pickle.load(tar.extractfile("cifar-100-python/test"),  encoding="bytes")
                x_train = train_dict[b'data'].reshape(-1, 3, 32, 32)
                y_train = np.array(train_dict[b'fine_labels'], dtype=np.int64)
                x_test  = test_dict[b'data'].reshape(-1, 3, 32, 32)
                y_test  = np.array(test_dict[b'fine_labels'],  dtype=np.int64)
                np.savez(cls.FILE, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
            tgz_path.unlink()

    @classmethod
    def load(cls, normalize=True):
        cls._download()
        data = np.load(cls.FILE)
        x_train, y_train = data["x_train"], data["y_train"]
        x_test,  y_test  = data["x_test"],  data["y_test"]
        if normalize:
            x_train = x_train.astype(np.float32) / 255.0
            x_test  = x_test.astype(np.float32)  / 255.0
        return (x_train, y_train), (x_test, y_test)

    @classmethod
    def load_as_dataloaders(cls, batch_size, shuffle=True):
        from LunarLearn.dataloader import DataLoader
        (x_train, y_train), (x_test, y_test) = cls.load()
        return (
            DataLoader(x_train, y_train, batch_size=batch_size, shuffle=shuffle),
            DataLoader(x_test,  y_test,  batch_size=batch_size, shuffle=False)
        )