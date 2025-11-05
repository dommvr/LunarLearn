import numpy as np
from urllib.request import urlretrieve
from pathlib import Path

class FashionMNIST:
    URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/fashion_mnist.npz"
    CACHE_DIR = Path.home() / ".lunarlearn" / "datasets"
    FILE = CACHE_DIR / "fashion_mnist.npz"

    @classmethod
    def _download(cls):
        cls.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if not cls.FILE.exists():
            print("Downloading FashionMNIST...")
            urlretrieve(cls.URL, cls.FILE)

    @classmethod
    def load(cls, normalize=True, flatten=False, add_channel_dim=False):
        """
        Returns:
            (x_train, y_train), (x_test, y_test)
            where x is either (m,28,28), (m,1,28,28) or (m,784).
        """
        cls._download()
        data = np.load(cls.FILE)
        x_train, y_train = data["x_train"], data["y_train"]
        x_test,  y_test  = data["x_test"],  data["y_test"]

        # Convert to float / normalize
        if normalize:
            x_train = x_train.astype(np.float32) / 255.0
            x_test  = x_test.astype(np.float32) / 255.0

        # Add channel dimension for CNNs (1 channel)
        if add_channel_dim:
            x_train = x_train[:, None, :, :]   # (m,1,28,28)
            x_test  = x_test[:, None, :, :]

        # Flatten for MLPs
        if flatten:
            x_train = x_train.reshape(x_train.shape[0], -1)  # (m,784)
            x_test  = x_test.reshape(x_test.shape[0],  -1)

        return (x_train, y_train), (x_test, y_test)
    
    @classmethod
    def load_as_dataloaders(cls, batch_size, shuffle=True, flatten=False, add_channel_dim=False):
        from LunarLearn.dataloader import DataLoader
        (X_train, Y_train), (X_test, Y_test) = cls.load(flatten=flatten, add_channel_dim=add_channel_dim)
        train_loader = DataLoader(X=X_train, Y=Y_train, batch_size=batch_size, shuffle=shuffle)
        test_loader  = DataLoader(X=X_test,  Y=Y_test,  batch_size=batch_size, shuffle=False)
        return train_loader, test_loader