import json
import csv
import matplotlib.pyplot as plt

import LunarLearn.backend as backend

xp = backend.xp

class History:
    """
    Training history tracker for logging and visualizing metrics over time.

    Stores metrics (loss, accuracy, learning rate, etc.) for each epoch or step.
    Supports export to JSON/CSV and convenient plotting of logged values.

    Attributes:
        records (list[dict]): A list of dictionaries, each containing metric
            names and their recorded values for a given epoch.

    Example:
        >>> history = History()
        >>> history.add(train_loss=0.45, val_loss=0.50, lr=0.001)
        >>> history.add(train_loss=0.30, val_loss=0.35, lr=0.0005)
        >>> history.plot(('train_loss', 'val_loss'))

        >>> history.to_json("training_log.json")
        >>> history.to_csv("training_log.csv")
    """
    def __init__(self):
        """Initialize an empty training history"""
        self.records = []

    def get_config(self):
        from LunarLearn.engine import serialize_value

        init_params = []
        extra_params = ["records"]

        return {
            "module": self.__class__.__module__,
            "class": self.__class__.__name__,
            "params": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k in init_params
            },
            "extra": {
                k: serialize_value(v)
                for k, v in self.__dict__.items()
                if k in extra_params
            }
        }

    @classmethod
    def from_config(cls, config):
        import importlib

        # Import the module and class
        module = importlib.import_module(config["module"])
        klass = getattr(module, config["class"])

        # Initialize object with params
        obj = klass(**config.get("params", {}))

        # Set extra attributes after init
        for k, v in config.get("extra", {}).items():
            setattr(obj, k, v)

        return obj

    def add(self, **metrics):
        """
        Add a new epoch entry.
        Example: history.add(loss=..., acc=..., lr=...)
        """
        self.records.append(metrics)

    def to_dict(self):
        """
        Returns a dictionary: keys are names, values are lists per epoch
        """
        output = {}
        for item in self.records:
            for k, v in item.items():
                output.setdefault(k, []).append(v)
        return output

    def to_json(self, path):
        """
        Save full record list to JSON file.
        """
        # convert any np/cp arrays to lists
        serializable_records = []
        for record in self.records:
            new_record = {}
            for k, v in record.items():
                if isinstance(v, (xp.ndarray,)):
                    new_record[k] = v.tolist()
                else:
                    new_record[k] = v
            serializable_records.append(new_record)

        with open(path, "w") as f:
            json.dump(serializable_records, f, indent=2)

    def to_csv(self, path):
        """
        Save full record list to CSV file.
        """
        if not self.records:
            return
        keys = list(self.records[0].keys())
        serializable_records = []
        for rec in self.records:
            new_rec = {}
            for k, v in rec.items():
                if hasattr(v, 'tolist'):  # works for np/cp arrays
                    new_rec[k] = v.tolist()
                else:
                    new_rec[k] = v
            serializable_records.append(new_rec)

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(serializable_records)

    def plot(self, keys=('loss','val_loss','acc','val_acc'), save=None):
        """
        Plot one or more metrics stored in history.
        Converts arrays to NumPy if needed.
        """
        data = self.to_dict()
        plt.figure(figsize=(10,4))
        
        for k in keys:
            if k in data:
                v = data[k]
                
                # If it's a single array, convert
                if hasattr(v, 'dtype') and 'cupy' in str(type(v)):
                    v = v.get()
                elif hasattr(v, '__array__'):
                    v = v.astype(float)
                
                # If it's a list, convert each element
                if isinstance(v, list):
                    v = [x.get() if 'cupy' in str(type(x)) else float(x) for x in v]
                
                plt.plot(v, label=k)
        
        plt.xlabel("Epoch")
        plt.legend()
        plt.grid(True)

        if save is not None:
            plt.savefig(save, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()