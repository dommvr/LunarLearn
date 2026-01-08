import os
import xml.etree.ElementTree as ET
import math
import csv
import numpy as np
import gzip

import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader import Dataset, DatasetBundle, ArrayDataset, TextLabelDataset
from LunarLearn.data.datasets.utils import (get_data_home,
                                            _download,
                                            _read_csv_rows,
                                            _maybe_shuffle,
                                            _to_float,
                                            _extract_tgz,
                                            _loadtxt_flexible,
                                            _get_rng,
                                            _extract_zip,
                                            _parse_idx_gz_images,
                                            _parse_idx_gz_labels,
                                            _download_from_mirrors,
                                            _unpickle,
                                            _md5)

xp = backend.xp
DTYPE = np.dtype(backend.DTYPE)


def load_iris(*, return_X_y=True, as_dataset=False, dtype=None, shuffle=False, random_state=None, data_home=None):
    """
    Iris (UCI): 150 x 4, 3 classes.
    """
    if dtype is None:
        dtype = DTYPE
    home = get_data_home(data_home)
    ddir = os.path.join(home, "iris")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    fpath = os.path.join(ddir, "iris.data")
    _download(url, fpath)

    X_list, y_list = [], []
    label_map = {}
    for row in _read_csv_rows(fpath, delimiter=","):
        if len(row) < 5:
            continue
        feats = [float(row[i]) for i in range(4)]
        label = row[4].strip()
        if label == "":
            continue
        if label not in label_map:
            label_map[label] = len(label_map)
        X_list.append(feats)
        y_list.append(label_map[label])

    X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y,
                        feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
                        target_names=sorted(label_map, key=lambda k: label_map[k]),
                        description="Iris (UCI)")


def load_wine(*, return_X_y=True, as_dataset=False, dtype=None, shuffle=False, random_state=None, data_home=None):
    """
    Wine (UCI): 178 x 13, 3 classes. First column is class label (1..3).
    """
    if dtype is None:
        dtype = DTYPE
    home = get_data_home(data_home)
    ddir = os.path.join(home, "wine")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    fpath = os.path.join(ddir, "wine.data")
    _download(url, fpath)

    X_list, y_list = [], []
    for row in _read_csv_rows(fpath, delimiter=","):
        if len(row) < 14:
            continue
        cls = int(row[0]) - 1  # make 0..2
        feats = [float(v) for v in row[1:14]]
        X_list.append(feats)
        y_list.append(cls)

    X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Wine (UCI)")


def load_breast_cancer_wisconsin(*, return_X_y=True, as_dataset=False, dtype=None,
                                 shuffle=False, random_state=None, data_home=None, drop_id=True):
    """
    Breast Cancer Wisconsin (Diagnostic) (UCI): 569 x 30, binary label (M/B).
    File format: id, diagnosis, 30 features.
    """
    if dtype is None:
        dtype = DTYPE
    home = get_data_home(data_home)
    ddir = os.path.join(home, "wdbc")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    fpath = os.path.join(ddir, "wdbc.data")
    _download(url, fpath)

    X_list, y_list = [], []
    for row in _read_csv_rows(fpath, delimiter=","):
        if len(row) < 32:
            continue
        # row[0]=id, row[1]=M/B, row[2:]=features
        label = row[1].strip()
        y_list.append(1 if label == "M" else 0)
        feats = row[2:] if drop_id else row[0:1] + row[2:]
        X_list.append([float(v) for v in feats])

    X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=["benign", "malignant"], description="WDBC (UCI)")


def load_digits(*, return_X_y=True, as_dataset=False, dtype=None, shuffle=False,
                    random_state=None, data_home=None, split="all"):
    """
    Optical Recognition of Handwritten Digits (UCI): 8x8 block features (64) + label.
    Files: optdigits.tra (train), optdigits.tes (test)
    Each row: 64 integers (0..16), then class (0..9).
    """
    if dtype is None:
        dtype = DTYPE
    home = get_data_home(data_home)
    ddir = os.path.join(home, "optdigits")
    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/"
    train_url = base + "optdigits.tra"
    test_url = base + "optdigits.tes"
    train_path = os.path.join(ddir, "optdigits.tra")
    test_path = os.path.join(ddir, "optdigits.tes")
    _download(train_url, train_path)
    _download(test_url, test_path)

    def read_file(p):
        Xl, yl = [], []
        for row in _read_csv_rows(p, delimiter=","):
            if len(row) < 65:
                continue
            feats = [float(int(v)) for v in row[:64]]
            lab = int(row[64])
            Xl.append(feats)
            yl.append(lab)
        return Xl, yl

    X_list, y_list = [], []
    if split in ("train", "all"):
        a, b = read_file(train_path)
        X_list += a
        y_list += b
    if split in ("test", "all"):
        a, b = read_file(test_path)
        X_list += a
        y_list += b
    if split not in ("train", "test", "all"):
        raise ValueError("split must be 'train', 'test', or 'all'")

    X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Optdigits (UCI)")


def load_titanic(*, return_X_y=True, as_dataset=False, dtype=None, shuffle=False,
                 random_state=None, data_home=None, encode="onehot"):
    """
    Titanic (OpenML CSV export): mixed types.
    Default: use a common 'starter' feature set and produce numeric X.

    encode:
      - 'onehot'  : one-hot for categorical (sex, embarked)
      - 'ordinal' : map categorical to ints
      - 'none'    : returns raw strings mixed in X (not recommended unless you handle it)
    """
    if dtype is None:
        dtype = DTYPE
    home = get_data_home(data_home)
    ddir = os.path.join(home, "titanic")
    url = "https://www.openml.org/data/get_csv/16826755/phpMYEkMl"
    fpath = os.path.join(ddir, "titanic.csv")
    _download(url, fpath)

    # Common, sane subset (because 'name' and 'ticket' are not features, they’re chaos):
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
    target = "survived"

    with open(fpath, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Collect columns
    sex_vals = sorted({(r.get("sex", "") or "").strip() for r in rows if (r.get("sex", "") or "").strip() not in ("", "?")})
    emb_vals = sorted({(r.get("embarked", "") or "").strip() for r in rows if (r.get("embarked", "") or "").strip() not in ("", "?")})

    def onehot(val, vocab):
        v = (val or "").strip()
        out = [0.0] * len(vocab)
        if v in vocab:
            out[vocab.index(v)] = 1.0
        return out

    X_list, y_list = [], []
    for r in rows:
        ys = (r.get(target, "") or "").strip()
        if ys in ("", "?"):
            continue
        y_list.append(int(float(ys)))

        feats = []
        # pclass numeric-ish
        feats.append(_to_float(r.get("pclass", "?")))
        # sex categorical
        if encode == "onehot":
            feats += onehot(r.get("sex", ""), sex_vals)
        elif encode == "ordinal":
            v = (r.get("sex", "") or "").strip()
            feats.append(float(sex_vals.index(v)) if v in sex_vals else math.nan)
        else:
            feats.append((r.get("sex", "") or "").strip())

        # numeric columns
        feats.append(_to_float(r.get("age", "?")))
        feats.append(_to_float(r.get("sibsp", "?")))
        feats.append(_to_float(r.get("parch", "?")))
        feats.append(_to_float(r.get("fare", "?")))

        # embarked categorical
        if encode == "onehot":
            feats += onehot(r.get("embarked", ""), emb_vals)
        elif encode == "ordinal":
            v = (r.get("embarked", "") or "").strip()
            feats.append(float(emb_vals.index(v)) if v in emb_vals else math.nan)
        else:
            feats.append((r.get("embarked", "") or "").strip())

        X_list.append(feats)

    if encode == "none":
        # Mixed types: return Python objects. If you do this, you *must* handle it downstream.
        X = X_list
    else:
        X = np.asarray(X_list, dtype=dtype)
    y = np.asarray(y_list, dtype=np.int64)

    if encode != "none":
        X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        if encode == "none":
            raise ValueError("as_dataset=True requires encode != 'none' (ArrayDataset expects numeric arrays).")
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=["died", "survived"], description="Titanic (OpenML CSV export)")


def load_adult(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    split="train",              # "train" | "test" | "all"
    one_hot=True,
    drop_missing=True,
):
    """
    Adult / Census Income (UCI): binary classification.
    y: 0 for <=50K, 1 for >50K

    Notes:
    - adult.test has a header/comment line and labels with trailing '.'
    - missing values are marked as '?'
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "adult")

    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult"
    train_url = f"{base}/adult.data"
    test_url  = f"{base}/adult.test"

    train_path = os.path.join(ddir, "adult.data")
    test_path  = os.path.join(ddir, "adult.test")

    if split in ("train", "all"):
        _download(train_url, train_path)
    if split in ("test", "all"):
        _download(test_url, test_path)
    if split not in ("train", "test", "all"):
        raise ValueError("split must be 'train', 'test', or 'all'")

    cols = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race","sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    def parse_file(path, is_test=False):
        rows = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if is_test and line.startswith("|"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) != 15:
                    continue
                d = dict(zip(cols, parts))
                d["income"] = d["income"].replace(".", "")
                rows.append(d)
        return rows

    rows = []
    if split in ("train", "all"):
        rows += parse_file(train_path, is_test=False)
    if split in ("test", "all"):
        rows += parse_file(test_path, is_test=True)

    if drop_missing:
        def ok(r):
            return all((v != "?") for v in r.values())
        rows = [r for r in rows if ok(r)]

    # y
    y_list = [1 if r["income"] == ">50K" else 0 for r in rows]
    y = np.asarray(y_list, dtype=np.int64)

    # X
    num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
                "capital-loss", "hours-per-week"]
    cat_cols = [c for c in cols if c not in num_cols + ["income"]]

    def to_float(s):
        s = (s or "").strip()
        if s == "" or s == "?" or s.lower() == "nan":
            return math.nan
        return float(s)

    if not one_hot:
        X_list = [[to_float(r[c]) for c in num_cols] for r in rows]
        X = np.asarray(X_list, dtype=dtype)
    else:
        # build vocab per categorical col
        vocabs = {}
        for c in cat_cols:
            vocabs[c] = sorted({r[c] for r in rows})
        offsets = {}
        off = 0
        for c in cat_cols:
            offsets[c] = off
            off += len(vocabs[c])
        total_cat = off

        X_num = np.asarray([[to_float(r[c]) for c in num_cols] for r in rows], dtype=dtype)
        X_cat = np.zeros((X_num.shape[0], total_cat), dtype=dtype)

        for i, r in enumerate(rows):
            for c in cat_cols:
                v = r[c]
                # linear search is fine here; dataset size ~48k.
                # if you care, build dict maps once.
                try:
                    j = vocabs[c].index(v)
                except ValueError:
                    continue
                X_cat[i, offsets[c] + j] = 1.0

        X = np.concatenate([X_num, X_cat], axis=1)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Adult / Census Income (UCI)")


def load_california_housing(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
):
    """
    California Housing (regression).
    Downloads a tgz archive and returns X:(N,8), y:(N,).
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "california_housing")
    tgz_path = os.path.join(ddir, "cal_housing.tgz")

    urls = [
        "https://figshare.com/ndownloader/files/5976036",
        "https://ndownloader.figshare.com/files/5976036",
    ]
    # try mirrors; _download is single-url so do a simple fallback
    if not os.path.exists(tgz_path):
        last = None
        for u in urls:
            try:
                _download(u, tgz_path)
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            raise last

    raw_dir = os.path.join(ddir, "raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        _extract_tgz(tgz_path, raw_dir)

    data_path = os.path.join(raw_dir, "CaliforniaHousing", "cal_housing.data")
    target_path = os.path.join(raw_dir, "CaliforniaHousing", "cal_housing.target")

    X_np = _loadtxt_flexible(data_path)
    y_np = _loadtxt_flexible(target_path)

    X = np.asarray(X_np, dtype=dtype)
    y = np.asarray(y_np, dtype=dtype)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="California Housing (regression)")


def load_diabetes(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    scaled=True,
):
    """
    Diabetes (regression), small classic dataset (442x10).
    Downloads the raw CSV files.
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "diabetes")

    data_path = os.path.join(ddir, "diabetes_data_raw.csv.gz")
    targ_path = os.path.join(ddir, "diabetes_target.csv.gz")

    data_url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/diabetes_data_raw.csv.gz"
    targ_url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/datasets/data/diabetes_target.csv.gz"

    _download(data_url, data_path)
    _download(targ_url, targ_path)

    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        X_np = np.loadtxt(f, delimiter=",", dtype=np.float32)
    with gzip.open(targ_path, "rt", encoding="utf-8") as f:
        y_np = np.loadtxt(f, delimiter=",", dtype=np.float32)

    if scaled:
        mu = X_np.mean(axis=0, keepdims=True)
        sig = X_np.std(axis=0, keepdims=True)
        sig[sig == 0] = 1.0
        X_np = (X_np - mu) / sig

    X = np.asarray(X_np, dtype=dtype)
    y = np.asarray(y_np, dtype=dtype)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Diabetes (regression)")


def load_ames_housing(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    one_hot=True,
    drop_missing=False,
    target_col=None,     # auto-detect if None
):
    """
    Ames Housing (regression).
    Downloads a CSV mirror and builds numeric X.
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "ames_housing")
    csv_path = os.path.join(ddir, "AmesHousing.csv")

    urls = [
        "https://raw.githubusercontent.com/bencmbit/datasets/master/AmesHousing.csv",
        "https://raw.githubusercontent.com/maibennett/sta235/main/exampleSite/content/Classes/Week3/2_OLS_Issues/data/AmesHousing.csv",
    ]
    if not os.path.exists(csv_path):
        last = None
        for u in urls:
            try:
                _download(u, csv_path)
                last = None
                break
            except Exception as e:
                last = e
        if last is not None:
            raise last

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise RuntimeError("AmesHousing CSV is empty or unreadable")

    if target_col is None:
        for cand in ("SalePrice", "Sale_Price", "saleprice", "sale_price"):
            if cand in rows[0]:
                target_col = cand
                break
        if target_col is None:
            raise ValueError("Could not auto-detect target column. Pass target_col explicitly.")

    if drop_missing:
        def ok(r):
            for v in r.values():
                s = "" if v is None else str(v).strip()
                if s == "" or s == "?" or s.lower() in ("na", "nan"):
                    return False
            return True
        rows = [r for r in rows if ok(r)]

    def to_float(s):
        s = "" if s is None else str(s).strip()
        if s == "" or s == "?" or s.lower() in ("na", "nan"):
            return math.nan
        try:
            return float(s)
        except Exception:
            return math.nan

    y_np = np.asarray([to_float(r[target_col]) for r in rows], dtype=np.float32)

    cols = [c for c in rows[0].keys() if c != target_col]

    # numeric vs categorical sniff
    num_cols, cat_cols = [], []
    for c in cols:
        seen = 0
        numeric = 0
        for r in rows[:200]:
            v = r.get(c, "")
            s = "" if v is None else str(v).strip()
            if s == "" or s == "?" or s.lower() in ("na", "nan"):
                continue
            seen += 1
            try:
                float(s)
                numeric += 1
            except Exception:
                pass
        if seen > 0 and (numeric / seen) > 0.95:
            num_cols.append(c)
        else:
            cat_cols.append(c)

    X_num = np.asarray([[to_float(r.get(c, "")) for c in num_cols] for r in rows], dtype=np.float32)

    if not one_hot or not cat_cols:
        X_np = X_num
    else:
        # build vocab per categorical column
        vocabs = {}
        for c in cat_cols:
            vals = []
            for r in rows:
                v = r.get(c, "")
                s = "" if v is None else str(v).strip()
                if s == "" or s == "?" or s.lower() in ("na", "nan"):
                    s = "__MISSING__"
                vals.append(s)
                r[c] = s
            vocabs[c] = sorted(set(vals))

        offsets = {}
        off = 0
        for c in cat_cols:
            offsets[c] = off
            off += len(vocabs[c])
        total_cat = off

        X_cat = np.zeros((X_num.shape[0], total_cat), dtype=np.float32)
        for i, r in enumerate(rows):
            for c in cat_cols:
                v = r[c]
                try:
                    j = vocabs[c].index(v)
                except ValueError:
                    continue
                X_cat[i, offsets[c] + j] = 1.0

        X_np = np.concatenate([X_num, X_cat], axis=1)

    X = np.asarray(X_np, dtype=dtype)
    y = np.asarray(y_np, dtype=dtype)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description=f"Ames Housing (regression), target={target_col}")


class TwentyNewsgroupsDataset(Dataset):
    """
    Map-style dataset returning dict samples:
      - {"text": str, "label": int}
      - {"input_ids": list[int] or xp.int64 array, "label": int} if tokenizer is provided
    """
    def __init__(self, root_dir, subset="train", tokenizer=None, to_tensor=False):
        super().__init__(to_tensor=to_tensor)
        if subset not in ("train", "test"):
            raise ValueError('subset must be "train" or "test"')
        self.subset = subset
        self.tokenizer = tokenizer

        base = os.path.join(root_dir, f"20news-bydate-{subset}")
        if not os.path.isdir(base):
            raise RuntimeError(f"Missing extracted folder: {base}")

        self.class_names = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples = []
        for c in self.class_names:
            cdir = os.path.join(base, c)
            for fn in os.listdir(cdir):
                p = os.path.join(cdir, fn)
                if os.path.isfile(p):
                    self.samples.append((p, self.class_to_idx[c]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with open(path, "r", encoding="latin-1", errors="ignore") as f:
            text = f.read()

        if self.tokenizer is None:
            return {"text": text, "label": int(label)}

        ids = self.tokenizer.encode(text)
        # keep ids as list; your collate_pad_tokens can handle list->pad
        return {"input_ids": ids, "label": int(label)}


def load_20newsgroups(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,                  # unused, kept for convention consistency
    shuffle=False,               # for X,y mode only
    random_state=None,           # for X,y mode only
    data_home=None,
    subset="train",              # "train" | "test"
    tokenizer=None,
):
    """
    20 Newsgroups (bydate): text multiclass.

    If as_dataset=True:
        returns TwentyNewsgroupsDataset yielding dict samples.
    If return_X_y=True:
        returns (texts, y) as Python lists (shuffled if requested).
    """
    home = get_data_home(data_home)
    ddir = os.path.join(home, "20newsgroups")

    url = "https://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz"
    tgz_path = os.path.join(ddir, "20news-bydate.tar.gz")
    _download(url, tgz_path)

    raw_dir = os.path.join(ddir, "raw")
    if not os.path.exists(raw_dir) or not os.listdir(raw_dir):
        _extract_tgz(tgz_path, raw_dir)

    if as_dataset:
        return TwentyNewsgroupsDataset(raw_dir, subset=subset, tokenizer=tokenizer, to_tensor=False)

    # return_X_y mode: materialize lists
    ds = TwentyNewsgroupsDataset(raw_dir, subset=subset, tokenizer=None, to_tensor=False)
    texts = []
    labels = []
    for i in range(len(ds)):
        s = ds[i]
        texts.append(s["text"])
        labels.append(int(s["label"]))

    y = np.asarray(labels, dtype=np.int64)
    if shuffle:
        # shuffle texts + labels together
        perm = _get_rng(random_state).permutation(len(texts))
        perm = [int(p) for p in perm.tolist()]
        texts = [texts[i] for i in perm]
        y = y[perm]

    if return_X_y:
        return texts, y
    return DatasetBundle(X=texts, y=y, description=f"20 Newsgroups ({subset})")


def load_sms_spam(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
):
    """
    SMS Spam Collection (UCI): 5574 messages labeled ham/spam.
    Returns:
      X: list[str]
      y: xp.int64 (0=ham, 1=spam)
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "sms_spam")
    os.makedirs(ddir, exist_ok=True)

    # UCI "static" download endpoint (zip)
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zpath = os.path.join(ddir, "sms_spam_collection.zip")
    _download(url, zpath)

    extracted = os.path.join(ddir, "extracted")
    marker = os.path.join(extracted, "SMSSpamCollection")
    if not os.path.exists(marker):
        _extract_zip(zpath, extracted)

    # File is typically: extracted/SMSSpamCollection (tab-separated)
    # Some zips include a folder; handle both.
    fpath = marker
    if not os.path.exists(fpath):
        # try find it
        for root, _, files in os.walk(extracted):
            if "SMSSpamCollection" in files:
                fpath = os.path.join(root, "SMSSpamCollection")
                break
    if not os.path.exists(fpath):
        raise FileNotFoundError("SMSSpamCollection not found after extraction")

    texts = []
    labels = []
    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            # label \t message
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            lab, msg = parts[0].strip().lower(), parts[1]
            y = 1 if lab == "spam" else 0
            texts.append(msg)
            labels.append(y)

    y = np.asarray(labels, dtype=np.int64)
    if shuffle:
        rng = _get_rng(random_state)
        perm = rng.permutation(len(texts))
        texts = [texts[int(i)] for i in perm]
        y = y[perm]

    if as_dataset:
        return TextLabelDataset(texts, y, to_tensor=False)
    if return_X_y:
        return texts, y
    return DatasetBundle(X=texts, y=y, target_names=["ham", "spam"], description="SMS Spam Collection (UCI)")


def load_imdb_reviews(
    *,
    subset="train",            # "train" | "test" | "all"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
):
    """
    Stanford Large Movie Review Dataset (aclImdb_v1): pos/neg movie reviews.
    Returns:
      X: list[str]
      y: xp.int64 (0=neg, 1=pos)
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "imdb_reviews")
    os.makedirs(ddir, exist_ok=True)

    url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tgz = os.path.join(ddir, "aclImdb_v1.tar.gz")
    _download(url, tgz)

    extracted = os.path.join(ddir, "aclImdb")
    if not os.path.exists(os.path.join(extracted, "train")):
        _extract_tgz(tgz, ddir)  # tar contains folder "aclImdb"

    def _load_split(split_dir):
        Xs, ys = [], []
        for label_name, yv in (("neg", 0), ("pos", 1)):
            p = os.path.join(split_dir, label_name)
            if not os.path.isdir(p):
                continue
            for fn in os.listdir(p):
                if not fn.endswith(".txt"):
                    continue
                fpath = os.path.join(p, fn)
                with open(fpath, "r", encoding="utf-8", errors="replace") as f:
                    Xs.append(f.read())
                ys.append(yv)
        return Xs, ys

    texts, labels = [], []
    if subset in ("train", "all"):
        Xs, ys = _load_split(os.path.join(extracted, "train"))
        texts += Xs
        labels += ys
    if subset in ("test", "all"):
        Xs, ys = _load_split(os.path.join(extracted, "test"))
        texts += Xs
        labels += ys

    y = np.asarray(labels, dtype=np.int64)
    if shuffle:
        rng = _get_rng(random_state)
        perm = rng.permutation(len(texts))
        texts = [texts[int(i)] for i in perm]
        y = y[perm]

    if as_dataset:
        return TextLabelDataset(texts, y, to_tensor=False)
    if return_X_y:
        return texts, y
    return DatasetBundle(X=texts, y=y, target_names=["neg", "pos"], description="IMDb Large Movie Review Dataset")


def load_mnist(
    *,
    subset="train",            # "train" | "test" | "all"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,            # scale to [0,1]
):
    """
    MNIST (IDX gzip). Returns X as (N,1,28,28) float in NCHW.
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "mnist")
    os.makedirs(ddir, exist_ok=True)

    # Mirrors similar to torchvision (plus a GitHub raw mirror).
    mirrors = [
        "https://ossci-datasets.s3.amazonaws.com/mnist",
        "http://yann.lecun.com/exdb/mnist",
        "https://raw.githubusercontent.com/fgnt/mnist/master",
    ]

    files = {
        "train_images": ("train-images-idx3-ubyte.gz", "f68b3c2dcbeaaa9fbdd348bbdeb94873"),
        "train_labels": ("train-labels-idx1-ubyte.gz", "d53e105ee54ea40749a09fcbcd1e9432"),
        "test_images":  ("t10k-images-idx3-ubyte.gz",  "9fb629c4189551a2d022fa330f9573f3"),
        "test_labels":  ("t10k-labels-idx1-ubyte.gz",  "ec29112dd5afa0611ce80d1b7f02629c"),
    }

    def _get(name):
        fn, md5 = files[name]
        fpath = os.path.join(ddir, fn)
        if not os.path.exists(fpath) or (_md5(fpath) != md5):
            _download_from_mirrors(mirrors, fn, fpath, expected_md5=md5)
        return fpath

    Xs, ys = [], []
    if subset in ("train", "all"):
        Xi = _parse_idx_gz_images(_get("train_images"))
        yi = _parse_idx_gz_labels(_get("train_labels"))
        Xs.append(Xi)
        ys.append(yi)
    if subset in ("test", "all"):
        Xi = _parse_idx_gz_images(_get("test_images"))
        yi = _parse_idx_gz_labels(_get("test_labels"))
        Xs.append(Xi)
        ys.append(yi)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)

    # NCHW
    X = X[:, None, :, :].astype(np.float32)
    if normalize:
        X /= 255.0

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=[str(i) for i in range(10)], description="MNIST")


def load_fashion_mnist(
    *,
    subset="train",            # "train" | "test" | "all"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,
):
    """
    Fashion-MNIST (IDX gzip). Returns X as (N,1,28,28) float in NCHW.
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "fashion_mnist")
    os.makedirs(ddir, exist_ok=True)

    base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"

    files = {
        "train_images": ("train-images-idx3-ubyte.gz", None),
        "train_labels": ("train-labels-idx1-ubyte.gz", None),
        "test_images":  ("t10k-images-idx3-ubyte.gz", None),
        "test_labels":  ("t10k-labels-idx1-ubyte.gz", None),
    }

    def _get(name):
        fn, _ = files[name]
        fpath = os.path.join(ddir, fn)
        if not os.path.exists(fpath) or os.path.getsize(fpath) == 0:
            _download(base + "/" + fn, fpath)
        return fpath

    Xs, ys = [], []
    if subset in ("train", "all"):
        Xi = _parse_idx_gz_images(_get("train_images"))
        yi = _parse_idx_gz_labels(_get("train_labels"))
        Xs.append(Xi); ys.append(yi)
    if subset in ("test", "all"):
        Xi = _parse_idx_gz_images(_get("test_images"))
        yi = _parse_idx_gz_labels(_get("test_labels"))
        Xs.append(Xi); ys.append(yi)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0).astype(np.int64)

    X = X[:, None, :, :].astype(np.float32)
    if normalize:
        X /= 255.0

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Fashion-MNIST")


def load_olivetti_faces(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    flatten=False,             # if False: NCHW (N,1,64,64), else (N,4096)
    normalize=True,
):
    """
    Olivetti faces is historically distributed as a .mat file.
    We download olivettifaces.mat and require scipy.io.loadmat to parse it.
    Returns:
      X: (N,1,64,64) or (N,4096)
      y: (N,) int64 labels (0..39)
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "olivetti_faces")
    os.makedirs(ddir, exist_ok=True)

    # Default URL used by sklearn historically.
    url = "http://cs.nyu.edu/~roweis/data/olivettifaces.mat"
    fpath = os.path.join(ddir, "olivettifaces.mat")
    _download(url, fpath)

    try:
        import scipy.io
        from scipy.io import loadmat
    except Exception as e:
        raise RuntimeError(
            "Olivetti faces loader needs scipy (scipy.io.loadmat) to parse .mat.\n"
            "Install scipy or choose a different source format."
        ) from e

    mat = loadmat(fpath)
    # Common keys: 'faces' (4096, 400) float, 'id' (1,400)
    if "faces" not in mat or "id" not in mat:
        raise KeyError(f"Unexpected .mat keys: {list(mat.keys())}")

    faces = mat["faces"]          # shape (4096, 400)
    ids = mat["id"].reshape(-1)   # shape (400,)

    X = faces.T.astype(np.float32)      # (400,4096)
    y = ids.astype(np.int64)

    if normalize:
        # many versions are already in [0,1], but clamp just in case
        X = np.clip(X, 0.0, 1.0)

    if not flatten:
        # reshape to (N,1,64,64) NCHW
        X = X.reshape(-1, 1, 64, 64)

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="Olivetti Faces")


def load_movielens(
    *,
    version="100k",            # "100k" | "1m"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
):
    """
    MovieLens ratings for MF/BiasedMF.
    Returns:
      X: (N,2) int64 where columns are [user_id, item_id] (0-based)
      y: (N,) float ratings
    """
    if dtype is None:
        dtype = DTYPE
    if version not in ("100k", "1m"):
        raise ValueError("version must be '100k' or '1m'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, f"movielens_{version}")
    os.makedirs(ddir, exist_ok=True)

    if version == "100k":
        url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
        zpath = os.path.join(ddir, "ml-100k.zip")
        _download(url, zpath)
        extracted = os.path.join(ddir, "ml-100k")
        if not os.path.exists(os.path.join(extracted, "u.data")):
            _extract_zip(zpath, ddir)

        ratings_path = os.path.join(extracted, "u.data")
        # u.data: user_id item_id rating timestamp  (tab-separated), user_id starts at 1, item_id starts at 1
        rows = []
        with open(ratings_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                u = int(parts[0]) - 1
                i = int(parts[1]) - 1
                r = float(parts[2])
                rows.append((u, i, r))

    else:
        url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
        zpath = os.path.join(ddir, "ml-1m.zip")
        _download(url, zpath)
        extracted = os.path.join(ddir, "ml-1m")
        if not os.path.exists(os.path.join(extracted, "ratings.dat")):
            _extract_zip(zpath, ddir)

        ratings_path = os.path.join(extracted, "ratings.dat")
        # ratings.dat: UserID::MovieID::Rating::Timestamp  (UserID starts at 1)
        rows = []
        with open(ratings_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split("::")
                if len(parts) < 3:
                    continue
                u = int(parts[0]) - 1
                i = int(parts[1]) - 1
                r = float(parts[2])
                rows.append((u, i, r))

    X = np.asarray([[u, i] for (u, i, _) in rows], dtype=np.int64)
    y = np.asarray([r for (_, _, r) in rows], dtype=dtype)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description=f"MovieLens {version}")


def load_goodbooks_10k(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    max_rows=None,             # IMPORTANT: ratings.csv is huge (~6M). None loads all.
):
    """
    GoodBooks-10k ratings for MF/BiasedMF.
    Returns:
      X: (N,2) int64 [user_id, book_id] (0-based)
      y: (N,) float rating
    """
    if dtype is None:
        dtype = DTYPE

    home = get_data_home(data_home)
    ddir = os.path.join(home, "goodbooks_10k")
    os.makedirs(ddir, exist_ok=True)

    # GitHub raw files (books.csv exists; ratings.csv is large).
    ratings_url = "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv"
    ratings_path = os.path.join(ddir, "ratings.csv")
    _download(ratings_url, ratings_path)

    rows_u = []
    rows_i = []
    rows_r = []

    # ratings.csv: user_id,book_id,rating
    with open(ratings_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for k, row in enumerate(reader):
            if max_rows is not None and k >= int(max_rows):
                break
            u = int(row["user_id"]) - 1
            b = int(row["book_id"]) - 1
            r = float(row["rating"])
            rows_u.append(u)
            rows_i.append(b)
            rows_r.append(r)

    X = np.asarray(np.stack([rows_u, rows_i], axis=1), dtype=np.int64)
    y = np.asarray(np.asarray(rows_r, dtype=np.float32), dtype=dtype)

    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, description="GoodBook-10k ratings")


def load_cifar10(
    *,
    subset="train",            # "train" | "test" | "all"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,            # scale to [0,1]
):
    """
    CIFAR-10 (python version). Returns X as (N,3,32,32) float in NCHW.
    y: xp.int64 in [0..9].
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "cifar10")
    os.makedirs(ddir, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    tgz = os.path.join(ddir, "cifar-10-python.tar.gz")
    expected_md5 = "c58f30108f718f92721af3b95e74349a"  # official
    if (not os.path.exists(tgz)) or (_md5(tgz) != expected_md5):
        _download(url, tgz)

    extracted_root = os.path.join(ddir, "cifar-10-batches-py")
    if not os.path.isdir(extracted_root):
        _extract_tgz(tgz, ddir)

    # class names
    meta = _unpickle(os.path.join(extracted_root, "batches.meta"))
    target_names = [n.decode("utf-8") for n in meta[b"label_names"]]

    def _load_batches(paths):
        Xs, ys = [], []
        for p in paths:
            d = _unpickle(p)
            Xs.append(d[b"data"])               # (N, 3072) uint8
            ys.extend(d[b"labels"])            # list[int]
        X = np.concatenate(Xs, axis=0)
        y = np.asarray(ys, dtype=np.int64)
        # data layout: R(1024) G(1024) B(1024) per row (documented)
        X = X.reshape(-1, 3, 32, 32).astype(np.float32)
        if normalize:
            X /= 255.0
        return X, y

    if subset == "train":
        batch_paths = [os.path.join(extracted_root, f"data_batch_{i}") for i in range(1, 6)]
        X, y = _load_batches(batch_paths)
    elif subset == "test":
        X, y = _load_batches([os.path.join(extracted_root, "test_batch")])
    else:
        train_paths = [os.path.join(extracted_root, f"data_batch_{i}") for i in range(1, 6)]
        test_path = [os.path.join(extracted_root, "test_batch")]
        Xtr, ytr = _load_batches(train_paths)
        Xte, yte = _load_batches(test_path)
        X = np.concatenate([Xtr, Xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=target_names, description="CIFAR-10 (python version)")


def load_cifar100(
    *,
    subset="train",            # "train" | "test" | "all"
    label_mode="fine",         # "fine" | "coarse"
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,            # scale to [0,1]
):
    """
    CIFAR-100 (python version). Returns X as (N,3,32,32) float in NCHW.
    y: xp.int64 in [0..99] if fine, or [0..19] if coarse.
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")
    if label_mode not in ("fine", "coarse"):
        raise ValueError("label_mode must be 'fine' or 'coarse'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "cifar100")
    os.makedirs(ddir, exist_ok=True)

    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    tgz = os.path.join(ddir, "cifar-100-python.tar.gz")
    expected_md5 = "eb9058c3a382ffc7106e4002c42a8d85"  # official
    if (not os.path.exists(tgz)) or (_md5(tgz) != expected_md5):
        _download(url, tgz)

    extracted_root = os.path.join(ddir, "cifar-100-python")
    if not os.path.isdir(extracted_root):
        _extract_tgz(tgz, ddir)

    meta = _unpickle(os.path.join(extracted_root, "meta"))
    fine_names = [n.decode("utf-8") for n in meta[b"fine_label_names"]]
    coarse_names = [n.decode("utf-8") for n in meta[b"coarse_label_names"]]

    def _load_split(name):
        d = _unpickle(os.path.join(extracted_root, name))
        X = d[b"data"]  # (N, 3072)
        if label_mode == "fine":
            y = np.asarray(d[b"fine_labels"], dtype=np.int64)
            tnames = fine_names
        else:
            y = np.asarray(d[b"coarse_labels"], dtype=np.int64)
            tnames = coarse_names

        X = X.reshape(-1, 3, 32, 32).astype(np.float32)
        if normalize:
            X /= 255.0
        return X, y, tnames

    if subset == "train":
        X, y, target_names = _load_split("train")
    elif subset == "test":
        X, y, target_names = _load_split("test")
    else:
        Xtr, ytr, target_names = _load_split("train")
        Xte, yte, _ = _load_split("test")
        X = np.concatenate([Xtr, Xte], axis=0)
        y = np.concatenate([ytr, yte], axis=0)

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    desc = f"CIFAR-100 (python version, label_mode={label_mode})"
    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=target_names, description=desc)


def load_svhn(
    *,
    subset="train",            # "train" | "test" | "all"
    include_extra=False,       # if True, add 'extra' to train/all
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,            # scale to [0,1]
):
    """
    SVHN (cropped digits). Returns X as (N,3,32,32) float in NCHW.
    y: xp.int64 in [0..9] with original '10' remapped to 0.
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    try:
        import scipy.io  # noqa: F401
        from scipy.io import loadmat
    except Exception as e:
        raise ImportError("SVHN loader requires scipy (pip install scipy).") from e

    home = get_data_home(data_home)
    ddir = os.path.join(home, "svhn")
    os.makedirs(ddir, exist_ok=True)

    base = "http://ufldl.stanford.edu/housenumbers"
    files = {
        "train": ("train_32x32.mat", os.path.join(ddir, "train_32x32.mat")),
        "test":  ("test_32x32.mat",  os.path.join(ddir, "test_32x32.mat")),
        "extra": ("extra_32x32.mat", os.path.join(ddir, "extra_32x32.mat")),
    }

    def _get(split):
        fn, fpath = files[split]
        if not os.path.exists(fpath):
            _download(f"{base}/{fn}", fpath)
        return fpath

    def _load_split(split):
        m = loadmat(_get(split))
        X = m["X"]  # (32, 32, 3, N)
        y = m["y"].astype(np.int64).reshape(-1)  # (N,)
        # label '10' means digit 0
        y[y == 10] = 0
        # to NCHW
        X = np.transpose(X, (3, 2, 0, 1)).astype(np.float32)  # (N,3,32,32)
        if normalize:
            X /= 255.0
        return X, y

    Xs, ys = [], []
    if subset in ("train", "all"):
        Xtr, ytr = _load_split("train")
        Xs.append(Xtr)
        ys.append(ytr)
        if include_extra:
            Xe, ye = _load_split("extra")
            Xs.append(Xe)
            ys.append(ye)
    if subset in ("test", "all"):
        Xte, yte = _load_split("test")
        Xs.append(Xte)
        ys.append(yte)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    target_names = [str(i) for i in range(10)]
    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=target_names, description="SVHN (cropped digits)")


def load_stl10(
    *,
    subset="train",            # "train" | "test" | "all"
    include_unlabeled=False,   # add unlabeled to train/all with y=-1
    return_X_y=True,
    as_dataset=False,
    dtype=None,
    shuffle=False,
    random_state=None,
    data_home=None,
    normalize=True,            # scale to [0,1]
):
    """
    STL-10 (binary). Returns X as (N,3,96,96) float in NCHW.
    y: xp.int64 in [0..9], unlabeled (if included) uses -1.
    """
    if dtype is None:
        dtype = DTYPE
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "stl10")
    os.makedirs(ddir, exist_ok=True)

    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    tgz = os.path.join(ddir, "stl10_binary.tar.gz")
    if not os.path.exists(tgz):
        _download(url, tgz)

    extracted_root = os.path.join(ddir, "stl10_binary")
    if not os.path.isdir(extracted_root):
        _extract_tgz(tgz, ddir)

    def _read_images_bin(path, n_expected=None):
        raw = np.fromfile(path, dtype=np.uint8)
        img_sz = 3 * 96 * 96
        if raw.size % img_sz != 0:
            raise ValueError(f"Corrupt STL-10 image file: {path}")
        n = raw.size // img_sz
        if (n_expected is not None) and (n != n_expected):
            # not fatal, but a nice sanity check
            pass

        flat = raw.reshape(n, 3, 96 * 96)  # channel-major blocks
        X = np.empty((n, 3, 96, 96), dtype=np.uint8)
        # pixels inside each channel are column-major
        for c in range(3):
            X[:, c] = flat[:, c].reshape(n, 96, 96, order="F")

        X = X.astype(np.float32)
        if normalize:
            X /= 255.0
        return X

    def _read_labels_bin(path):
        y = np.fromfile(path, dtype=np.uint8).astype(np.int64)
        # labels are 1..10 -> map to 0..9
        y = y - 1
        return y

    def _load_split(split):
        if split == "train":
            X = _read_images_bin(os.path.join(extracted_root, "train_X.bin"))
            y = _read_labels_bin(os.path.join(extracted_root, "train_y.bin"))
            return X, y
        if split == "test":
            X = _read_images_bin(os.path.join(extracted_root, "test_X.bin"))
            y = _read_labels_bin(os.path.join(extracted_root, "test_y.bin"))
            return X, y
        if split == "unlabeled":
            X = _read_images_bin(os.path.join(extracted_root, "unlabeled_X.bin"))
            y = -np.ones((X.shape[0],), dtype=np.int64)
            return X, y
        raise ValueError(split)

    Xs, ys = [], []
    if subset in ("train", "all"):
        Xtr, ytr = _load_split("train")
        Xs.append(Xtr)
        ys.append(ytr)
        if include_unlabeled:
            Xu, yu = _load_split("unlabeled")
            Xs.append(Xu)
            ys.append(yu)
    if subset in ("test", "all"):
        Xte, yte = _load_split("test")
        Xs.append(Xte)
        ys.append(yte)

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)

    X = np.asarray(X, dtype=dtype)
    y = np.asarray(y, dtype=np.int64)
    X, y = _maybe_shuffle(X, y, shuffle, random_state)

    target_names = [
        "airplane", "bird", "car", "cat", "deer",
        "dog", "horse", "monkey", "ship", "truck"
    ]
    if as_dataset:
        return ArrayDataset(X, y, dtype=dtype, to_tensor=True)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=target_names, description="STL-10 (binary)")



def load_ag_news(
    *,
    subset="train",              # "train" | "test" | "all"
    return_X_y=True,
    as_dataset=False,
    dtype=None,                  # kept for convention
    shuffle=False,
    random_state=None,
    data_home=None,
    download=True,
):
    """
    AG News (4 classes). CSV format: label,title,description
    Labels in file are 1..4, we convert to 0..3.

    Note: This uses a widely-used public mirror of the original AG News CSV splits.
    """
    if subset not in ("train", "test", "all"):
        raise ValueError("subset must be 'train', 'test', or 'all'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "ag_news")
    os.makedirs(ddir, exist_ok=True)

    # Mirror used by multiple toolchains (torchtext historically referenced similar splits).
    urls = {
        "train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
        "test":  "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    }

    def _load_split(split):
        fpath = os.path.join(ddir, f"{split}.csv")
        if download and not os.path.exists(fpath):
            _download(urls[split], fpath)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"AG News split not found: {fpath} (set download=True or place it manually).")

        texts = []
        labels = []
        for row in _read_csv_rows(fpath, delimiter=","):
            if len(row) < 3:
                continue
            y = int(row[0]) - 1
            title = row[1].strip()
            desc = row[2].strip()
            txt = (title + " " + desc).strip()
            if txt:
                texts.append(txt)
                labels.append(y)
        return texts, np.asarray(labels, dtype=np.int64)

    if subset == "train":
        X, y = _load_split("train")
    elif subset == "test":
        X, y = _load_split("test")
    else:
        Xtr, ytr = _load_split("train")
        Xte, yte = _load_split("test")
        X = Xtr + Xte
        y = np.concatenate([ytr, yte], axis=0)

    if shuffle:
        rng = _get_rng(random_state)
        perm = rng.permutation(len(X))
        X = [X[i] for i in perm]
        y = y[perm]

    if as_dataset:
        return TextLabelDataset(X, y, to_tensor=False)
    if return_X_y:
        return X, y
    return DatasetBundle(X=X, y=y, target_names=["World", "Sports", "Business", "Sci/Tech"], description="AG News")


def load_tiny_shakespeare(
    *,
    return_X_y=True,
    as_dataset=False,
    dtype=None,                  # dtype for X if you want (token ids usually int64)
    shuffle=False,
    random_state=None,
    data_home=None,
    download=True,
    # windowing:
    block_size=128,
    stride=128,
    tokenizer=None,              # optional: must provide encode(text)->list[int]
    return_text=False,
):
    """
    Tiny Shakespeare raw text -> LM windows.

    If return_text=True: returns raw text string.
    Else:
      X: (N, T) int64
      y: (N, T) int64 (next-token targets)

    If tokenizer is None:
      builds a simple char-level vocab.
    """
    home = get_data_home(data_home)
    ddir = os.path.join(home, "tiny_shakespeare")
    os.makedirs(ddir, exist_ok=True)

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    fpath = os.path.join(ddir, "input.txt")

    if download and not os.path.exists(fpath):
        _download(url, fpath)
    if not os.path.exists(fpath):
        raise FileNotFoundError("Tiny Shakespeare not found in cache. Set download=True or place input.txt manually.")

    with open(fpath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if return_text:
        return text

    if tokenizer is not None:
        ids = tokenizer.encode(text)
        ids = np.asarray(ids, dtype=np.int64)
        vocab_size = int(ids.max()) + 1 if ids.size else 0
        meta_vocab = None
    else:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        ids = np.asarray([stoi[ch] for ch in text], dtype=np.int64)
        vocab_size = len(chars)
        meta_vocab = {"itos": chars, "stoi": stoi}

    T = int(block_size)
    S = int(stride)
    if T < 2:
        raise ValueError("block_size must be >= 2")
    if ids.shape[0] < T + 1:
        raise RuntimeError("Text too short for requested block_size")

    X_list = []
    Y_list = []
    n = int(ids.shape[0])
    for start in range(0, n - (T + 1) + 1, S):
        chunk = ids[start:start + T + 1]
        X_list.append(chunk[:-1])
        Y_list.append(chunk[1:])

    X = np.stack(X_list, axis=0).astype(np.int64, copy=False)
    y = np.stack(Y_list, axis=0).astype(np.int64, copy=False)

    if shuffle:
        rng = _get_rng(random_state)
        perm = rng.permutation(X.shape[0])
        X = X[perm]
        y = y[perm]

    # dtype arg is mostly irrelevant for token ids, but keep the convention:
    if dtype is not None:
        # only apply if user really wants (e.g., int32); default stays int64
        X = np.asarray(X, dtype=dtype)
        y = np.asarray(y, dtype=dtype)
    else:
        # enforce int64 by default
        X = np.asarray(X, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)

    if as_dataset:
        # Dense windows -> ArrayDataset works perfectly
        return ArrayDataset(X, y, dtype=X.dtype, to_tensor=True)

    if return_X_y:
        return X, y

    out = DatasetBundle(X=X, y=y, description="Tiny Shakespeare LM windows")
    #out.vocab_size = vocab_size
    #out.vocab = meta_vocab
    return out


class OxfordIIITPetDataset(Dataset):
    """
    Returns dict sample:
      {
        "image_path": str,
        "label": int64 (0..36)          [if classification]
        "bin_label": int64 (0=cat,1=dog) [optional]
        "mask_path": str                [if segmentation]
        "image_id": str
      }
    Keeps images/masks on disk as paths.
    """
    def __init__(self, root, *, subset="trainval", task="classification",
                 include_binary=False, to_tensor=False):
        super().__init__(to_tensor=to_tensor)
        self.root = root
        self.task = task
        self.include_binary = include_binary

        img_dir = os.path.join(root, "images")
        ann_dir = os.path.join(root, "annotations")
        split_path = os.path.join(ann_dir, f"{subset}.txt")
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Missing Pet split file: {split_path}")
        self.img_dir = img_dir
        self.trimap_dir = os.path.join(ann_dir, "trimaps")

        self.ids = []
        self.labels = []
        self.bin_labels = []

        # format: <image_id> <class_id> <species_id> <breed_id?>
        # torchvision uses: image, label, binary_label, _ (unused)
        with open(split_path, "r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                image_id = parts[0]
                label = int(parts[1]) - 1
                bin_label = int(parts[2]) - 1  # 1=cat,2=dog in file -> 0,1
                self.ids.append(image_id)
                self.labels.append(label)
                self.bin_labels.append(bin_label)

        self.labels = np.asarray(self.labels, dtype=np.int64)
        self.bin_labels = np.asarray(self.bin_labels, dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        out = {"image_path": img_path, "image_id": image_id}

        if self.task in ("classification", "both"):
            out["label"] = np.asarray(int(self.labels[idx]), dtype=np.int64)
            if self.include_binary:
                out["bin_label"] = np.asarray(int(self.bin_labels[idx]), dtype=np.int64)

        if self.task in ("segmentation", "both"):
            # trimaps png: typically values {1,2,3} in file
            mask_path = os.path.join(self.trimap_dir, f"{image_id}.png")
            out["mask_path"] = mask_path

        return out


def load_oxford_iiit_pet(
    *,
    subset="trainval",            # "trainval" | "test"
    return_X_y=True,
    as_dataset=False,
    dtype=None,                   # kept for convention
    shuffle=False,
    random_state=None,
    data_home=None,
    task="classification",        # "classification" | "segmentation" | "both"
    include_binary=False,
    download=True,
):
    """
    Oxford-IIIT Pet.
    Return types:
      - return_X_y=True:
          X = list[str image paths]
          y = np.int64 labels (classification) OR list mask paths (segmentation) OR dict list (both)
      - as_dataset=True: OxfordIIITPetDataset (dict samples w/ paths)
    """
    if subset not in ("trainval", "test"):
        raise ValueError("subset must be 'trainval' or 'test'")
    if task not in ("classification", "segmentation", "both"):
        raise ValueError("task must be 'classification', 'segmentation', or 'both'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "oxford-iiit-pet")
    os.makedirs(ddir, exist_ok=True)

    url_images = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    url_anns   = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz"

    images_tgz = os.path.join(ddir, "images.tar.gz")
    anns_tgz   = os.path.join(ddir, "annotations.tar.gz")

    img_dir = os.path.join(ddir, "images")
    ann_dir = os.path.join(ddir, "annotations")

    if download and not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
        if not os.path.exists(images_tgz):
            _download(url_images, images_tgz)
        if not os.path.exists(anns_tgz):
            _download(url_anns, anns_tgz)
        _extract_tgz(images_tgz, ddir)
        _extract_tgz(anns_tgz, ddir)

    if not (os.path.isdir(img_dir) and os.path.isdir(ann_dir)):
        raise FileNotFoundError("Oxford-IIIT Pet not found in cache. Set download=True or extract into <data_home>/oxford-iiit-pet/.")

    if as_dataset:
        return OxfordIIITPetDataset(ddir, subset=subset, task=task, include_binary=include_binary, to_tensor=False)

    ds = OxfordIIITPetDataset(ddir, subset=subset, task=task, include_binary=include_binary, to_tensor=False)

    # Build X, y in the most useful shapes
    X = []
    if task == "classification":
        y = np.zeros((len(ds),), dtype=np.int64)
        for i in range(len(ds)):
            s = ds[i]
            X.append(s["image_path"])
            y[i] = int(s["label"])
        if shuffle:
            rng = _get_rng(random_state)
            perm = rng.permutation(len(X))
            X = [X[i] for i in perm]
            y = y[perm]
        if return_X_y:
            return X, y
        return DatasetBundle(X=X, y=y, description=f"Oxford-IIIT Pet ({subset}, classification)")

    elif task == "segmentation":
        y = []
        for i in range(len(ds)):
            s = ds[i]
            X.append(s["image_path"])
            y.append(s["mask_path"])
        if shuffle:
            rng = _get_rng(random_state)
            perm = rng.permutation(len(X))
            X = [X[i] for i in perm]
            y = [y[i] for i in perm]
        if return_X_y:
            return X, y
        return DatasetBundle(X=X, y=y, description=f"Oxford-IIIT Pet ({subset}, segmentation)")

    else:  # both
        y = []
        for i in range(len(ds)):
            s = ds[i]
            X.append(s["image_path"])
            tgt = {"label": int(s["label"]), "mask_path": s["mask_path"], "image_id": s["image_id"]}
            if include_binary and "bin_label" in s:
                tgt["bin_label"] = int(s["bin_label"])
            y.append(tgt)
        if shuffle:
            rng = _get_rng(random_state)
            perm = rng.permutation(len(X))
            X = [X[i] for i in perm]
            y = [y[i] for i in perm]
        if return_X_y:
            return X, y
        return DatasetBundle(X=X, y=y, description=f"Oxford-IIIT Pet ({subset}, both)")


_VOC_CLASSES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"
]
_VOC_CLASS_TO_IDX = {c: i for i, c in enumerate(_VOC_CLASSES)}


def _parse_voc2007_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in _VOC_CLASS_TO_IDX:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin"))
        ymin = float(bnd.findtext("ymin"))
        xmax = float(bnd.findtext("xmax"))
        ymax = float(bnd.findtext("ymax"))
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(_VOC_CLASS_TO_IDX[name])

    boxes = np.asarray(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
    return boxes, labels

class VOC2007DetectionDataset(Dataset):
    """
    Map-style dataset returning dict samples:
      {"image_path": str, "boxes": (N,4) float32, "labels": (N,) int64, "image_id": str}

    Note: This keeps images on disk (paths). Load/augment in transforms or collate.
    """
    def __init__(self, voc_root, *, subset="trainval", to_tensor=False):
        super().__init__(to_tensor=to_tensor)
        base = os.path.join(voc_root, "VOCdevkit", "VOC2007")
        self.img_dir = os.path.join(base, "JPEGImages")
        self.ann_dir = os.path.join(base, "Annotations")
        split_file = os.path.join(base, "ImageSets", "Main", f"{subset}.txt")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Missing VOC split file: {split_file}")

        with open(split_file, "r", encoding="utf-8", errors="replace") as f:
            self.ids = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        image_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{image_id}.jpg")
        xml_path = os.path.join(self.ann_dir, f"{image_id}.xml")
        boxes, labels = _parse_voc2007_xml(xml_path)
        return {"image_path": img_path, "boxes": boxes, "labels": labels, "image_id": image_id}


def load_voc2007_detection(
    *,
    subset="trainval",          # "train" | "val" | "trainval" | "test"
    return_X_y=True,
    as_dataset=False,
    dtype=None,                 # kept for convention; boxes are float32, labels int64
    shuffle=False,
    random_state=None,
    data_home=None,
    download=True,
):
    """
    VOC2007 detection.
    Return types:
      - return_X_y=True: X=list[str image paths], y=list[{"boxes","labels","image_id"}]
      - as_dataset=True: VOC2007DetectionDataset (returns dict samples)
    """
    if subset not in ("train", "val", "trainval", "test"):
        raise ValueError("subset must be 'train', 'val', 'trainval', or 'test'")

    home = get_data_home(data_home)
    ddir = os.path.join(home, "voc2007")
    os.makedirs(ddir, exist_ok=True)

    url_trainval = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    url_test     = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar"

    trainval_tgz = os.path.join(ddir, "VOCtrainval_06-Nov-2007.tar")
    test_tgz     = os.path.join(ddir, "VOCtest_06-Nov-2007.tar")

    expected_root = os.path.join(ddir, "VOCdevkit", "VOC2007")

    if download and not os.path.isdir(expected_root):
        # train/val/trainval live in trainval tar
        if not os.path.exists(trainval_tgz):
            _download(url_trainval, trainval_tgz)
        _extract_tgz(trainval_tgz, ddir)

        # test split is separate tar
        if subset == "test" and not os.path.exists(os.path.join(ddir, "VOCdevkit", "VOC2007", "ImageSets", "Main", "test.txt")):
            if not os.path.exists(test_tgz):
                _download(url_test, test_tgz)
            _extract_tgz(test_tgz, ddir)

    if not os.path.isdir(expected_root):
        raise FileNotFoundError("VOC2007 not found in cache. Set download=True or extract into <data_home>/voc2007/.")

    if as_dataset:
        return VOC2007DetectionDataset(ddir, subset=subset, to_tensor=False)

    # Build X, y lists
    ds = VOC2007DetectionDataset(ddir, subset=subset, to_tensor=False)
    X = []
    y = []
    for i in range(len(ds)):
        s = ds[i]
        X.append(s["image_path"])
        y.append({"boxes": s["boxes"], "labels": s["labels"], "image_id": s["image_id"]})

    if shuffle:
        rng = _get_rng(random_state)
        perm = rng.permutation(len(X))
        X = [X[i] for i in perm]
        y = [y[i] for i in perm]

    if return_X_y:
        return X, y

    return DatasetBundle(X=X, y=y, target_names=_VOC_CLASSES, description=f"VOC2007 detection ({subset})")


# ----------------------------
# Registry 
# ----------------------------
_DATASETS = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer_wisconsin": load_breast_cancer_wisconsin,
    "digits": load_digits,
    "titanic": load_titanic,
    "adult": load_adult,
    "california_housing": load_california_housing,
    "diabetes": load_diabetes,
    "ames_housing": load_ames_housing,
    "20newsgroups": load_20newsgroups,
    "sms_spam": load_sms_spam,
    "imdb_reviews": load_imdb_reviews,
    "mnist": load_mnist,
    "fashion_mnist": load_fashion_mnist,
    "olivetti_faces": load_olivetti_faces,
    "movielens": load_movielens,
    "goodbooks_10k": load_goodbooks_10k,
    "cifar10": load_cifar10,
    "cifar100": load_cifar100,
    "svhn": load_svhn,
    "stl10": load_stl10,
    "ag_news": load_ag_news,
    "tiny_shakespeare": load_tiny_shakespeare,
    "oxford_iiit_pet": load_oxford_iiit_pet,
    "voc2007_detection": load_voc2007_detection
}


def list_datasets():
    return sorted(_DATASETS.keys())


def load(name: str, **kwargs):
    name = name.strip().lower()
    if name not in _DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {list_datasets()}")
    return _DATASETS[name](**kwargs)