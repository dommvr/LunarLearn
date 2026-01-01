import os
import csv
import math
import hashlib
import urllib.request
import tarfile
import numpy as np
import gzip
import zipfile
import pickle

import LunarLearn.core.backend.backend as backend

xp = backend.xp
SEED = backend.SEED


def get_data_home(data_home=None) -> str:
    if data_home is None:
        data_home = os.environ.get("MYML_DATA", os.path.join(os.path.expanduser("~"), ".cache", "myml", "datasets"))
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download(url: str, dst_path: str, *, overwrite=False, timeout=30):
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if (not overwrite) and os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return

    tmp = dst_path + ".tmp"
    with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dst_path)


"""
def _download(url, fpath, *, overwrite=False, expected_md5=None):
    import urllib.request
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    if (not overwrite) and os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        if expected_md5 is None or _md5(fpath) == expected_md5:
            return fpath

    tmp = fpath + ".tmp"
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as w:
        w.write(r.read())
    os.replace(tmp, fpath)

    if expected_md5 is not None:
        got = _md5(fpath)
        if got != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {fpath}: got {got}, expected {expected_md5}")
    return fpath

def _download_from_mirrors(mirrors, filename, fpath, *, expected_md5=None):
    last_err = None
    for base in mirrors:
        url = base.rstrip("/") + "/" + filename
        try:
            return _download(url, fpath, overwrite=True, expected_md5=expected_md5)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download {filename} from mirrors. Last error: {last_err}")
"""


def _download_from_mirrors(mirrors, filename, fpath, *, timeout=30):
    last_err = None
    for base in mirrors:
        url = base.rstrip("/") + "/" + filename
        try:
            return _download(url, fpath, overwrite=True, timeout=timeout)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to download {filename} from mirrors. Last error: {last_err}")


def _get_rng(random_state):
    """
    Return a RandomState-like object for both numpy and cupy.
    """
    if random_state is None:
        random_state = SEED
    # numpy.random.RandomState, cupy.random.RandomState both exist
    return xp.random.RandomState(int(random_state))


def _maybe_shuffle(X, y, shuffle: bool, random_state=None):
    if not shuffle:
        return X, y
    rng = _get_rng(random_state)
    idx = rng.permutation(X.shape[0])
    return X[idx], y[idx]


def _read_csv_rows(path: str, *, delimiter=","):
    with open(path, "r", newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f, delimiter=delimiter)
        for row in reader:
            if not row:
                continue
            yield row


def _to_float(s: str):
    s = s.strip()
    if s == "" or s == "?" or s.lower() == "nan":
        return math.nan
    return float(s)


def _to_int(s: str):
    s = s.strip()
    if s == "" or s == "?" or s.lower() == "nan":
        return None
    return int(s)


def _extract_tgz(tgz_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with tarfile.open(tgz_path, "r:gz") as tf:
        tf.extractall(out_dir)
    return out_dir


def _extract_zip(zpath, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(out_dir)
    return out_dir


def _loadtxt_flexible(path):
    for delim in (",", None, " "):
        try:
            return np.loadtxt(path, delimiter=delim).astype(np.float32)
        except Exception:
            continue
    raise RuntimeError(f"Could not parse numeric file: {path}")


def _parse_idx_gz_images(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    # IDX header: magic(4) = 0x00000803, n(4), rows(4), cols(4)
    magic = int.from_bytes(data[0:4], "big")
    if magic != 0x00000803:
        raise ValueError(f"Bad IDX image magic: {magic}")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=16)
    arr = arr.reshape(n, rows, cols)
    return arr

def _parse_idx_gz_labels(gz_path):
    with gzip.open(gz_path, "rb") as f:
        data = f.read()
    # magic(4)=0x00000801, n(4)
    magic = int.from_bytes(data[0:4], "big")
    if magic != 0x00000801:
        raise ValueError(f"Bad IDX label magic: {magic}")
    n = int.from_bytes(data[4:8], "big")
    arr = np.frombuffer(data, dtype=np.uint8, offset=8)
    arr = arr.reshape(n,)
    return arr


def _unpickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="bytes")