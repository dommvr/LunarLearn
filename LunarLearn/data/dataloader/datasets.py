import os
import numpy as np
from PIL import Image
import csv
import json
import random
from collections import OrderedDict
from dataclasses import dataclass

import LunarLearn.core.backend.backend as backend
from LunarLearn.core import Tensor
from LunarLearn.data.dataloader.utils import (_to_tensor_tree,
                                              _is_np_scalar,
                                              _pad_1d,
                                              _read_text_file,
                                              _tokenize_text)

DTYPE = np.dtype(backend.DTYPE)
USING = backend.USING


@dataclass
class DatasetBundle:
    X: object
    y: object
    feature_names: list | None = None
    target_names: list | None = None
    description: str | None = None


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
    def __init__(self, X, Y, to_tensor=True):
        super().__init__(to_tensor)
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
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

    generator_fn should return an iterator/generator yielding samples.
    Samples can be (x,y), tuples, dicts, etc.
    """
    def __init__(self, generator_fn, length=None, to_tensor=True):
        super().__init__(to_tensor)
        self.generator_fn = generator_fn
        self.length = length

    def __len__(self):
        # Only return an int if it's known
        if self.length is None:
            raise TypeError("GeneratorDataset length is unknown (set length=... to enable __len__).")
        return int(self.length)

    def __iter__(self):
        for sample in self.generator_fn():
            if self.to_tensor:
                sample = _to_tensor_tree(sample)
            yield sample


class DictDataset(Dataset):
    """
    Map-style dataset returning dict samples.

    Supports:
      - list[dict] (most common)
      - dict of arrays/lists (columnar): {"image": X, "boxes": B, ...}
    """
    def __init__(self, data, to_tensor=True):
        super().__init__(to_tensor)
        if isinstance(data, dict):
            # columnar storage
            self._columnar = True
            self.data = data
            # infer length from first key
            k0 = next(iter(data.keys()))
            self.m = len(data[k0])
        elif isinstance(data, (list, tuple)):
            self._columnar = False
            self.data = list(data)
            self.m = len(self.data)
        else:
            raise TypeError("DictDataset expects list[dict] or dict-of-arrays")

    def __len__(self):
        return self.m

    def __getitem__(self, idx):
        if self._columnar:
            sample = {k: v[idx] for k, v in self.data.items()}
        else:
            sample = self.data[idx]

        # convert numpy->xp if needed (light-touch)
        for k, v in list(sample.items()):
            if _is_np_scalar:
                sample[k] = np.asarray(v)
        return _to_tensor_tree(sample) if self.to_tensor else sample


class SequenceDataset(Dataset):
    """
    Map-style dataset for token sequences (ids) and optional labels.

    Each item returns a dict:
      {
        "input_ids": (L,) or (max_len,) int64,
        "attention_mask": (max_len,) int64 (if padded),
        "length": int64,
        "label": int64 or one-hot (optional)
      }

    If max_len is None -> returns variable-length input_ids, no mask.
    If max_len is set -> pads/truncates to max_len and returns mask.
    """
    def __init__(
        self,
        sequences,
        labels=None,
        max_len=None,
        pad_id=0,
        truncate=True,
        one_hot=False,
        num_classes=None,
        to_tensor=True,
    ):
        super().__init__(to_tensor)
        self.sequences = sequences
        self.labels = labels
        self.max_len = max_len
        self.pad_id = int(pad_id)
        self.truncate = truncate
        self.one_hot = one_hot
        self.num_classes = num_classes

        if labels is not None and len(labels) != len(sequences):
            raise ValueError("labels must have same length as sequences")

        if one_hot and (num_classes is None):
            # infer if possible
            if labels is None:
                raise ValueError("one_hot=True requires labels and num_classes (or inferable labels)")
            self.num_classes = int(max(labels)) + 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        ids = self.sequences[idx]
        ids = np.asarray(ids, dtype=np.int64)

        if self.max_len is None:
            sample = {
                "input_ids": ids,
                "length": np.asarray(ids.shape[0], dtype=np.int64),
            }
        else:
            padded, mask, L = _pad_1d(ids, int(self.max_len), pad_id=self.pad_id, truncate=self.truncate)
            sample = {
                "input_ids": padded,
                "attention_mask": mask,
                "length": L,
            }

        if self.labels is not None:
            lab = self.labels[idx]
            if self.one_hot:
                y = np.eye(int(self.num_classes), dtype=DTYPE)[int(lab)]
            else:
                y = np.asarray(int(lab), dtype=np.int64)
            sample["label"] = y

        return _to_tensor_tree(sample) if self.to_tensor else sample
    

class PackedSequenceDataset(Dataset):
    """
    Packs many token sequences into one long stream and returns fixed-size blocks.

    Input:
      sequences: list of sequences (list[int] or xp arrays)
      sep_id: optional separator inserted between sequences (e.g., EOS)
      block_size: output block length
      stride: step between blocks (default=block_size)
      next_token: if True, returns x/y shifted blocks
      drop_last: if True, drop trailing partial blocks

    Output:
      - if next_token:
          {"input_ids": (block_size,), "labels": (block_size,)}
      - else:
          {"input_ids": (block_size,)}
    """
    def __init__(
        self,
        sequences,
        block_size,
        stride=None,
        sep_id=None,
        next_token=True,
        drop_last=True,
        to_tensor=True,
    ):
        super().__init__(to_tensor)

        self.block_size = int(block_size)
        self.stride = int(stride) if stride is not None else int(block_size)
        self.sep_id = None if sep_id is None else int(sep_id)
        self.next_token = bool(next_token)
        self.drop_last = bool(drop_last)

        # Build one long token stream
        parts = []
        for seq in sequences:
            ids = np.asarray(seq, dtype=np.int64).ravel()
            parts.append(ids)
            if self.sep_id is not None:
                parts.append(np.asarray([self.sep_id], dtype=np.int64))

        if len(parts) == 0:
            self.stream = np.asarray([], dtype=np.int64)
        else:
            self.stream = np.concatenate(parts, axis=0).astype(np.int64)

        L = int(self.stream.shape[0])

        need = self.block_size + (1 if self.next_token else 0)
        if L < need:
            self.starts = [] if self.drop_last else [0]
        else:
            self.starts = list(range(0, L - need + 1, self.stride))

        if (not self.drop_last) and self.starts:
            # ensure we cover the tail with one last block start
            last = self.starts[-1]
            tail_start = max(0, L - need)
            if tail_start > last:
                self.starts.append(tail_start)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, idx):
        s = int(self.starts[int(idx)])
        if self.next_token:
            chunk = self.stream[s : s + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            sample = {"input_ids": x, "labels": y}
        else:
            x = self.stream[s : s + self.block_size]
            sample = {"input_ids": x}

        return _to_tensor_tree(sample) if self.to_tensor else sample


class ImageDataset(Dataset):
    def __init__(
        self,
        path,
        size=(64, 64),
        to_tensor=True,
        one_hot=False,          # better default
        transform=None,
        target_transform=None,
        return_path=False,
        label_dtype=np.int64,   # int labels by default
        x_dtype=None,           # defaults to DTYPE
    ):
        super().__init__(to_tensor)
        self.folder_path = path
        self.size = size
        self.to_tensor = to_tensor
        self.one_hot = one_hot
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.label_dtype = label_dtype
        self.x_dtype = DTYPE if x_dtype is None else x_dtype

        self.class_names = sorted(
            d for d in os.listdir(self.folder_path)
            if os.path.isdir(os.path.join(self.folder_path, d))
        )
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}

        self.samples = []
        for cls_name in self.class_names:
            cls_folder = os.path.join(self.folder_path, cls_name)
            files = sorted(
                f for f in os.listdir(cls_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            )
            for f in files:
                self.samples.append((os.path.join(cls_folder, f), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.size, Image.Resampling.LANCZOS)

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))  # (C,H,W)

        # optional transform on numpy/CPU
        if self.transform is not None:
            x = self.transform(x)

        # labels
        if self.one_hot:
            y = np.zeros((len(self.class_names),), dtype=np.float32)
            y[label] = 1.0
            if self.target_transform is not None:
                y = self.target_transform(y)
            y = np.asarray(y, dtype=self.x_dtype)  # one-hot is float
        else:
            y = label
            if self.target_transform is not None:
                y = self.target_transform(y)
            y = np.asarray(y, dtype=self.label_dtype)

        x = np.asarray(x, dtype=self.x_dtype)

        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)

        if self.return_path:
            return x, y, img_path
        return x, y
    

class CSVImageDataset(Dataset):
    def __init__(
        self,
        csv_file,
        img_root="",
        size=(64, 64),
        to_tensor=True,
        one_hot=False,
        delimiter=",",
        num_classes=None,
        transform=None,
        target_transform=None,
        return_path=False,
        label_dtype=np.int64,
        x_dtype=None,
    ):
        super().__init__(to_tensor)
        self.csv_file = csv_file
        self.img_root = img_root
        self.size = size
        self.to_tensor = to_tensor
        self.one_hot = one_hot
        self.transform = transform
        self.target_transform = target_transform
        self.return_path = return_path
        self.label_dtype = label_dtype
        self.x_dtype = DTYPE if x_dtype is None else x_dtype

        self.samples = []
        with open(csv_file, "r", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                p = row["path"]
                path = p if os.path.isabs(p) else os.path.join(img_root, p)
                label = int(row["label"])
                self.samples.append((path, label))

        if num_classes is not None:
            self.num_classes = int(num_classes)
        else:
            labels = [lbl for _, lbl in self.samples]
            self.num_classes = (max(labels) + 1) if labels else 0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        img = img.resize(self.size, Image.Resampling.LANCZOS)

        x = np.asarray(img, dtype=np.float32) / 255.0
        x = np.transpose(x, (2, 0, 1))

        if self.transform is not None:
            x = self.transform(x)

        if self.one_hot:
            y = np.zeros((self.num_classes,), dtype=np.float32)
            y[label] = 1.0
            if self.target_transform is not None:
                y = self.target_transform(y)
            y = np.asarray(y, dtype=self.x_dtype)
        else:
            y = label
            if self.target_transform is not None:
                y = self.target_transform(y)
            y = np.asarray(y, dtype=self.label_dtype)

        x = np.asarray(x, dtype=self.x_dtype)

        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)

        if self.return_path:
            return x, y, img_path
        return x, y


class TextFolderDataset(Dataset):
    """
    Folder structure:
      root/
        class_a/*.txt
        class_b/*.txt

    Returns either:
      - {"text": str, "label": ...} if tokenizer is None
      - token dict if tokenizer provided (like SequenceDataset)
    """
    def __init__(
        self,
        root,
        tokenizer=None,
        max_len=None,
        pad_id=0,
        truncate=True,
        one_hot=True,
        encoding="utf-8",
        extensions=(".txt",),
        to_tensor=True,
        dtype=None,
    ):
        super().__init__(to_tensor)
        if dtype is None:
            dtype = DTYPE

        self.root = root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = int(pad_id)
        self.truncate = truncate
        self.one_hot = one_hot
        self.encoding = encoding
        self.extensions = tuple(e.lower() for e in extensions)
        self.dtype = dtype

        self.class_names = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.num_classes = len(self.class_names)

        self.samples = []
        for cls in self.class_names:
            folder = os.path.join(root, cls)
            for fn in os.listdir(folder):
                if fn.lower().endswith(self.extensions):
                    self.samples.append((os.path.join(folder, fn), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        text = _read_text_file(path, encoding=self.encoding)

        if self.tokenizer is None:
            sample = {"text": text}
        else:
            ids = _tokenize_text(text, self.tokenizer)
            if self.max_len is None:
                sample = {"input_ids": ids, "length": np.asarray(ids.shape[0], dtype=np.int64)}
            else:
                padded, mask, L = _pad_1d(ids, int(self.max_len), pad_id=self.pad_id, truncate=self.truncate)
                sample = {"input_ids": padded, "attention_mask": mask, "length": L}

        if self.one_hot:
            y = np.eye(self.num_classes, dtype=self.dtype)[int(label)]
        else:
            y = np.asarray(int(label), dtype=np.int64)
        sample["label"] = y

        return _to_tensor_tree(sample) if self.to_tensor else sample
    

class CSVTextDataset(Dataset):
    """
    CSV-based text dataset.
    Supports either:
      - inline text column: text_col="text"
      - file path column:   text_col="path" + text_root

    Required label_col (default "label") for supervised.
    If label_col is None -> returns unsupervised samples.

    Returns dict samples:
      - raw text: {"text": str, ...}
      - tokenized: {"input_ids", "attention_mask", "length", ...}
    """
    def __init__(
        self,
        csv_file,
        text_col="text",
        label_col="label",
        delimiter=",",
        text_root="",
        tokenizer=None,
        max_len=None,
        pad_id=0,
        truncate=True,
        one_hot=False,
        num_classes=None,
        encoding="utf-8",
        to_tensor=True,
        dtype=None,
    ):
        super().__init__(to_tensor)
        if dtype is None:
            dtype = DTYPE

        self.csv_file = csv_file
        self.text_col = text_col
        self.label_col = label_col
        self.delimiter = delimiter
        self.text_root = text_root
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = int(pad_id)
        self.truncate = truncate
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.encoding = encoding
        self.dtype = dtype

        self.rows = []
        with open(csv_file, "r", encoding=encoding, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                self.rows.append(row)

        if self.label_col is not None and self.one_hot and self.num_classes is None:
            labels = [int(r[self.label_col]) for r in self.rows]
            self.num_classes = int(max(labels)) + 1

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        raw = row[self.text_col]

        # If it's a path, load file
        if self.text_root or os.path.exists(raw):
            candidate = os.path.join(self.text_root, raw) if self.text_root else raw
            if os.path.exists(candidate):
                text = _read_text_file(candidate, encoding=self.encoding)
            else:
                text = raw
        else:
            text = raw

        if self.tokenizer is None:
            sample = {"text": text}
        else:
            ids = _tokenize_text(text, self.tokenizer)
            if self.max_len is None:
                sample = {"input_ids": ids, "length": np.asarray(ids.shape[0], dtype=np.int64)}
            else:
                padded, mask, L = _pad_1d(ids, int(self.max_len), pad_id=self.pad_id, truncate=self.truncate)
                sample = {"input_ids": padded, "attention_mask": mask, "length": L}

        if self.label_col is not None:
            lab = int(row[self.label_col])
            if self.one_hot:
                y = np.eye(int(self.num_classes), dtype=self.dtype)[lab]
            else:
                y = np.asarray(lab, dtype=np.int64)
            sample["label"] = y

        return _to_tensor_tree(sample) if self.to_tensor else sample


class JSONLTextDataset(Dataset):
    """
    JSONL dataset (one JSON object per line).

    Supports arbitrary schemas:
      - classification: {"text":..., "label":...}
      - QA: {"context":..., "question":..., "answer":...}
      - captioning: {"text":..., "image_path":...} (you can extend)

    If tokenizer is set and text_key is provided, tokenizes that field.
    If text_key is None, returns the full dict (and optionally tensorifies numeric arrays).
    """
    def __init__(
        self,
        jsonl_file,
        text_key="text",
        label_key="label",
        tokenizer=None,
        max_len=None,
        pad_id=0,
        truncate=True,
        one_hot=False,
        num_classes=None,
        encoding="utf-8",
        to_tensor=True,
        dtype=None,
    ):
        super().__init__(to_tensor)
        if dtype is None:
            dtype = DTYPE

        self.jsonl_file = jsonl_file
        self.text_key = text_key
        self.label_key = label_key
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.pad_id = int(pad_id)
        self.truncate = truncate
        self.one_hot = one_hot
        self.num_classes = num_classes
        self.encoding = encoding
        self.dtype = dtype

        self.items = []
        with open(jsonl_file, "r", encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))

        if self.label_key is not None and self.one_hot and self.num_classes is None:
            labels = [int(it[self.label_key]) for it in self.items if self.label_key in it]
            if labels:
                self.num_classes = int(max(labels)) + 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = dict(self.items[idx])  # copy

        # If tokenizing a specific key
        if self.tokenizer is not None and self.text_key is not None and self.text_key in it:
            text = it[self.text_key]
            ids = _tokenize_text(text, self.tokenizer)
            if self.max_len is None:
                out = {"input_ids": ids, "length": np.asarray(ids.shape[0], dtype=np.int64)}
            else:
                padded, mask, L = _pad_1d(ids, int(self.max_len), pad_id=self.pad_id, truncate=self.truncate)
                out = {"input_ids": padded, "attention_mask": mask, "length": L}

            # optionally carry raw text too
            out["text"] = text

            # label handling
            if self.label_key is not None and self.label_key in it:
                lab = int(it[self.label_key])
                if self.one_hot:
                    y = np.eye(int(self.num_classes), dtype=DTYPE)[lab]
                else:
                    y = np.asarray(lab, dtype=np.int64)
                out["label"] = y

            # keep extra fields (context/question/answer, ids, etc.)
            for k, v in it.items():
                if k not in (self.text_key, self.label_key):
                    out[k] = v
            return _to_tensor_tree(out) if self.to_tensor else out

        # otherwise: return full dict
        return _to_tensor_tree(it) if self.to_tensor else it


class TextFileDataset(Dataset):
    """
    Single text file dataset.

    Modes:
      - line_mode=True: each line is a sample
      - line_mode=False: the whole file is one stream and you emit windows

    Tokenization:
      - If tokenizer is None: returns {"text": ...}
      - If tokenizer provided: returns token samples.

    LM options:
      - block_size: fixed-length chunk size (required for token stream mode)
      - next_token: if True, returns {"input_ids": x, "labels": y} where y is shifted
    """
    def __init__(
        self,
        path,
        tokenizer=None,
        encoding="utf-8",
        line_mode=True,
        strip_lines=True,

        # token stream options
        block_size=None,
        stride=None,          # if None: stride=block_size
        drop_last=True,
        next_token=False,

        pad_id=0,             # only used in line_mode with max_len
        max_len=None,         # optional per-line padding/truncation
        truncate=True,

        to_tensor=True,
    ):
        super().__init__(to_tensor)
        self.path = path
        self.tokenizer = tokenizer
        self.encoding = encoding
        self.line_mode = line_mode
        self.strip_lines = strip_lines

        self.block_size = block_size
        self.stride = stride
        self.drop_last = drop_last
        self.next_token = next_token

        self.pad_id = int(pad_id)
        self.max_len = max_len
        self.truncate = truncate

        if not os.path.exists(path):
            raise FileNotFoundError(path)

        if not line_mode:
            if tokenizer is None:
                raise ValueError("tokenizer is required for line_mode=False (stream/window mode)")
            if block_size is None:
                raise ValueError("block_size is required for line_mode=False")

        if self.line_mode:
            with open(path, "r", encoding=encoding) as f:
                lines = f.readlines()
            if strip_lines:
                lines = [ln.strip() for ln in lines]
                lines = [ln for ln in lines if ln != ""]
            self.lines = lines
        else:
            text = _read_text_file(path, encoding=encoding)
            self.tokens = _tokenize_text(text, tokenizer)  # (L,)
            self.tokens = self.tokens.astype(np.int64)
            self.block_size = int(block_size)
            self.stride = int(stride) if stride is not None else int(block_size)

            # compute windows
            L = int(self.tokens.shape[0])
            # if next_token, we need block_size+1 tokens to make x/y
            need = self.block_size + (1 if self.next_token else 0)
            starts = list(range(0, max(0, L - need + 1), self.stride))
            if not starts:
                starts = [0] if (not self.drop_last and L >= 1) else []
            self.starts = starts

    def __len__(self):
        if self.line_mode:
            return len(self.lines)
        return len(self.starts)

    def __getitem__(self, idx):
        if self.line_mode:
            text = self.lines[int(idx)]
            if self.tokenizer is None:
                sample = {"text": text}
                return _to_tensor_tree(sample) if self.to_tensor else sample

            ids = _tokenize_text(text, self.tokenizer).astype(np.int64)

            # optional per-line pad/truncate
            if self.max_len is not None:
                T = int(self.max_len)
                L = int(ids.shape[0])
                if self.truncate and L > T:
                    ids = ids[:T]
                    L = T
                out = np.full((T,), self.pad_id, dtype=np.int64)
                out[:L] = ids
                mask = np.zeros((T,), dtype=np.int64)
                mask[:L] = 1
                sample = {"input_ids": out, "attention_mask": mask, "length": np.asarray(L, dtype=np.int64)}
            else:
                sample = {"input_ids": ids, "length": np.asarray(ids.shape[0], dtype=np.int64)}

            return _to_tensor_tree(sample) if self.to_tensor else sample

        # stream/window mode
        s = int(self.starts[int(idx)])
        if self.next_token:
            chunk = self.tokens[s : s + self.block_size + 1]
            x = chunk[:-1]
            y = chunk[1:]
            sample = {"input_ids": x, "labels": y}
        else:
            chunk = self.tokens[s : s + self.block_size]
            sample = {"input_ids": chunk}

        return _to_tensor_tree(sample) if self.to_tensor else sample


class TextLabelDataset(Dataset):
    """
    Map-style dataset for text classification.
    Returns dict samples: {"text": <str>, "label": <int>}
    """
    def __init__(self, texts, labels, to_tensor=False):
        super().__init__(to_tensor=to_tensor)
        self.texts = list(texts)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}


class AugmentedDataset(Dataset):
    """
    Wraps a dataset and applies transform(sample) -> sample.
    Works for samples that are tuples, dicts, etc.
    """
    def __init__(self, dataset, transform=None, to_tensor=None):
        # If to_tensor is None: follow base dataset behavior.
        super().__init__(to_tensor if to_tensor is not None else getattr(dataset, "to_tensor", True))
        self.dataset = dataset
        self.transform = transform
        self._base_to_tensor = getattr(dataset, "to_tensor", False)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]  # can be anything
        if self.transform is not None:
            sample = self.transform(sample)

        # Only tensorify here if:
        # - wrapper wants tensors AND
        # - base dataset didn't already tensorify
        if self.to_tensor and not self._base_to_tensor:
            sample = _to_tensor_tree(sample)
        return sample
    

class AugmentedIterableDataset(IterableDataset):
    """
    Iterable version: applies transform(sample)->sample.
    """
    def __init__(self, dataset, transform=None, to_tensor=None):
        super().__init__(to_tensor if to_tensor is not None else getattr(dataset, "to_tensor", True))
        self.dataset = dataset
        self.transform = transform
        self._base_to_tensor = getattr(dataset, "to_tensor", False)

    def __len__(self):
        # only valid if underlying has a real __len__
        return len(self.dataset)

    def __iter__(self):
        for sample in self.dataset:
            if self.transform is not None:
                sample = self.transform(sample)
            if self.to_tensor and not self._base_to_tensor:
                sample = _to_tensor_tree(sample)
            yield sample


class LazyDataset(Dataset):
    """
    Lazily loads data only when indexed (no preloading).

    Args:
        loader_fn (callable): Function that loads a single item given index.
        length (int): Total number of samples.
        to_tensor (bool): Convert output to Tensors.
    """
    def __init__(self, loader_fn, length, to_tensor=True):
        super().__init__(to_tensor)
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


class ConcatDataset(Dataset):
    """
    Concatenate multiple datasets into one.

    Args:
        datasets (list): List of Dataset objects.
    """
    def __init__(self, datasets, to_tensor=True):
        super().__init__(to_tensor)
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
        x, y = self.datasets[dataset_idx][sample_idx]
        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)
        return x, y


class PairedDataset(Dataset):
    """
    Dataset for paired inputs (e.g., (src, tgt), (image, mask)).

    Args:
        X (array-like or list): Source samples.
        Y (array-like or list): Target samples (must be same length as X).
        to_tensor (bool): Whether to convert outputs to Tensors.
    """
    def __init__(self, X, Y, to_tensor=True):
        super().__init__(to_tensor)
        assert len(X) == len(Y), "X and Y must have the same length"
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]

        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)

        return x, y
    

class MixedDataset(Dataset):
    """
    Dataset that randomly samples from multiple datasets.

    Useful for multitask training or mixing domains.

    Args:
        datasets (list): List of datasets to mix.
        sampling_probs (list, optional): Probabilities for sampling from each dataset.
                                         If None, uniform sampling is used.
    """
    def __init__(self, datasets, sampling_probs=None, to_tensor=True):
        super().__init__(to_tensor)
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
        x, y = dataset[sample_idx]
        if self.to_tensor:
            x = Tensor(x, requires_grad=False)
            y = Tensor(y, requires_grad=False)
        return x, y


class SubsetDataset(Dataset):
    """
    Wraps a dataset and exposes only selected indices.
    Works with any sample type (tuple/dict/etc.).
    """
    def __init__(self, dataset, indices, to_tensor=None):
        # If to_tensor is None: follow base dataset behavior.
        super().__init__(to_tensor if to_tensor is not None else getattr(dataset, "to_tensor", True))
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample = self.dataset[self.indices[idx]]
        # If underlying dataset already returned tensors, leave them.
        if self.to_tensor and not getattr(self.dataset, "to_tensor", False):
            sample = _to_tensor_tree(sample)
        return sample
    

class CacheDataset(Dataset):
    """
    Wraps a map-style dataset and caches __getitem__ results.

    Useful for:
      - ImageDataset decoding/resizing
      - tokenization-heavy text datasets
      - expensive synthetic rendering (if deterministic per index)

    Params:
      max_size: None -> unbounded cache, else LRU with max entries
      copy: if True, attempts shallow copies of dict/list to avoid accidental mutation
    """
    def __init__(self, dataset, max_size=10000, copy=False, to_tensor=None):
        super().__init__(to_tensor if to_tensor is not None else getattr(dataset, "to_tensor", True))
        self.dataset = dataset
        self.max_size = max_size
        self.copy = copy
        self.cache = OrderedDict()

        if isinstance(dataset, IterableDataset):
            raise TypeError("CacheDataset requires a map-style Dataset (iterable datasets can't be indexed).")

    def __len__(self):
        return len(self.dataset)

    def _maybe_copy(self, x):
        if not self.copy:
            return x
        if isinstance(x, dict):
            return dict(x)
        if isinstance(x, list):
            return list(x)
        if isinstance(x, tuple):
            return tuple(x)
        return x

    def __getitem__(self, idx):
        idx = int(idx)
        if idx in self.cache:
            v = self.cache.pop(idx)
            self.cache[idx] = v  # mark as most recently used
            return self._maybe_copy(v)

        v = self.dataset[idx]
        # If the base dataset returns arrays but this wrapper wants tensors:
        if self.to_tensor and not getattr(self.dataset, "to_tensor", False):
            v = _to_tensor_tree(v)

        # store
        self.cache[idx] = v
        if self.max_size is not None and len(self.cache) > int(self.max_size):
            self.cache.popitem(last=False)  # evict LRU
        return self._maybe_copy(v)