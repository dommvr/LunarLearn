import numpy as np

import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader.utils import (_is_np_array,
                                              _try_stack_np,
                                              _is_np_scalar, 
                                              _is_xp_array, 
                                              _try_stack_any)
from LunarLearn.core import Tensor

xp = backend.xp


def _collate(batch):
    """
    CPU-first collate:
      - dict: collate per key
      - tuple/list: collate per field
      - strings: keep list
      - Tensor: keep list (don't guess how to stack your Tensor safely)
      - np.ndarray: np.stack if possible
      - scalars (python or np scalar): np.asarray
      - None: keep list (optional fields)
      - fallback: keep list
    """
    if len(batch) == 0:
        return batch

    first = batch[0]

    # If any element is None, keep as list (optional fields)
    # (do NOT early-return for whole batch; only for the field we're collating)
    if first is None:
        return list(batch)

    # Dict samples
    if isinstance(first, dict):
        out = {}
        keys = set()
        for b in batch:
            if isinstance(b, dict):
                keys.update(b.keys())
        for k in keys:
            out[k] = _collate([b.get(k, None) if isinstance(b, dict) else None for b in batch])
        return out

    # Tuple/list samples (zip fields)
    if isinstance(first, (tuple, list)):
        fields = list(zip(*batch))
        return type(first)(_collate(list(f)) for f in fields)

    # Strings
    if isinstance(first, str):
        return list(batch)

    # Tensors: keep as list by default (safe, avoids breaking your Tensor semantics)
    if isinstance(first, Tensor):
        return list(batch)

    # NumPy arrays -> stack if possible
    if _is_np_array(first):
        return _try_stack_np(batch)

    # xp arrays (already backend arrays) -> try stack via xp
    if _is_xp_array(first):
        return _try_stack_any(batch)

    # Numeric scalars (python or numpy scalar) -> CPU vector
    if isinstance(first, (int, float, bool)) or _is_np_scalar(first):
        return np.asarray(batch)

    # Fallback: keep list
    return list(batch)


def collate_xy_stack(batch):
    """
    CPU-first (NumPy) collate for classic (x,y) datasets.
    Returns:
      X: np.ndarray stacked if possible else list
      Y: np.ndarray stacked if possible else list
    Then you run: batch = to_backend((X,Y), wrap_tensors=...)
    """
    xs, ys = zip(*batch)

    # If tensors slipped in, treat them as already-decided objects: keep list
    if isinstance(xs[0], Tensor) or isinstance(ys[0], Tensor):
        return list(xs), list(ys)

    # X: try stack (fast path)
    try:
        X = np.stack([np.asarray(x) for x in xs], axis=0)
    except Exception:
        X = [np.asarray(x) for x in xs]

    # Y: stack if possible, else list (ragged)
    try:
        Y = np.stack([np.asarray(y) for y in ys], axis=0)
    except Exception:
        Y = [np.asarray(y) for y in ys]

    return X, Y


def collate_detection(batch):
    if len(batch) == 0:
        return {"images": [], "targets": []}

    # accept either "image" or "images"
    key_img = "image" if ("image" in batch[0]) else "images"

    first_img = batch[0][key_img]

    # Images
    if isinstance(first_img, Tensor):
        images = [b[key_img] for b in batch]  # keep list of Tensors
    else:
        try:
            images = np.stack([np.asarray(b[key_img]) for b in batch], axis=0)
        except Exception:
            images = [np.asarray(b[key_img]) for b in batch]

    # Targets (ragged -> list of dicts)
    targets = []
    for b in batch:
        t = {
            "boxes": np.asarray(b["boxes"], dtype=np.float32),
            "labels": np.asarray(b["labels"], dtype=np.int64),
        }
        # keep optional extra fields per target if present
        for k, v in b.items():
            if k not in (key_img, "boxes", "labels"):
                t[k] = v
        targets.append(t)

    return {"images": images, "targets": targets}


def collate_segmentation(batch, image_key="image", mask_key="mask"):
    """
    CPU-first segmentation collate.
    Returns:
      {"images": (B,C,H,W) np or list, "masks": (B,H,W) np or list}
    """
    if len(batch) == 0:
        return {"images": [], "masks": []}

    first_img = batch[0][image_key]
    first_msk = batch[0][mask_key]

    # If tensors appear, keep lists (don't guess stacking)
    if isinstance(first_img, Tensor):
        images = [b[image_key] for b in batch]
    else:
        imgs = [np.asarray(b[image_key]) for b in batch]
        try:
            images = np.stack(imgs, axis=0)
        except Exception:
            images = imgs  # ragged fallback

    if isinstance(first_msk, Tensor):
        masks = [b[mask_key] for b in batch]
    else:
        msks = [np.asarray(b[mask_key]) for b in batch]
        try:
            masks = np.stack(msks, axis=0)
        except Exception:
            masks = msks

    return {"images": images, "masks": masks}


def collate_pad_tokens(batch, pad_id=0, label_key="label", ids_key="input_ids"):
    """
    CPU-first token padding collate.
    Returns:
      {
        "input_ids": (B,T) np.int64,
        "lengths": (B,) np.int64,
        "attention_mask": (B,T) np.int64,
        "labels": ... optional
      }
    """
    if len(batch) == 0:
        return {"input_ids": np.zeros((0, 0), dtype=np.int64),
                "lengths": np.zeros((0,), dtype=np.int64),
                "attention_mask": np.zeros((0, 0), dtype=np.int64)}

    # Gather ids as np.int64 vectors
    ids_list = []
    for b in batch:
        ids = b[ids_key]
        if isinstance(ids, Tensor):
            # keep tensor as-is: cannot safely pad without assuming tensor is CPU/contiguous
            # fallback: keep list of tensors
            ids_list.append(ids)
        else:
            arr = np.asarray(ids, dtype=np.int64).reshape(-1)
            ids_list.append(arr)

    # If token ids are tensors, return a safe ragged batch
    if isinstance(ids_list[0], Tensor):
        out = {"input_ids": ids_list}
        if label_key is not None and label_key in batch[0]:
            out["labels"] = [b.get(label_key, None) for b in batch]
        return out

    lengths = np.asarray([int(x.shape[0]) for x in ids_list], dtype=np.int64)
    T = int(lengths.max()) if lengths.size else 0

    X = np.full((len(batch), T), int(pad_id), dtype=np.int64)
    for i, ids in enumerate(ids_list):
        L = int(ids.shape[0])
        if L:
            X[i, :L] = ids

    attention_mask = np.zeros((len(batch), T), dtype=np.int64)
    for i, L in enumerate(lengths.tolist()):
        if L:
            attention_mask[i, :int(L)] = 1

    out = {"input_ids": X, "lengths": lengths, "attention_mask": attention_mask}

    # Optional labels
    if label_key is not None and label_key in batch[0]:
        labs = [b.get(label_key, None) for b in batch]
        if any(l is None for l in labs):
            out["labels"] = labs
        else:
            l0 = labs[0]
            if isinstance(l0, Tensor):
                out["labels"] = labs  # safe: keep as list
            else:
                l0a = np.asarray(l0)
                if l0a.ndim == 0:
                    out["labels"] = np.asarray([int(l) for l in labs], dtype=np.int64)
                else:
                    # one-hot or multi-target
                    try:
                        out["labels"] = np.stack([np.asarray(l) for l in labs], axis=0)
                    except Exception:
                        out["labels"] = [np.asarray(l) for l in labs]

    return out


def collate_clip(batch, image_key="image"):
    """
    CPU-first CLIP-style collate.
    Returns:
      {"images": (B,C,H,W) np or list, "text": list[str]}  OR pos/neg lists.
    """
    if len(batch) == 0:
        return {"images": [], "text": []}

    first_img = batch[0][image_key]
    if isinstance(first_img, Tensor):
        images = [b[image_key] for b in batch]
    else:
        imgs = [np.asarray(b[image_key]) for b in batch]
        try:
            images = np.stack(imgs, axis=0)
        except Exception:
            images = imgs

    out = {"images": images}

    if "text_pos" in batch[0]:
        out["text_pos"] = [b["text_pos"] for b in batch]
        out["text_neg"] = [b["text_neg"] for b in batch]
    else:
        out["text"] = [b["text"] for b in batch]

    return out


def collate_pad_fields(batch, pad_id=0, keys=("input_ids",), return_mask=True):
    """
    CPU-first: pads multiple 1D token fields to max length in batch.
    Returns dict with padded fields, lengths, and optional attention_mask.

    Notes:
      - If a particular key is missing or None in any sample, that field is returned as a list.
      - If the first sample's field is a Tensor, that field is returned as a list (safe).
      - All padded fields are np.int64.
    """
    out = {}
    lengths_ref = None
    maxT_ref = None

    for key in keys:
        # gather sequences
        seqs = []
        missing = False

        for b in batch:
            v = b.get(key, None)
            if v is None:
                missing = True
                break
            if isinstance(v, Tensor):
                # safest: don't try to pad/stack tensors in CPU collate
                missing = True
                break
            seqs.append(np.asarray(v, dtype=np.int64).reshape(-1))

        if missing:
            out[key] = [b.get(key, None) for b in batch]
            continue

        lengths = np.asarray([int(s.shape[0]) for s in seqs], dtype=np.int64)
        T = int(lengths.max()) if lengths.size else 0

        X = np.full((len(batch), T), int(pad_id), dtype=np.int64)
        for i, s in enumerate(seqs):
            L = int(s.shape[0])
            if L:
                X[i, :L] = s

        out[key] = X

        # Use the first successfully padded key as the reference for lengths/mask
        if lengths_ref is None:
            lengths_ref = lengths
            maxT_ref = T
            if return_mask:
                mask = np.zeros((len(batch), T), dtype=np.int64)
                for i, L in enumerate(lengths.tolist()):
                    if L:
                        mask[i, :int(L)] = 1
                out["attention_mask"] = mask

    if lengths_ref is not None:
        out["lengths"] = lengths_ref

    # keep other non-token fields as lists (strings, metadata, etc.)
    # use union of keys to be robust
    keys_all = set()
    for b in batch:
        if isinstance(b, dict):
            keys_all.update(b.keys())

    for k in keys_all:
        if k in keys or k in ("attention_mask", "lengths"):
            continue
        out[k] = [b.get(k, None) if isinstance(b, dict) else None for b in batch]

    return out


def collate_lm_next_token(batch, ids_key="input_ids", labels_key="labels"):
    """
    CPU-first LM next-token collate.
    Expects each sample:
      {ids_key: (T,), labels_key: (T,)} as list/np arrays.
    Returns:
      {"input_ids": (B,T) np.int64, "labels": (B,T) np.int64}
    If tensors appear, returns lists (safe).
    """
    if len(batch) == 0:
        return {"input_ids": np.zeros((0, 0), dtype=np.int64),
                "labels": np.zeros((0, 0), dtype=np.int64)}

    x0 = batch[0].get(ids_key)
    y0 = batch[0].get(labels_key)

    if isinstance(x0, Tensor) or isinstance(y0, Tensor):
        return {
            "input_ids": [b.get(ids_key, None) for b in batch],
            "labels": [b.get(labels_key, None) for b in batch],
        }

    xs = [np.asarray(b[ids_key], dtype=np.int64).reshape(-1) for b in batch]
    ys = [np.asarray(b[labels_key], dtype=np.int64).reshape(-1) for b in batch]

    try:
        X = np.stack(xs, axis=0)
    except Exception:
        X = xs
    try:
        Y = np.stack(ys, axis=0)
    except Exception:
        Y = ys

    return {"input_ids": X, "labels": Y}


def collate_seq2seq_pad(batch, pad_id=0, src_key="src", tgt_key="tgt"):
    """
    CPU-first seq2seq padding collate.

    Expects each sample dict:
      {src_key: 1D ids, tgt_key: 1D ids}

    Returns:
      {
        "src_ids": (B,Ts) np.int64,
        "src_lengths": (B,) np.int64,
        "tgt_ids": (B,Tt) np.int64,
        "tgt_lengths": (B,) np.int64,
        ...extra fields as lists
      }

    If tensors appear in src/tgt, returns lists for those fields (safe).
    """
    if len(batch) == 0:
        return {"src_ids": np.zeros((0, 0), dtype=np.int64),
                "src_lengths": np.zeros((0,), dtype=np.int64),
                "tgt_ids": np.zeros((0, 0), dtype=np.int64),
                "tgt_lengths": np.zeros((0,), dtype=np.int64)}

    # Detect tensor inputs
    if isinstance(batch[0].get(src_key), Tensor) or isinstance(batch[0].get(tgt_key), Tensor):
        out = {
            "src_ids": [b.get(src_key, None) for b in batch],
            "tgt_ids": [b.get(tgt_key, None) for b in batch],
        }
        # keep extra fields
        keys_all = set()
        for b in batch:
            keys_all.update(b.keys())
        for k in keys_all:
            if k in (src_key, tgt_key):
                continue
            out[k] = [b.get(k, None) for b in batch]
        return out

    srcs = [np.asarray(b[src_key], dtype=np.int64).reshape(-1) for b in batch]
    tgts = [np.asarray(b[tgt_key], dtype=np.int64).reshape(-1) for b in batch]

    src_len = np.asarray([int(s.shape[0]) for s in srcs], dtype=np.int64)
    tgt_len = np.asarray([int(t.shape[0]) for t in tgts], dtype=np.int64)

    Ts = int(src_len.max()) if src_len.size else 0
    Tt = int(tgt_len.max()) if tgt_len.size else 0

    src_ids = np.full((len(batch), Ts), int(pad_id), dtype=np.int64)
    tgt_ids = np.full((len(batch), Tt), int(pad_id), dtype=np.int64)

    for i, s in enumerate(srcs):
        L = int(s.shape[0])
        if L:
            src_ids[i, :L] = s
    for i, t in enumerate(tgts):
        L = int(t.shape[0])
        if L:
            tgt_ids[i, :L] = t

    out = {
        "src_ids": src_ids,
        "src_lengths": src_len,
        "tgt_ids": tgt_ids,
        "tgt_lengths": tgt_len,
    }

    # keep extra fields
    keys_all = set()
    for b in batch:
        keys_all.update(b.keys())
    for k in keys_all:
        if k in (src_key, tgt_key):
            continue
        out[k] = [b.get(k, None) for b in batch]

    return out


def collate_video(batch, key="video"):
    """
    CPU-first video collate.
    Accepts either:
      - samples as np arrays (T,C,H,W) or list-like
      - samples as dicts with video at `key`

    Returns:
      - if dict samples: {"video": (B,T,C,H,W) np or list, ...other fields collated with _collate}
      - else: (B,T,C,H,W) np or list
    """
    if len(batch) == 0:
        return {"video": []} if isinstance(batch, list) else []

    if isinstance(batch[0], dict):
        v0 = batch[0].get(key, None)

        if isinstance(v0, Tensor):
            vids = [b.get(key, None) for b in batch]
            V = vids
        else:
            vids = [np.asarray(b[key]) for b in batch]
            try:
                V = np.stack(vids, axis=0)
            except Exception:
                V = vids

        out = {"video": V}

        keys_all = set()
        for b in batch:
            keys_all.update(b.keys())

        for k in keys_all:
            if k == key:
                continue
            out[k] = _collate([b.get(k, None) for b in batch])

        return out

    # non-dict samples
    if isinstance(batch[0], Tensor):
        return list(batch)

    vids = [np.asarray(v) for v in batch]
    try:
        return np.stack(vids, axis=0)
    except Exception:
        return vids


def collate_tracking(batch):
    """
    CPU-first tracking collate.

    Expects samples dict with:
      "video": (T,C,H,W)
      "boxes": (T,K,4)  (fixed K)
      "track_ids": (K,) or list length K
      optional "visible": (T,K)

    Returns:
      {
        "video": (B,T,C,H,W) np or list,
        "boxes": (B,T,K,4) np or list,
        "visible": (B,T,K) np or list (if present),
        "track_ids": from first sample,
        ... extra fields collated
      }

    If tensors appear in video/boxes/visible, returns lists for those fields (safe).
    """
    if len(batch) == 0:
        return {"video": [], "boxes": [], "track_ids": None}

    v0 = batch[0].get("video")
    b0 = batch[0].get("boxes")

    tensor_mode = isinstance(v0, Tensor) or isinstance(b0, Tensor)

    if tensor_mode:
        V = [b.get("video", None) for b in batch]
        B = [b.get("boxes", None) for b in batch]
    else:
        vids = [np.asarray(b["video"]) for b in batch]
        boxes = [np.asarray(b["boxes"]) for b in batch]
        try:
            V = np.stack(vids, axis=0)
        except Exception:
            V = vids
        try:
            B = np.stack(boxes, axis=0)
        except Exception:
            B = boxes

    out = {"video": V, "boxes": B}

    if "visible" in batch[0]:
        vis0 = batch[0].get("visible")
        if isinstance(vis0, Tensor):
            out["visible"] = [b.get("visible", None) for b in batch]
        else:
            vis = [np.asarray(b["visible"]) for b in batch]
            try:
                out["visible"] = np.stack(vis, axis=0)
            except Exception:
                out["visible"] = vis

    # Usually same for all samples; keep from first
    out["track_ids"] = batch[0].get("track_ids", None)

    # keep any extra metadata
    keys_all = set()
    for b in batch:
        keys_all.update(b.keys())

    for k in keys_all:
        if k in ("video", "boxes", "visible", "track_ids"):
            continue
        out[k] = _collate([b.get(k, None) for b in batch])

    return out