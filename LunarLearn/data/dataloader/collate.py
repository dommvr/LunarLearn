import LunarLearn.core.backend.backend as backend
from LunarLearn.data.dataloader.utils import _is_xp_array, _try_stack_any
from LunarLearn.core import Tensor

xp = backend.xp


def _collate(batch):
    """
    Collate a list of samples into a batched structure.
    Supports:
      - (x,y) tuples or any tuple/list
      - dict samples (detection, QA, CLIP, etc.)
      - xp arrays (stack if possible)
      - scalars
      - strings (keep list)
    """
    if len(batch) == 0:
        return batch

    # If any element is None, keep as list (optional fields)
    if any(b is None for b in batch):
        return list(batch)

    first = batch[0]

    # Dict samples
    if isinstance(first, dict):
        out = {}
        for k in first.keys():
            out[k] = _collate([b.get(k, None) for b in batch])
        return out

    # Tuple/list samples (zip fields)
    if isinstance(first, (tuple, list)):
        fields = list(zip(*batch))
        return type(first)(_collate(list(f)) for f in fields)

    # Strings
    if isinstance(first, str):
        return list(batch)

    # Tensors: keep as-is (avoid breaking)
    if isinstance(first, Tensor) or _is_xp_array(first):
        return _try_stack_any(batch)

    # Numeric scalars -> vector
    if isinstance(first, (int, float, bool)):
        return xp.asarray(batch)

    # Fallback: keep list
    return list(batch)


def collate_xy_stack(batch):
    xs, ys = zip(*batch)

    # allow Tensor inputs
    if isinstance(xs[0], Tensor):
        xs_data = [x.data for x in xs]
        X = Tensor(xp.stack(xs_data, axis=0), requires_grad=False)
    else:
        X = xp.stack([xp.asarray(x) for x in xs], axis=0)

    # y might be scalar or vector
    if isinstance(ys[0], Tensor):
        ys_data = [y.data for y in ys]
        try:
            Y = Tensor(xp.stack(ys_data, axis=0), requires_grad=False)
        except Exception:
            Y = list(ys)
    else:
        ys_arr = [xp.asarray(y) for y in ys]
        try:
            Y = xp.stack(ys_arr, axis=0)
        except Exception:
            # if truly ragged, keep list
            Y = ys_arr

    return X, Y


def collate_detection(batch):
    # accept either "image" or "images"
    key_img = "image" if "image" in batch[0] else "images"
    images = xp.stack([b[key_img] if not isinstance(b[key_img], Tensor) else b[key_img].data for b in batch], 0)
    images = Tensor(images, requires_grad=False) if isinstance(batch[0][key_img], Tensor) else images

    targets = []
    for b in batch:
        t = {"boxes": b["boxes"], "labels": b["labels"]}
        # keep optional extra fields per target if present
        for k in b.keys():
            if k not in (key_img, "boxes", "labels"):
                t[k] = b[k]
        targets.append(t)

    return {"images": images, "targets": targets}


def collate_segmentation(batch):
    imgs = [b["image"].data if isinstance(b["image"], Tensor) else b["image"] for b in batch]
    msks = [b["mask"].data if isinstance(b["mask"], Tensor) else b["mask"] for b in batch]

    images = xp.stack(imgs, 0)
    masks = xp.stack(msks, 0)

    # preserve tensor-ness if inputs were tensors
    if isinstance(batch[0]["image"], Tensor):
        images = Tensor(images, requires_grad=False)
    if isinstance(batch[0]["mask"], Tensor):
        masks = Tensor(masks, requires_grad=False)

    return {"images": images, "masks": masks}


def collate_pad_tokens(batch, pad_id=0, label_key="label", ids_key="input_ids"):
    # input_ids can be list or xp array or Tensor
    ids_list = []
    for b in batch:
        ids = b[ids_key]
        if isinstance(ids, Tensor):
            ids = ids.data
        ids_list.append(xp.asarray(ids, dtype=xp.int64))

    lengths = xp.asarray([int(x.shape[0]) for x in ids_list], dtype=xp.int64)
    T = int(lengths.max())
    X = xp.full((len(batch), T), int(pad_id), dtype=xp.int64)

    for i, ids in enumerate(ids_list):
        X[i, :ids.shape[0]] = ids

    out = {"input_ids": X, "lengths": lengths}

    # optional attention mask
    mask = xp.zeros((len(batch), T), dtype=xp.int64)
    for i, L in enumerate(lengths.tolist()):
        mask[i, :int(L)] = 1
    out["attention_mask"] = mask

    # optional labels
    if label_key is not None and label_key in batch[0]:
        labs = [b.get(label_key, None) for b in batch]
        if any(l is None for l in labs):
            out["labels"] = labs
        else:
            # if labels are arrays (one-hot), stack; if scalars, vectorize
            l0 = labs[0]
            if isinstance(l0, Tensor):
                labs_data = [l.data for l in labs]
                try:
                    out["labels"] = Tensor(xp.stack(labs_data, 0), requires_grad=False)
                except Exception:
                    out["labels"] = labs
            else:
                l0a = xp.asarray(l0)
                if l0a.ndim == 0:
                    out["labels"] = xp.asarray([int(l) for l in labs], dtype=xp.int64)
                else:
                    out["labels"] = xp.stack([xp.asarray(l) for l in labs], 0)

    return out


def collate_clip(batch):
    imgs = [b["image"].data if isinstance(b["image"], Tensor) else b["image"] for b in batch]
    images = xp.stack(imgs, 0)
    if isinstance(batch[0]["image"], Tensor):
        images = Tensor(images, requires_grad=False)

    out = {"images": images}

    if "text_pos" in batch[0]:
        out["text_pos"] = [b["text_pos"] for b in batch]
        out["text_neg"] = [b["text_neg"] for b in batch]
    else:
        out["text"] = [b["text"] for b in batch]

    return out


def collate_pad_fields(batch, pad_id=0, keys=("input_ids",), return_mask=True):
    """
    Pads multiple 1D token fields to max length in batch.
    Returns dict with padded fields, lengths, and optional attention_mask.

    Example:
      collate_pad_fields(batch, keys=("input_ids","labels"))
    """
    out = {}
    lens = None

    for key in keys:
        seqs = []
        for b in batch:
            v = b.get(key, None)
            if v is None:
                seqs = None
                break
            if isinstance(v, Tensor):
                v = v.data
            seqs.append(xp.asarray(v, dtype=xp.int64))

        if seqs is None:
            out[key] = [b.get(key, None) for b in batch]
            continue

        lengths = xp.asarray([int(s.shape[0]) for s in seqs], dtype=xp.int64)
        T = int(lengths.max())
        X = xp.full((len(batch), T), int(pad_id), dtype=xp.int64)
        for i, s in enumerate(seqs):
            X[i, :s.shape[0]] = s
        out[key] = X

        if lens is None:
            lens = lengths
            if return_mask:
                mask = xp.zeros((len(batch), T), dtype=xp.int64)
                for i, L in enumerate(lengths.tolist()):
                    mask[i, :int(L)] = 1
                out["attention_mask"] = mask

    if lens is not None:
        out["lengths"] = lens

    # keep other non-token fields as lists (strings, metadata)
    for k in batch[0].keys():
        if k in keys or k in ("attention_mask", "lengths"):
            continue
        out[k] = [b.get(k, None) for b in batch]

    return out


def collate_lm_next_token(batch):
    """
    Expects each sample to have xp arrays:
      {"input_ids": (T,), "labels": (T,)}  or Tensors with .data
    Returns stacked:
      input_ids: (N,T)
      labels: (N,T)
    """
    xs = [b["input_ids"].data if isinstance(b["input_ids"], Tensor) else b["input_ids"] for b in batch]
    ys = [b["labels"].data if isinstance(b["labels"], Tensor) else b["labels"] for b in batch]
    X = xp.stack(xs, 0).astype(xp.int64)
    Y = xp.stack(ys, 0).astype(xp.int64)
    return {"input_ids": X, "labels": Y}


def collate_seq2seq_pad(batch, pad_id=0, src_key="src", tgt_key="tgt"):
    """
    Expects each sample:
      {src_key: 1D ids, tgt_key: 1D ids}
    Returns:
      src_ids: (N, Ts), src_lengths
      tgt_ids: (N, Tt), tgt_lengths
      plus carries any extra fields as lists.
    """
    def get_ids(b, key):
        v = b[key]
        if isinstance(v, Tensor):
            v = v.data
        return xp.asarray(v, dtype=xp.int64)

    srcs = [get_ids(b, src_key) for b in batch]
    tgts = [get_ids(b, tgt_key) for b in batch]

    src_len = xp.asarray([int(s.shape[0]) for s in srcs], dtype=xp.int64)
    tgt_len = xp.asarray([int(s.shape[0]) for s in tgts], dtype=xp.int64)

    Ts = int(src_len.max())
    Tt = int(tgt_len.max())

    src_ids = xp.full((len(batch), Ts), int(pad_id), dtype=xp.int64)
    tgt_ids = xp.full((len(batch), Tt), int(pad_id), dtype=xp.int64)

    for i, s in enumerate(srcs):
        src_ids[i, :s.shape[0]] = s
    for i, t in enumerate(tgts):
        tgt_ids[i, :t.shape[0]] = t

    out = {
        "src_ids": src_ids,
        "src_lengths": src_len,
        "tgt_ids": tgt_ids,
        "tgt_lengths": tgt_len,
    }

    # keep extra fields
    for k in batch[0].keys():
        if k in (src_key, tgt_key):
            continue
        out[k] = [b.get(k, None) for b in batch]

    return out


def collate_video(batch, key="video"):
    """
    Accepts either:
      - samples as xp arrays (T,C,H,W)
      - samples as dicts with video at `key`
    Returns:
      {"video": (N,T,C,H,W)} or just xp array depending on input type.
    """
    if isinstance(batch[0], dict):
        vids = [b[key].data if isinstance(b[key], Tensor) else b[key] for b in batch]
        V = xp.stack(vids, 0)
        out = {"video": V}
        for k in batch[0].keys():
            if k == key:
                continue
            out[k] = _collate([b.get(k, None) for b in batch])
        return out

    vids = [v.data if isinstance(v, Tensor) else v for v in batch]
    return xp.stack(vids, 0)


def collate_tracking(batch):
    """
    Expects samples dict with:
      "video": (T,C,H,W)
      "boxes": (T,K,4)  (fixed K)
      "track_ids": (K,) or list length K
      optional "visible": (T,K)

    Returns stacked video/boxes/visible, and a single track_ids (from first).
    """
    vids = [b["video"].data if isinstance(b["video"], Tensor) else b["video"] for b in batch]
    boxes = [b["boxes"].data if isinstance(b["boxes"], Tensor) else b["boxes"] for b in batch]

    V = xp.stack(vids, 0)
    B = xp.stack(boxes, 0)

    out = {"video": V, "boxes": B}

    if "visible" in batch[0]:
        vis = [b["visible"].data if isinstance(b["visible"], Tensor) else b["visible"] for b in batch]
        out["visible"] = xp.stack(vis, 0)

    # Usually same for all samples, so take first
    out["track_ids"] = batch[0]["track_ids"]

    # keep any extra metadata
    for k in batch[0].keys():
        if k in ("video", "boxes", "visible", "track_ids"):
            continue
        out[k] = _collate([b.get(k, None) for b in batch])

    return out