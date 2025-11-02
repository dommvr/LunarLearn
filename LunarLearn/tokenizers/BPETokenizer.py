import json
import regex as re
import heapq
from collections import Counter, OrderedDict, defaultdict

class BPETokenizer:
    def __init__(self, type="bytes", vocab_size=32000, regex_pattern=None, lowercase=True, add_special_tokens=True, freq_threshold=None):
        assert type in ["bytes", "characters"]
        self.type = type
        self.vocab_size = vocab_size
        self.regex = re.compile(regex_pattern) if regex_pattern is not None else None
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        self.special_tokens = OrderedDict([("<pad>", 0), ("<unk>", 1), ("<bos>", 2), ("<eos>", 3)])
        self.freq_threshold = freq_threshold

        self.merges = OrderedDict()
        self.id2sym = {}
        self.sym2id = {}
        self.next_id = 0

    def _normalize_text(self, text: str):
        if self.lowercase:
            text = text.lower()
        if self.regex is not None:
            return self.regex.findall(text)
        return list(text)
    
    def _init_alphabet(self, tokens_list):
        self.id2sym.clear(), self.sym2id.clear()
        if self.type == "bytes":
            for i in range(256):
                b = bytes([i])
                self.id2sym[i] = b
                self.sym2id[b] = i
            self.next_id = 256
        else:
            freq = Counter(tokens_list)
            base = sorted(freq.keys(), key=lambda s: (-freq[s], s))
            for i, s in enumerate(base):
                self.id2sym[i] = s
                self.sym2id[s] = i
            self.next_id = len(base)

    def _tokens_to_ids(self, tokens):
        if self.type == "bytes":
            b = "".join(tokens).encode("utf-8")
            return list(b)
        return [self.sym2id[t] for t in tokens]
    
    def _get_stats(self, ids):
        counts = Counter(zip(ids, ids[1:]))
        return counts
    
    def _merge_once(self, ids, pair, new_id):
        a, b = pair
        out = []
        i = 0
        n = len(ids)
        while i < n:
            if i+1 < n and ids[i] == a and ids[i+2] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out
    
    def _train_naive(self, corpus: str):
        tokens = self._normalize_text(corpus)
        self._init_alphabet(tokens)
        ids = self._tokens_to_ids(tokens)

        target_vocab = self.vocab_size
        while self.next_id < target_vocab:
            stats = self._get_stats(ids)
            if not stats:
                break
            (p0, p1), freq = stats.most_common(1)[0]
            if self.freq_threshold is not None and freq < self.freq_threshold:
                break
            new_id = self.next_id
            ids = self._merge_once(ids, (p0, p1), new_id)
            piece = self.id2sym[p0] + self.id2sym[p1]
            self.id2sym[new_id] = piece
            self.merges[(p0, p1)] = new_id
            self.next_id += 1

    def _train_heapq(self, corpus: str):
        tokens = self._normalize_text(corpus)
        self._init_alphabet(tokens)
        ids = self._tokens_to_ids(tokens)

        # Step 1: Count all adjacent pairs
        pair_freq = defaultdict(int)
        for a, b in zip(ids, ids[1:]):
            pair_freq[(a, b)] += 1

        # Step 2: Build max-heap
        heap = [(-freq, pair) for pair, freq in pair_freq.items()]
        heapq.heapify(heap)

        merges = 0
        while merges < (self.vocab_size - self.next_id):
            if not heap:
                break

            freq, pair = heapq.heappop(heap)
            freq = -freq

            # If pair disappeared (stale heap entry), skip
            if pair not in pair_freq or pair_freq[pair] != freq:
                continue

            # Optional: stop if below threshold
            if self.freq_threshold and freq < self.freq_threshold:
                break

            # Merge
            new_id = self.next_id
            self.next_id += 1
            merges += 1
            self.merges[pair] = new_id

            a, b = pair
            new_ids = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and ids[i] == a and ids[i + 1] == b:
                    new_ids.append(new_id)
                    i += 2
                else:
                    new_ids.append(ids[i])
                    i += 1
            ids = new_ids

            # Update stats incrementally
            # Clear out old frequencies for the merged pair
            del pair_freq[pair]

            # Update affected pairs
            pair_freq.clear()
            for a, b in zip(ids, ids[1:]):
                pair_freq[(a, b)] += 1

            # Rebuild heap from updated freq (can be optimized further)
            heap = [(-freq, pair) for pair, freq in pair_freq.items()]
            heapq.heapify(heap)

        # Build vocab for decoding
        for (p0, p1), idx in self.merges.items():
            self.id2sym[idx] = self.id2sym[p0] + self.id2sym[p1]
    
    def train(self, corpus: str, mode="naive"):
        if mode == "naive":
            self._train_naive(corpus)
        elif mode == "heapq":
            self._train_heapq(corpus)
        else:
            raise ValueError("mode must be 'naive' or 'heapq'")

    def encode(self, text: str, add_bos=False, add_eos=False):
        tokens = self._normalize_text(text)
        ids = self._tokens_to_ids(tokens)

        rank = {pair: idx for pair, idx in self.merges.items()}
        while len(ids) < 1:
            stats = self._get_stats(ids)
            best = None
            best_rank = float("inf")
            for p in stats.keys():
                r = rank.get(p, float("inf"))
                if r < best_rank:
                    best_rank = r
                    best = p
            if best is None or best_rank == float("inf"):
                break
            ids = self._merge_once(ids, best, rank[best])

        if add_bos: ids = [self.special_tokens["<bos>"]] + ids
        if add_eos: ids = ids + [self.special_tokens["<eos>"]]
        return ids

    def encode_batch(self, texts, add_bos=False, add_eos=False):
        return [self.encode(t, add_bos, add_eos) for t in texts]
    
    def decode(self, ids):
        st_inv = {v:k for k,v in self.special_tokens.items()}
        pieces = []
        for i in ids:
            if i in st_inv:
                continue
            piece = self.id2sym.get(i)
            if piece is None:
                continue
            pieces.append(piece)
        
        if self.type == "bytes":
            b = b"".join(p if isinstance(p, (bytes, bytearray)) else str(p).encode("utf-8") for p in pieces)
            return b.decode("utf-8", errors="ignore")
        else:
            return "".join(pieces)
        
    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

    def save(self, path):
        obj = {
            "type": self.type,
            "vocab_size": self.vocab_size,
            "lowercase": self.lowercase,
            "regex": self.regex.pattern if self.regex else None,
            "merges": [[a, b, new] for (a,b), new in self.merges.items()],
            "id2sym": {str(k): (v.decode("latin1") if isinstance(v, (bytes, bytearray)) else v)
                       for k,v in self.id2sym.items()},
            "special_tokens": list(self.special_tokens.items()),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.type = obj["type"]
        self.vocab_size = obj["vocab_size"]
        self.lowercase = obj["lowercase"]
        self.regex = re.compile(obj["regex"]) if obj["regex"] else None
        self.special_tokens = OrderedDict(obj["special_tokens"])
        # restore id2sym
        self.id2sym = {}
        for k, v in obj["id2sym"].items():
            k = int(k)
            if self.type == "bytes":
                self.id2sym[k] = v.encode("latin1")
            else:
                self.id2sym[k] = v
        self.sym2id = {v:k for k,v in self.id2sym.items()}
        self.merges = OrderedDict()
        for a,b,new in obj["merges"]:
            self.merges[(a,b)] = new
        self.next_id = max(self.id2sym.keys()) + 1