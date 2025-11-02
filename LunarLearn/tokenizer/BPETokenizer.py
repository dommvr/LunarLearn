import json
import regex as re

class BPETokenizer:
    def __init__(self, type="bytes", vocab_size=32000, regex_pattern=None, lowercase=True):
        assert type in ["bytes", "characters"]
        self.type = type
        self.vocab_size = vocab_size
        self.regex_pattern = re.compile(regex_pattern) if regex_pattern is not None else None
        self.lowercase = lowercase

        self.merges = None
        self.vocab = None

    def _normalize(self, text: str):
        if self.lowercase:
            text = text.lower()
        if self.regex_pattern is not None:
            return re.findall(self.regex_pattern, text)
        return list(text)

    def _to_tokens(self, text: str):
        if self.type == "bytes":
            tokens = text.encode("utf-8")
            tokens = list(map(int, tokens))
        else:
            tokens = list(tokens)
        return tokens

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts
    
    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def _get_vocab(self):
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        return vocab

    def train(self, corpus):
        corpus = self._normalize(corpus)
        tokens = self._to_tokens(corpus)
        base_vocab = 256 if self.type == "bytes" else len(set(corpus))
        num_merges = self.vocab_size - base_vocab
        ids = list(tokens)

        merges = {}
        for i in range(num_merges):
            stats = self._get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = base_vocab + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx

        self.merges = merges
        self.vocab = self._get_vocab()
        return merges

    def encode(self, text: str):
        tokens = self._to_tokens(text)
        while len(tokens) >= 2:
            stats = self._get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break # nothing else can be merged
            idx = self.merges[pair]
            tokens = self._merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        if self.type == "bytes":
            tokens = b"".join(self.vocab[idx] for idx in ids)
            text = tokens.decode("utf-8", errors="ignore")
        else:
            text = "".join(ids)
        return text

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.merges, f, indent=2)

    def save_vocab(self, path: str):
        with open(path, "w") as f:
            json.dump(self.vocab, f, indent=2)

    def load(self, path: str):
        with open(path, "r") as f:
            self.merges = json.loads(f)
        self.merges = {tuple(map(int, k.strip('()').split(', '))): v for k, v in self.merges.items()}
        self.vocab = self._get_vocab()