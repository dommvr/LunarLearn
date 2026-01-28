import math
import heapq
import json
from collections import defaultdict
from .utlis import compile_pattern


class UnigramTokenizer:
    def __init__(self,
                 vocab_size=32000,
                 max_vocab_mul=10,
                 max_len=10,
                 regex_pattern=None,
                 lowercase=True,
                 add_special_tokens=True,
                 freq_threshold=None,
                 prune_ratio=0.1):
        self.vocab_size = vocab_size
        self.max_vocab = vocab_size * max_vocab_mul
        self.max_len = max_len
        self.regex = compile_pattern(regex_pattern) if regex_pattern is not None else None
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        self.freq_threshold = freq_threshold
        self.prune_ratio = prune_ratio

        self.id2sym = []
        self.sym2id = {}
        self.probs = {}

    def _normalize_text(self, text: str):
        # 1. Lowercase normalization
        if self.lowercase:
            text = text.lower()

        # 2. Apply regex pattern if provided (token-level segmentation)
        if self.regex is not None:
            tokens = self.regex.findall(text)
            text = " ".join(tokens)  # rebuild text with spaces preserved
        else:
            # Default: keep spaces for visible marker encoding
            text = re.sub(r"\s+", " ", text.strip())

        # 3. Replace spaces with visible marker for reversible decoding
        return "▁" + text.replace(" ", "▁")
    
    def _prepare_corpus(self, corpus):
        if isinstance(corpus, str):
            sentences = re.split(r"[.!?\n]+", corpus)
            sentences = [s.strip() for s in sentences if s.strip()]
        else:
            sentences = corpus
        return [self._normalize_text(s) for s in sentences]
    
    def _init_vocab(self, corpus):
        counts = defaultdict(int)
        for sentence in corpus:
            for i in range(len(sentence)):
                for j in range(i + 1, min(i + self.max_len, len(sentence)) + 1):
                    sub = sentence[i:j]
                    counts[sub] += 1

        if self.freq_threshold is not None:
            counts = {k: v for k, v in counts.items() if v >= self.freq_threshold}

        sorted_counts = dict(sorted(counts.items(), key=lambda x: -x[1])[:self.max_vocab])

        if len(sorted_counts) == 0:
            raise ValueError("Empty vocabulary after frequency filtering")
        
        alphabet = {ch for s in corpus for ch in set(s)}
        for ch in alphabet:
            if ch not in sorted_counts:
                sorted_counts[ch] = 1  # tiny count floor
        
        self.id2sym = list(sorted_counts.keys())
        self.sym2id = {sym: i for i, sym in enumerate(self.id2sym)}
        self.probs = {sym: (1.0 / len(self.id2sym)) for sym in self.id2sym}

        if self.add_special_tokens:
            for tok in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                if tok not in self.sym2id:
                    self.sym2id[tok] = len(self.id2sym)
                    self.id2sym.append(tok)
                    self.probs[tok] = 1e-8  # tiny mass so they survive until pruning
            # renormalize
            Z = sum(self.probs.values())
            self.probs = {w: p / Z for w, p in self.probs.items()}

    def _logsumexp_list(self, arr):
        if not arr:
            return -float("inf")
        m = max(arr)
        return m + math.log(sum(math.exp(a - m) for a in arr))

    def forward_logprob(self, sentence):
        N = len(sentence)
        logp = [-float("inf")] * (N + 1)
        logp[0] = 0.0
        for i in range(1, N + 1):
            cand = []
            j0 = max(0, i - self.max_len)
            for j in range(j0, i):
                sub = sentence[j:i]
                if sub in self.probs:
                    cand.append(logp[j] + math.log(self.probs[sub]))
            if cand:
                logp[i] = self._logsumexp_list(cand)
        return logp[-1]

    def forward_backward(self, sentence):
        N = len(sentence)
        forward = [0.0] * (N + 1)
        backward = [0.0] * (N + 1)
        forward[0] = 1.0
        backward[N] = 1.0

        # Forward
        for i in range(1, N + 1):
            for j in range(max(0, i - self.max_len), i):
                sub = sentence[j:i]
                if sub in self.probs:
                    forward[i] += forward[j] * self.probs[sub]

        # Backward
        for i in range(N - 1, -1, -1):
            for j in range(i + 1, min(N, i + self.max_len) + 1):
                sub = sentence[i:j]
                if sub in self.probs:
                    backward[i] += self.probs[sub] * backward[j]

        sent_prob = forward[N]
        expected_counts = defaultdict(float)
        for i in range(N):
            for j in range(i + 1, min(N, i + self.max_len) + 1):
                sub = sentence[i:j]
                if sub in self.probs:
                    contrib = (forward[i] * self.probs[sub] * backward[j]) / (sent_prob + 1e-12)
                    expected_counts[sub] += contrib

        return expected_counts, math.log(sent_prob + 1e-12)
    
    def m_step(self, total_counts, eps=1e-12):
        V = len(total_counts)
        Z = sum(total_counts.values()) + eps * V
        new_probs = {w: (c + eps) / Z for w, c in total_counts.items()}

        # Keep special tokens sticky with tiny mass
        if self.add_special_tokens:
            for tok in ("<pad>", "<unk>", "<bos>", "<eos>"):
                if tok not in new_probs:
                    new_probs[tok] = eps  # tiny floor

            # renormalize after injecting specials
            Z2 = sum(new_probs.values())
            new_probs = {w: p / Z2 for w, p in new_probs.items()}

        self.probs = new_probs

    def prune(self):
        # don't drop below target vocab size
        cur = len(self.probs)
        can_drop = max(0, cur - self.vocab_size)
        num_to_remove = min(int(self.prune_ratio * cur), can_drop)
        if num_to_remove <= 0:
            return
        
        must_keep = set()
        if self.add_special_tokens:
            must_keep |= {"<pad>", "<unk>", "<bos>", "<eos>"}
        # keep all length-1 symbols to guarantee segmentability
        must_keep |= {s for s in self.probs if len(s) == 1}

        heap = [(p, s) for s, p in self.probs.items() if s not in must_keep]
        heapq.heapify(heap)

        # remove lowest-probability tokens
        for _ in range(num_to_remove):
            if not heap:
                break
            _, sym = heapq.heappop(heap)
            self.probs.pop(sym, None)

        # Rebuild symbol tables
        self.id2sym = list(self.probs.keys())
        self.sym2id = {sym: i for i, sym in enumerate(self.id2sym)}

        # renormalize
        Z = sum(self.probs.values())
        if Z > 0:
            self.probs = {w: p / Z for w, p in self.probs.items()}

    def train(self, corpus, max_iters=5):
        corpus = self._prepare_corpus(corpus)
        self._init_vocab(corpus)

        prev_log_likelihood = -float("inf")

        for it in range(max_iters):
            total_counts = defaultdict(float)
            total_log_likelihood = 0.0

            for sentence in corpus:
                counts, logp = self.forward_backward(sentence)
                total_log_likelihood += logp
                for k, v in counts.items():
                    total_counts[k] += v

            self.m_step(total_counts)

            self.prune()

            if len(self.probs) > self.vocab_size:
                old = self.prune_ratio
                self.prune_ratio = 1.0
                self.prune()
                self.prune_ratio = old

            print(f"[iter {it+1}] log-likelihood: {total_log_likelihood:.4f}, vocab: {len(self.probs)}")

            # Convergence check
            if abs(total_log_likelihood - prev_log_likelihood) < 1e-5:
                break
            prev_log_likelihood = total_log_likelihood

    def encode(self, sentence, add_bos=False, add_eos=False):
        """Segment a sentence into most probable sequence of tokens using Viterbi."""
        sentence = self._normalize_text(sentence)
        N = len(sentence)
        best_logprob = [-float("inf")] * (N + 1)
        best_logprob[0] = 0.0
        best_seg = [-1] * (N + 1)

        # DP forward pass
        for i in range(1, N + 1):
            for j in range(max(0, i - self.max_len), i):
                sub = sentence[j:i]
                if sub in self.probs:
                    logp = best_logprob[j] + math.log(self.probs[sub])
                    if logp > best_logprob[i]:
                        best_logprob[i] = logp
                        best_seg[i] = j

        # Backtrack for best segmentation
        tokens = []
        i = N
        while i > 0:
            j = best_seg[i]
            if j == -1:
                # unknown token fallback — use single char
                j = i - 1
                sub = sentence[j:i]
            else:
                sub = sentence[j:i]
            tokens.append(sub)
            i = j
        tokens.reverse()

        # Convert to ids
        unk_id = self.sym2id.get("<unk>")
        ids = [self.sym2id.get(tok, unk_id if unk_id is not None else 0) for tok in tokens]
        if add_bos and "<bos>" in self.sym2id:
            ids = [self.sym2id["<bos>"]] + ids
        if add_eos and "<eos>" in self.sym2id:
            ids = ids + [self.sym2id["<eos>"]]
        return ids
    
    def encode_batch(self, texts, add_bos=False, add_eos=False):
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode(self, ids, strip_specials=True):
        tokens = [self.id2sym[i] if i < len(self.id2sym) else "<unk>" for i in ids]
        if strip_specials and self.add_special_tokens:
            specials = {"<pad>", "<unk>", "<bos>", "<eos>"}
            tokens = [t for t in tokens if t not in specials]
        text = "".join(tokens).replace("▁", " ").strip()
        return text
    
    def decode_batch(self, batch_ids, strip_specials=True):
        return [self.decode(ids, strip_specials=strip_specials) for ids in batch_ids]
    
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "max_len": self.max_len,
                "regex_pattern": self.regex.pattern if self.regex else None,
                "lowercase": self.lowercase,
                "add_special_tokens": self.add_special_tokens,
                "id2sym": self.id2sym,
                "probs": [self.probs[s] for s in self.id2sym],
            }, f, ensure_ascii=False)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

            self.vocab_size = obj["vocab_size"]
            self.max_len = obj["max_len"]
            # restore config
            self.lowercase = obj.get("lowercase", True)
            self.add_special_tokens = obj.get("add_special_tokens", True)
            pattern = obj.get("regex_pattern")
            self.regex = re.compile(pattern) if pattern else None

            self.id2sym = obj["id2sym"]
            self.sym2id = {s: i for i, s in enumerate(self.id2sym)}
            self.probs = {s: p for s, p in zip(self.id2sym, obj["probs"])}