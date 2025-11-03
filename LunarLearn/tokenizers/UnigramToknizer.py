import regex as re
from collections import defaultdict

class UnigramTokenizer:
    def __init__(self, vocab_size=32000, max_vocab_mul=10, max_len=10, regex_pattern=None, lowercase=True, add_special_tokens=True, freq_threshold=None):
        self.vocab_size = vocab_size
        self.max_vocab = vocab_size * max_vocab_mul
        self.max_len = max_len
        self.regex = re.compile(regex_pattern) if regex_pattern is not None else None
        self.lowercase = lowercase
        self.add_special_tokens = add_special_tokens
        self.freq_threshold = freq_threshold

        self.id2sym = []
        self.sys2id = {}
        self.prob = {}

    def _split_to_sentence(self, corpus):
        sentences = re.split(r"[.!?\n]+", corpus)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _normalize_text(self, text: str):
        if self.lowercase:
            text = text.lower()
        if self.regex is not None:
            return self.regex.findall(text)
        return list(text)
    
    def _init_alphabet(self, corpus):
        counts = defaultdict(int)
        for 
        for i in range(len(tokens)):
            for j in range(i+1, min(i+self.max_len, len(tokens)) + 1):
                sub = "".join(tokens[i:j])
                counts[sub] += 1
        sorted_counts = {k: v for k, v in reversed(sorted(counts.items(), key=lambda item: item[1])) if v >= self.freq_threshold}
        for _ in range(len(sorted_counts-self.max_vocab)):
            sorted_counts.popitem()
        
        self.id2sym = list(sorted_counts.keys())
        self.sys2id = {sym: i for i, sym in enumerate(self.id2sym)}
        self.prob = {sym: (1/len(self.id2sym)) for sym in self.id2sym}

    def train(self, corpus):
        pass

    def encode(self):
        pass

    def decode(self):
        pass