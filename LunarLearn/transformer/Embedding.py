import LunarLearn.backend as backend
from LunarLearn.layers.BaseLayer import BaseLayer
from LunarLearn.tensor import Tensor
from LunarLearn.tensor import Parameter

xp = backend.xp

class Embedding(BaseLayer):
    def __init__(self, vocab_size, emb_dim, padding_idx, keep_prob=1.0):
        super().__init__(trainable=True)
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.padding_idx = padding_idx
        self.keep_prob = keep_prob

    def initialize(self, input_shape):
        scale = 1 / xp.sqrt(self.emb_dim)
        W = xp.random.randn(self.vocab_size, self.emb_dim) * scale
        if self.padding_idx is not None:
            W[self.padding_idx] = 0
        self.W = Parameter(W, requires_grad=True)
        if self.padding_idx is not None:
            self.W.register_grad_hook(lambda g: g.at[self.padding_idx].set(0))
        self.output_shape = (input_shape[1], self.emb_dim)

    def forward(self, x: Tensor) -> Tensor:
        from LunarLearn.regularizers import dropout
        if self.W is None:
            self.initialize(x.shape[1:])
        W = self.W.to_compute()
        out = W[x]
        out = dropout(out, self.keep_prob, training=self.training)

        return out

