import LunarLearn.core.backend.backend as backend
from LunarLearn.nn.regularizers import BaseRegularizer
from LunarLearn.core import Tensor, ops

xp = backend.xp
DTYPE = backend.DTYPE

class HessianTrace(BaseRegularizer):
    """
    Hutchinson-based Hessian trace regularizer.

    Approximates Tr(H) of the loss wrt parameters using a single (or multiple)
    random probes. Requires higher-order gradients (i.e., backprop through
    gradients). If unavailable, raises a RuntimeError.

    Args:
        lam (float): Regularization strength.
        epsilon (float): Finite-difference step for symmetric gradient norm probe.
        probes (int): Number of Hutchinson samples to average.
        with_layer (bool): If True, uses model.parameters(with_layer=True) so
                           per-layer LR/opts still work in the main optimizer.
    """
    def __init__(self, lam: float = 1e-4, epsilon: float = 1e-3, probes: int = 1, with_layer: bool = True):
        self.lam = lam
        self.epsilon = epsilon
        self.probes = probes
        self.with_layer = with_layer

    def _sample_probe_like_params(self, params):
        """Create a list of Rademacher/Gaussian probes with same shapes as params."""
        vs = []
        for p in params:
            # p can be a Tensor or a {'param': Tensor, 'layer': ...}
            t = p["param"] if isinstance(p, dict) else p
            if not getattr(t, "requires_grad", False):
                vs.append(None)
                continue
            # Rademacher ±1 works well; Gaussian is also fine.
            rnd = xp.random.choice(xp.array([-1.0, 1.0], dtype=t.dtype), size=t.data.shape)
            v = Tensor(rnd, requires_grad=False, dtype=t.dtype)
            vs.append(v)
        return vs

    def _shift_params(self, params, vs, sign):
        """θ ← θ + sign * ε * v (in-place)."""
        for p, v in zip(params, vs):
            t = p["param"] if isinstance(p, dict) else p
            if v is None: 
                continue
            t.data += sign * self.epsilon * v.data

    def _grad_norm_sq(self, model, loss_fn, X, y):
        """
        Compute ||∇_θ L||^2 (second-order graph must be retained).
        Assumes backend supports higher-order grads.
        """
        # Forward + scalar loss
        preds = model(X)
        loss = loss_fn(preds, y)
        # Backprop to parameters, retaining graph for higher-order
        # Your autograd API likely: loss.backward(create_graph=True) — if not,
        # replace with your equivalent flag.
        try:
            loss.backward(create_graph=True)
        except TypeError:
            # Fall back to a backend flag if that's your API
            if hasattr(backend, "BACKWARD_CREATE_GRAPH"):
                old = backend.BACKWARD_CREATE_GRAPH
                backend.BACKWARD_CREATE_GRAPH = True
                loss.backward()
                backend.BACKWARD_CREATE_GRAPH = old
            else:
                raise RuntimeError(
                    "Higher-order gradients are required by HessianTraceRegularizer "
                    "but your backend doesn't support create_graph=True."
                )

        # Accumulate ||grad||^2 and clear grads for the next call
        total = None
        for p in model.parameters(with_layer=self.with_layer):
            t = p["param"] if isinstance(p, dict) else p
            g = t.grad
            if g is None:
                continue
            gn = ops.sum(g * g)
            total = gn if total is None else (total + gn)

        # IMPORTANT: do NOT zero grads here — the outer training loop will.
        return total if total is not None else Tensor(0.0)

    def __call__(self, model, loss_fn=None, X=None, y=None):
        """
        Estimate Tr(H) via Hutchinson probes and return lam * Tr(H).

        You must pass (loss_fn, X, y) so we can recompute the loss at shifted
        parameters. This is different from simple param-only regularizers.

        Returns:
            Tensor: scalar regularization penalty added to main loss.
        """
        if loss_fn is None or X is None or y is None:
            raise ValueError(
                "HessianTraceRegularizer requires (loss_fn, X, y). "
                "Call as: reg(model, loss_fn, X, y)."
            )

        params = model.parameters(with_layer=self.with_layer)
        acc = None

        for _ in range(self.probes):
            # Sample a probe v for each parameter
            vs = self._sample_probe_like_params(params)

            # θ+ = θ + ε v
            self._shift_params(params, vs, +1.0)
            gpos = self._grad_norm_sq(model, loss_fn, X, y)

            # Undo shift to base θ
            self._shift_params(params, vs, -1.0)

            # θ- = θ - ε v
            self._shift_params(params, vs, -1.0)
            gneg = self._grad_norm_sq(model, loss_fn, X, y)

            # Restore base θ
            self._shift_params(params, vs, +1.0)

            # Hutchinson symmetric difference approximation: (||g+||^2 - ||g-||^2)/(2ε)
            est = (gpos - gneg) * (0.5 / self.epsilon)
            acc = est if acc is None else (acc + est)

        if self.probes > 1:
            acc = acc * (1.0 / float(self.probes))

        return acc * self.la