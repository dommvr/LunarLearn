import numpy as np
import copy

def gradient_check(model, X, y, epsilon=1e-5, tolerance=1e-7, num_checks=10):
    """
    Gradient checker for LunarLearn models.
    
    Args:
        model: BaseModel() instance with .forward(), .backward() and layers holding W, b, dW, db
        X: input batch
        y: target batch
        epsilon: small step for finite differences
        tolerance: maximum allowed relative error
        num_checks: how many random params to test
    
    Returns:
        True if gradients are correct, False otherwise
    """
    # Forward + backward to compute analytical grads
    loss = model.forward(X, y)
    model.backward()
    
    print(f"Initial loss: {loss:.6f}")
    
    for layer in model.layers:
        # Only check trainable layers
        if not hasattr(layer, "W"):
            continue

        for param_name, param, grad in [
            ("W", layer.W, layer.dW),
            ("b", layer.b, layer.db),
        ]:
            if param is None:
                continue
            
            # Pick random indices to check
            for _ in range(num_checks):
                idx = tuple(np.random.randint(s) for s in param.shape)
                old_val = param[idx]
                
                # Numerical gradient
                param[idx] = old_val + epsilon
                loss_plus = model.forward(X, y)
                
                param[idx] = old_val - epsilon
                loss_minus = model.forward(X, y)
                
                param[idx] = old_val  # restore
                
                g_num = (loss_plus - loss_minus) / (2 * epsilon)
                g_anal = grad[idx]
                
                # Relative error
                rel_error = abs(g_num - g_anal) / max(1e-8, abs(g_num) + abs(g_anal))
                
                print(f"[{layer.__class__.__name__}.{param_name}{idx}] "
                      f"anal={g_anal:.6e}, num={g_num:.6e}, err={rel_error:.2e}")
                
                if rel_error > tolerance:
                    print("❌ Gradient check FAILED!")
                    return False
    
    print("✅ All gradients check out!")
    return True
