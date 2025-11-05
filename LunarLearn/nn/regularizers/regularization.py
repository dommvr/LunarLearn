import numpy as np
import LunarLearn.backend as backend

xp = backend.xp
DTYPE = backend.DTYPE

def dropout(values, keep_prob):
    D = (xp.random.rand(*values.shape) < keep_prob)
    inv_keep_prob = 1.0 / keep_prob
    new_values = values * D.astype(values.dtype) * inv_keep_prob
    return new_values, D

def dropout_backward(values, keep_prob, D):
    inv_keep_prob = 1.0 / keep_prob
    new_values = (values * D.astype(values.dtype)) * inv_keep_prob
    return new_values

def L1_loss(Y, layers, lambd):
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.abs(layers[i].W, dtype=DTYPE), dtype=DTYPE)
    loss = W_sum * (lambd/Y.shape[1])
    return loss

def L2_loss(Y, layers, lambd):
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.square(layers[i], dtype=DTYPE), dtype=DTYPE)
    loss = W_sum * (1/Y.shape[1]) * (lambd/2)
    return loss

def ElasticNet_loss(Y, layers, lambd1, lambd2):
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.abs(layers[i].W, dtype=DTYPE), dtype=DTYPE) * lambd1 + xp.sum(xp.square(layers[i], dtype=DTYPE), dtype=DTYPE) * lambd2
    loss = W_sum * (1/Y.shape[1])
    return loss

def L1_backward(dW, W, lambd, m0):
    new_dW = dW + (lambd/m0) * xp.sign(W)
    return new_dW, dW

def L2_backward(dW, W, lambd, m0):
    new_dW = dW + (lambd/m0) * W
    return new_dW, dW

def ElasticNet_backward(dW, W, lambd1, lambd2, m0):
    new_dW = dW + (1/m0) * (lambd1 * xp.sign(W) + 2 * lambd2 * W)
    return new_dW, dW

def MaxNorm(W, max_norm=3.0, axis=1):
    max_norm = xp.array(max_norm, dtype=DTYPE)
    norm = xp.linalg.norm(W, axis=axis, keepdims=True).astype(DTYPE)
    desired = xp.clip(norm, 0, max_norm)
    new_W = W * (desired / (xp.array(1e-8, dtype=DTYPE))+norm)
    return new_W, W