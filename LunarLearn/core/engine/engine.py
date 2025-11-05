import json
import numpy as np

from LunarLearn.initializations import He, Xavier, LeCun, Orthogonal
from LunarLearn.initializations import He_Conv2D, Xavier_Conv2D, LeCun_Conv2D, Orthogonal_Conv2D
from LunarLearn.activations import linear, sigmoid, ReLU, leaky_ReLU, TanH, softmax
from LunarLearn.activations import linear_derivative, sigmoid_derivative, ReLU_derivative, leaky_ReLU_derivative, TanH_derivative, softmax_derivative, softmax_CCE_derivative
from LunarLearn.loss import MSE, MAE, Huber, BCE, CCE, SCCE
from LunarLearn.loss import MSE_derivative, MAE_derivative, Huber_derivative, BCE_derivative, CCE_derivative, CCE_softmax_derivative, SCCE_derivative
from LunarLearn.fusion import sigBCE, sigMSE, softCCE, linMSE, linMAE, TanMSE
from LunarLearn.lr_decay import warm_up, early_stopping, time_based_decay, step_decay, fixed_step_decay, exp_decay, exponential_decay, polynomial_decay, cosine_annealing_decay, warm_restarts_cosine_annealing_decay, cyclical_decay, plateau
from LunarLearn.regularizers import L1_loss, L2_loss, L1_backward, L2_backward, ElasticNet_backward
from LunarLearn.layers import InputConv2D

import LunarLearn.backend as backend

xp = backend.xp
DTYPE = backend.DTYPE
C_DTYPE = backend.C_DTYPE
SAFE_FACTOR = backend.SAFE_FACTOR
MIXED_PRECISION = backend.MIXED_PRECISION
SCALLING_FACTOR = backend.SCALLING_FACTOR

if xp.__name__ == 'cupy':
    try:
        from cupyx.scatter_add import scatter_add
    except ImportError:
        from cupyx._scatter import scatter_add

def initialize_weights(name, activation, W_shape, b_shape, uniform, gain):
    if name == 'He':
        W = He(W_shape, uniform)
    elif name == 'Xavier':
        W = Xavier(W_shape, uniform)
    elif name == 'LeCun':
        W = LeCun(W_shape, uniform)
    elif name == 'orthogonal':
        W = Orthogonal(W_shape, gain)
    else:
        if activation == 'ReLU' or activation == 'leaky_ReLU':
            W = He(W_shape)
        else:
            W = Xavier(W_shape)
    
    if activation == 'ReLU' or activation == 'leaky_ReLU':
        b = xp.full(b_shape, 0.01, dtype=DTYPE)
    else:
        b = xp.zeros(b_shape, dtype=DTYPE)

    return W, b

def initialize_layers(layers):

    for i in range(1, len(layers)):

        trainable_layers = []
        if layers[i].trainable:
            trainable_layers.append(layers[i])
        
        input_shape = layers[i-1].output_shape
        layers[i].initialize(input_shape)

    return trainable_layers

def initialize_parameters(layers):
    for i in range(1, len(layers)):

        trainable_layers = []
        if hasattr(layers[i], 'W'):
            trainable_layers.append(layers[i])
        
        if layers[i].__class__.__name__ == 'Dense':
            if layers[i].w_init == 'He':
                layers[i].W = He((layers[i].nodes, layers[i-1].nodes), uniform=layers[i].uniform)
            elif layers[i].w_init == 'Xavier':
                layers[i].W = Xavier((layers[i].nodes, layers[i-1].nodes), uniform=layers[i].uniform)
            elif layers[i].w_init == 'LeCun':
                layers[i].W = LeCun((layers[i].nodes, layers[i-1].nodes), uniform=layers[i].uniform)
            elif layers[i].w_init == 'orthogonal':
                layers[i].W = Orthogonal((layers[i].nodes, layers[i-1].nodes), gain=layers[i].gain)
            else: #Auto initialization, choose best initialization for each layer activation function
                if layers[i].activation == 'TanH' or layers[i].activation == 'sigmoid' or layers[i].activation == 'softmax':
                    layers[i].W = Xavier((layers[i].nodes, layers[i-1].nodes))
                else: #ReLU
                    layers[i].W = He((layers[i].nodes, layers[i-1].nodes))
            #Bias initialization
            if layers[i].activation == 'ReLU' or layers[i].activation == 'leaky_ReLU':
                layers[i].b = xp.full((layers[i].nodes, 1), 0.01, dtype=DTYPE)
            else:
                layers[i].b = xp.zeros((layers[i].nodes, 1), dtype=DTYPE)

            if MIXED_PRECISION:
                layers[i].W_fp16 = layers[i].W.astype(C_DTYPE)
                layers[i].b_fp16 = layers[i].b.astype(C_DTYPE)

        elif layers[i].__class__.__name__ == 'Conv2D':
            if layers[i].w_init == 'He':
                layers[i].W = He_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'Xavier':
                layers[i].W = Xavier_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'LeCun':
                layers[i].W = LeCun_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'orthogonal':
                layers[i].W = Orthogonal_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C), gain=layers[i].gain)
            else: #Auto initialization, choose best initialization for each layer activation function
                if layers[i].activation == 'TanH' or layers[i].activation == 'sigmoid' or layers[i].activation == 'softmax':
                    layers[i].W = Xavier_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C))
                else: #ReLU
                    #layers[i].W = He_Conv2D((layers[i].f, layers[i].f, layers[i-1].n_C, layers[i].n_C))
                    layers[i].W = He_Conv2D((layers[i].n_C, layers[i-1].n_C, layers[i].f, layers[i].f))

            #Bias initialization
            '''if layers[i].activation == 'ReLU' or layers[i].activation == 'leaky_ReLU':
                layers[i].b = xp.full((1, 1, 1, layers[i].n_C), 0.01, dtype=DTYPE)
            else:
                layers[i].b = xp.zeros((1, 1, 1, layers[i].n_C), dtype=DTYPE)'''

            if layers[i].activation == 'TanH' or layers[i].activation == 'sigmoid' or layers[i].activation == 'softmax':
                layers[i].b = xp.zeros((layers[i].n_C, 1))
            else:
                layers[i].b = xp.full((layers[i].n_C, 1), 0.01, dtype=DTYPE)

            if MIXED_PRECISION:
                layers[i].W_fp16 = layers[i].W.astype(C_DTYPE)
                layers[i].b_fp16 = layers[i].b.astype(C_DTYPE)

            #Conv2D m, n_H, n_W initialization
            if layers[i].padding == 'same':
                layers[i].padding = ((layers[i].strides-1) * layers[i-1].n_H + layers[i].f - layers[i].strides) // 2
            layers[i].n_H = int((layers[i-1].n_H + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_W = int((layers[i-1].n_W + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_C_prev = layers[i-1].n_C

        elif layers[i].__class__.__name__ == 'Conv2DTranspose':
            if layers[i].w_init == 'He':
                layers[i].W = He_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'Xavier':
                layers[i].W = Xavier_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'LeCun':
                layers[i].W = LeCun_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C), uniform=layers[i].uniform)
            elif layers[i].w_init == 'orthogonal':
                layers[i].W = Orthogonal_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C), gain=layers[i].gain)
            else: #Auto initialization, choose best initialization for each layer activation function
                if layers[i].activation == 'TanH' or layers[i].activation == 'sigmoid' or layers[i].activation == 'softmax':
                    layers[i].W = Xavier_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C))
                else: #ReLU
                    layers[i].W = He_Conv2D((layers[i].f, layers[i].f, layers[i].n_C, layers[i-1].n_C))

            #Bias initialization
            if layers[i].activation == 'ReLU' or layers[i].activation == 'leaky_ReLU':
                layers[i].b = xp.full((1, 1, 1, layers[i].n_C), 0.01, dtype=DTYPE)
            else:
                layers[i].b = xp.zeros((1, 1, 1, layers[i].n_C), dtype=DTYPE)

            #Conv2D m, n_H, n_W initialization
            if layers[i].padding == 'same':
                layers[i].padding = ((layers[i].strides-1) * layers[i-1].n_H + layers[i].f - layers[i].strides) // 2
            layers[i].n_H = int((layers[i-1].n_H + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_W = int((layers[i-1].n_W + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_C_prev = layers[i-1].n_C

        elif layers[i].__class__.__name__ == 'MaxPool2D' or layers[i].__class__.__name__ == 'AveragePool2D':
            if layers[i].padding == 'same':
                layers[i].padding = ((layers[i].strides-1) * layers[i-1].n_H + layers[i].f - layers[i].strides) // 2
            layers[i].n_H = int((layers[i-1].n_H + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_W = int((layers[i-1].n_W + 2 * layers[i].padding - layers[i].f) / layers[i].strides) + 1
            layers[i].n_C = layers[i-1].n_C

        elif layers[i].__class__.__name__ == 'Flatten':
            layers[i].nodes = layers[i-1].n_H * layers[i-1].n_W * layers[i-1].n_C

        elif layers[i].__class__.__name__ == 'BatchNorm2D':
            '''layers[i].W = xp.ones((1, 1, 1, layers[i-1].n_C), dtype=DTYPE)
            layers[i].b = xp.zeros((1, 1, 1, layers[i-1].n_C), dtype=DTYPE)
            layers[i].running_mean = xp.zeros((1, 1, 1, layers[i-1].n_C), dtype=DTYPE)
            layers[i].running_var = xp.ones((1, 1, 1, layers[i-1].n_C), dtype=DTYPE)'''
            layers[i].W = xp.ones((1, layers[i-1].n_C, 1, 1), dtype=DTYPE)
            layers[i].b = xp.zeros((1, layers[i-1].n_C, 1, 1), dtype=DTYPE)
            layers[i].W_fp16 = layers[i].W.astype(C_DTYPE)
            layers[i].b_fp16 = layers[i].b.astype(C_DTYPE)
            layers[i].running_mean = xp.zeros((1, layers[i-1].n_C, 1, 1), dtype=DTYPE)
            layers[i].running_var = xp.ones((1, layers[i-1].n_C, 1, 1), dtype=DTYPE)
            layers[i].n_H = layers[i-1].n_H
            layers[i].n_W = layers[i-1].n_W
            layers[i].n_C = layers[i-1].n_C

        elif layers[i].__class__.__name__ == 'ResBlock':
            layers[i].layers.insert(0, InputConv2D(shape=layers[i-1].A.shape))
            initialize_parameters(layers[i].layers)
            initialize_parameters([layers[i].layers[0], layers[i].match_dim_layer])
        
        elif layers[i].__class__.__name__ == 'ReLU':
            layers[i].n_H = layers[i-1].n_H
            layers[i].n_W = layers[i-1].n_W
            layers[i].n_C = layers[i-1].n_C

    return trainable_layers      

def activation(x, activation_name):
    if activation_name == 'linear':
        return linear(x)
    elif activation_name == 'sigmoid':
        return sigmoid(x)
    elif activation_name == 'ReLU':
        return ReLU(x)
    elif activation_name == 'leaky_ReLU':
        return leaky_ReLU(x)
    elif activation_name == 'TanH':
        return TanH(x)
    elif activation_name == 'softmax':
        return softmax(x)
    
def activation_derivative(x, activation_name):
    if activation_name == 'linear':
        return linear_derivative(x)
    elif activation_name == 'sigmoid':
        return sigmoid_derivative(x)
    elif activation_name == 'ReLU':
        return ReLU_derivative(x)
    elif activation_name == 'leaky_ReLU':
        return leaky_ReLU_derivative(x)
    elif activation_name == 'TanH':
        return TanH_derivative(x)
    elif activation_name == 'softmax':
        return softmax_derivative(x)

def random_mini_batches(X, Y, mini_batch_size=64, seed=11):
    xp.random.seed(seed)
    mini_batches = []

    if mini_batch_size == 0:
        return [(X, Y)]
    
    if X.ndim == 2:
        m = X.shape[1]
    elif X.ndim == 4:
        m = X.shape[0]
    else:
        raise ValueError('expected 2D or 4D array')
    
    permutation = xp.random.permutation(m).astype(xp.int64)

    if X.ndim == 2:
        X_shuffled = X[:, permutation]
        if Y is not None:
            Y_shuffled = Y[:, permutation]
    else:
        X_shuffled = X[permutation, :, :, :]
        #Y_shuffled = Y[permutation, :]  # assuming Y is 2D for classification
        if Y is not None:
            Y_shuffled = Y[:, permutation]

    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        start = k * mini_batch_size
        end = (k + 1) * mini_batch_size
        X_batch = X_shuffled[:, start:end] if X.ndim == 2 else X_shuffled[start:end]
        if Y is not None:
            Y_batch = Y_shuffled[:, start:end] if X.ndim == 2 else Y_shuffled[:, start:end]#Y_shuffled[start:end]
        else:
            Y_batch = Y
        mini_batches.append((X_batch, Y_batch))

    if m % mini_batch_size != 0:
        X_batch = X_shuffled[:, num_complete_minibatches * mini_batch_size:] if X.ndim == 2 else X_shuffled[num_complete_minibatches * mini_batch_size:]
        if Y is not None:
            Y_batch = Y_shuffled[:, num_complete_minibatches * mini_batch_size:] if X.ndim == 2 else Y_shuffled[:, num_complete_minibatches * mini_batch_size:]#Y_shuffled[num_complete_minibatches * mini_batch_size:]
        else:
            Y_batch = Y
        mini_batches.append((X_batch, Y_batch))

    return mini_batches

def order_mini_batches(X, Y, mini_batch_size=64):
    mini_batches = []

    if mini_batch_size == 0:
        return [(X, Y)]
    
    if X.ndim == 2:
        m = X.shape[1]
    elif X.ndim == 4:
        m = X.shape[0]
    else:
        raise ValueError('expected 2D or 4D array')

    num_complete_minibatches = m // mini_batch_size

    for k in range(num_complete_minibatches):
        start = k * mini_batch_size
        end = (k + 1) * mini_batch_size
        X_batch = X[:, start:end] if X.ndim == 2 else X[start:end]
        if Y is not None:
            Y_batch = Y[:, start:end] if X.ndim == 2 else Y[:, start:end]#Y_shuffled[start:end]
        else:
            Y_batch = Y
        mini_batches.append((X_batch, Y_batch))

    if m % mini_batch_size != 0:
        X_batch = X[:, num_complete_minibatches * mini_batch_size:] if X.ndim == 2 else X[num_complete_minibatches * mini_batch_size:]
        if Y is not None:
            Y_batch = Y[:, num_complete_minibatches * mini_batch_size:] if X.ndim == 2 else Y[:, num_complete_minibatches * mini_batch_size:]#Y_shuffled[num_complete_minibatches * mini_batch_size:]
        else:
            Y_batch = Y
        mini_batches.append((X_batch, Y_batch))

    return mini_batches

def learning_rate_decay(optimizer, epoch, epochs, cost, name, decay_rate, milestones, lr_min, lr_max, learning_rate0, warmup, patience, multiplier, p):
    if learning_rate0 > 0:
        optimizer.learning_rate0 = learning_rate0
    if lr_max == 0:
        lr_max = optimizer.learning_rate0
    if warmup > 0:
        if epoch < warmup:
            return warm_up(lr_max, epoch, warmup)
    if name == 'time_based':
        return time_based_decay(optimizer, epoch, decay_rate)
    elif name == 'stepy':
        return step_decay(optimizer, epoch, milestones, decay_rate)
    elif name == 'fixed_step':
        return fixed_step_decay(optimizer, epoch, milestones, decay_rate)
    elif name == 'exp':
        return exp_decay(optimizer, epoch, decay_rate)
    elif name == 'exponential':
        return exponential_decay(optimizer, epoch, decay_rate, milestones)
    elif name == 'polynomial':
        return polynomial_decay(optimizer, epoch, epochs, p)
    elif name == 'cosine_annealing':
        return cosine_annealing_decay(epoch, epochs, lr_min, lr_max)
    elif name == 'warm_restarts_cosine':
        return warm_restarts_cosine_annealing_decay(epoch, lr_min, lr_max, milestones, multiplier)
    elif name == 'cyclical':
        return cyclical_decay(optimizer, epoch, milestones, lr_max)
    elif name == 'plateau':
        return plateau(optimizer, cost, decay_rate, patience)
    
def n_correct_outputs(Y, Y_pred):
    labels = xp.argmax(Y, axis=0)
    labels_pred = xp.argmax(Y_pred, axis=0)
    n_correct = xp.sum(labels == labels_pred, dtype=DTYPE)
    return n_correct

def forward_propagation(X, layers, training=True):
    layers[0].A = X
    for i in range(1, len(layers)):
        layers[i].forward_step(layers[i-1].A, training)

        if layers[i].A is not None:
            # Always: raw activation
            if layers[i].has_hooks("activation"):
                layers[i].run_hooks("activation", layers[i].A)

            # Conditional probes
            if layers[i].has_hooks("activation_norm"):
                act_norm = xp.linalg.norm(layers[i].A)
                layers[i].run_hooks("activation_norm", act_norm)

            if layers[i].has_hooks("activation_entropy"):
                probs = xp.clip(layers[i].A, 1e-9, 1.0)
                entropy = float(-xp.mean(probs * xp.log(probs)))
                layers[i].run_hooks("activation_entropy", entropy)

def compute_loss(Y, Y_pred, loss_name, layers, regularization, regularization_args):
    if regularization:
        lambd = regularization_args['lambd']
    #if loss_name == 'auto': --> choose best loss function for model
    if loss_name == 'MSE':
        loss =  MSE(Y, Y_pred)
    elif loss_name == 'MAE':
        loss = MAE(Y, Y_pred)
    elif loss_name == 'Huber':
        loss = Huber(Y, Y_pred)
    elif loss_name == 'BCE':
        loss = BCE(Y, Y_pred)
    elif loss_name == 'CCE':
        loss = CCE(Y, Y_pred)
    elif loss_name == 'SCCE':
        loss = SCCE(Y, Y_pred)
    if regularization == 'L1':
        loss += L1_loss(Y, layers, lambd)
    elif regularization == 'L2':
        loss += L2_loss(Y, layers, lambd)
    return loss

def compute_loss_derivative(Y, Y_pred, loss_name, layers):
    if loss_name == 'MSE':
        loss_derivative = MSE_derivative(Y, Y_pred)
    elif loss_name == 'MAE':
        loss_derivative = MAE_derivative(Y, Y_pred)
    elif loss_name == 'Huber':
        loss_derivative = Huber_derivative(Y, Y_pred)
    elif loss_name == 'BCE':
        loss_derivative = BCE_derivative(Y, Y_pred)
    elif loss_name == 'CCE' and layers[-1].activation == 'softmax':
        loss_derivative = CCE_softmax_derivative(Y, Y_pred)
    elif loss_name == 'CCE':
        loss_derivative = CCE_derivative(Y, Y_pred)
    elif loss_name == 'SCCE':
        loss_derivative = SCCE_derivative(Y, Y_pred)
    return loss_derivative
    
def backward_propagation(loss_name, loss_derivative, fused_dZ, layers):
    if loss_derivative is not None:
        layers[-1].dA = loss_derivative
    else:
        layers[-1].dZ = fused_dZ

    for i in reversed(range(1, len(layers))): #normalize params throw diffrent layers
        layers[i-1].dA = layers[i].backward_step(layers[i-1].A, loss_name)

        if layers[i].dW is not None:
            if layers[i].has_hooks("grad"):
                layers[i].run_hooks("grad", layers[i].dW)

            if layers[i].has_hooks("grad_norm"):
                grad_norm = xp.linalg.norm(layers[i].dW)
                layers[i].run_hooks("grad_norm", grad_norm)

            if layers[i].has_hooks("fisher"):
                fisher = float(xp.mean(layers[i].dW ** 2))
                layers[i].run_hooks("fisher", fisher)

            if layers[i].has_hooks("grad_snr"):
                grad_mean = float(xp.mean(layers[i].dW))
                grad_std = float(xp.std(layers[i].dW) + 1e-9)
                snr = grad_mean / grad_std
                layers[i].run_hooks("grad_snr", snr)

        if layers[i].hessian.get("HdW") is not None and layers[i].has_hooks("hessian"):
            hessian_norm = xp.linalg.norm(layers[i].hessian["HdW"])
            layers[i].run_hooks("hessian", hessian_norm)

def dW_regularization(dW, W, regularization_name, regularization_args, m0):
    lambd = regularization_args['lambd']
    lambd1 = regularization_args['lambd1']
    lambd2 = regularization_args['lambd2']
    if regularization_name == 'L1':
        new_dW, old_dW = L1_backward(dW, W, lambd, m0) 
    elif regularization_name == 'L2':
        new_dW, old_dW = L2_backward(dW, W, lambd, m0)
    elif regularization_name == 'ElasticNet':
        new_dW, old_dW = ElasticNet_backward(dW, W, lambd1, lambd2, m0)
    return new_dW, old_dW

def print_epoch_info(epoch, epochs, n_train_examples, n_train_correct, n_test_examples, n_test_correct, epoch_loss, print_every):
    accuracy = round(((n_train_correct / n_train_examples) * 100).item(), 2)
    test_accuracy = round(((n_test_correct / n_test_examples) * 100).item(), 2)
    if epoch == 0:
        print(f'Epoch: {epoch+1}/{epochs}   accuracy: {accuracy}%   test_accuracy: {test_accuracy}%    loss: {epoch_loss}')
    if (epoch+1) % print_every == 0:
        print(f'Epoch: {epoch+1}/{epochs}   accuracy: {accuracy}%   test_accuracy: {test_accuracy}%    loss: {epoch_loss}')

def add_history(history, n_train_examples, n_train_correct, n_test_examples, n_test_correct, loss, test_loss, learning_rate):
    accuracy = round(((n_train_correct / n_train_examples) * 100).item(), 2)
    test_accuracy = round(((n_test_correct / n_test_examples) * 100).item(), 2)
    history['loss'].append(loss)
    history['accuracy'].append(accuracy)
    history['test_loss'].append(test_loss)
    history['test_accuracy'].append(test_accuracy)
    history['learning_rate'].append(learning_rate)
    #return history

def zero_padding(X, pad, kernel_size=0, strides=0):
    #X.shape = (m. n_H, n_W, n_C)
    if pad == 'same':
        pad = ((strides-1) * X.shape[1] + kernel_size - strides) // 2
        X_pad = xp.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=((0,0), (0,0), (0,0), (0,0)))
    else:
        X_pad = xp.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), 'constant', constant_values=((0,0), (0,0), (0,0), (0,0)))
    return X_pad

def conv2D_kernel_operation(X_slice, W, b):
    Z = xp.sum(X_slice * W, dtype=DTYPE)
    Z += b
    return Z

def create_mask_from_window(window):
    mask = window == xp.max(window)
    return mask

def distribute_value(dZ, shape):
    (n_H, n_W) = shape
    denom = n_H * n_W
    inv_denom = 1.0 / denom
    average = dZ * inv_denom
    distribution = xp.ones(shape, dtype=DTYPE) * average
    return distribution

def safe_batch_size(X_shape, f, s, safety_factor=SAFE_FACTOR):
    """
    Estimate safe batch size based on GPU free memory.
    X_shape expected NCHW: (m, C, H, W)
    """
    m, C, H, W = X_shape
    H_out = (H - f) // s + 1
    W_out = (W - f) // s + 1

    # number of elements produced by im2col per image
    cols_per_image = f * f * C * H_out * W_out
    bytes_per_image = int(cols_per_image * xp.dtype(DTYPE).itemsize)

    if xp.__name__ == "cupy":
        free_bytes, _ = xp.cuda.runtime.memGetInfo()
        avail = int(free_bytes * safety_factor)
    else:
        # on CPU assume we can take whole dataset (or a large default)
        avail = bytes_per_image * m  # allow full batch

    batch_size = max(1, min(m, avail // max(1, bytes_per_image)))
    return batch_size

def im2col_vectorized(X, f, s):
    m, n_C, n_H, n_W = X.shape
    n_H_out = (n_H - f) // s + 1
    n_W_out = (n_W - f) // s + 1

    i0 = xp.repeat(xp.arange(f), f)
    i0 = xp.tile(i0, n_C)
    i1 = s * xp.repeat(xp.arange(n_H_out), n_W_out)

    j0 = xp.tile(xp.arange(f), f * n_C)
    j1 = s * xp.tile(xp.arange(n_W_out), n_H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = xp.repeat(xp.arange(n_C), f * f).reshape(-1, 1)

    k = k.astype(xp.int32)
    i = i.astype(xp.int32)
    j = j.astype(xp.int32)

    cols = X[:, k, i, j]  # (m, f_h * f_w * n_C, n_H_out * n_W_out)
    cols = cols.transpose(1, 2, 0).reshape(f * f * n_C, -1)
    return cols

def im2col_safe_batch(X, f, s):
    m = X.shape[0]
    batch = safe_batch_size(X.shape, f, s)
    out_list = []
    for start in range(0, m, batch):
        end = min(start + batch, m)
        Xb = X[start:end]
        out_list.append(im2col_vectorized(Xb, f, s))
    return xp.concatenate(out_list, axis=1).astype(DTYPE)


def im2col(X, f, s):
    # try to do full vectorized; caller may catch OOM and call safe batch
    try:
        return im2col_vectorized(X, f, s)
    except Exception as ex:
        # On CuPy an OOM is raised as cupy.cuda.memory.OutOfMemoryError,
        # but catching general exceptions ensures fallback.
        return im2col_safe_batch(X, f, s)


def col2im_vectorized(cols, X_shape, f, s):
    """
    Channel-first col2im: X_shape = (m, C, H, W)
    cols shape (C*f*f, H_out*W_out*m)
    """
    m, C, H, W = X_shape
    H_out = (H - f) // s + 1
    W_out = (W - f) // s + 1

    # Same index generation as im2col
    i0 = xp.repeat(xp.arange(f), f)
    i0 = xp.tile(i0, C)
    i1 = s * xp.repeat(xp.arange(H_out), W_out)

    j0 = xp.tile(xp.arange(f), f * C)
    j1 = s * xp.tile(xp.arange(W_out), H_out)

    i = i0.reshape(-1, 1) + i1.reshape(1, -1)   # (C*f*f, H_out*W_out)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)   # (C*f*f, H_out*W_out)
    k = xp.repeat(xp.arange(C), f * f).reshape(-1, 1)  # (C*f*f, 1)

    # Prepare flat indices
    cols_reshaped = cols.reshape(C * f * f, H_out * W_out, m)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)  # (m, C*f*f, H_out*W_out)

    #batch_idx = xp.repeat(xp.arange(m), C * f * f * H_out * W_out)
    batch_idx = xp.repeat(xp.arange(m), i.size)
    total = batch_idx.size
    k_idx = xp.tile(k, (1, H_out * W_out)).ravel()
    k_idx = xp.tile(k, m)
    k_idx = xp.repeat(xp.arange(C), f * f * H_out * W_out)
    k_idx = xp.tile(k_idx, int(total / k_idx.size))
    i_idx = xp.tile(i, (m, 1)).ravel()
    j_idx = xp.tile(j, (m, 1)).ravel()
    k_idx = k_idx.astype(xp.int32)
    i_idx = i_idx.astype(xp.int32)
    j_idx = j_idx.astype(xp.int32)

    vals = cols_reshaped.ravel()
    flat_idx = xp.ravel_multi_index((batch_idx, k_idx, i_idx, j_idx), X_shape)

    X_flat = xp.zeros(m * C * H * W, dtype=cols.dtype)
    if xp.__name__ == 'cupy':
        scatter_add(X_flat, flat_idx, vals)
    else:
        xp.add.at(X_flat, flat_idx, vals)

    return X_flat.reshape(X_shape)


def col2im_safe_batch(cols, X_shape, f, s):
    m, C, H, W = X_shape
    batch = safe_batch_size(X_shape, f, s)
    H_out = (H - f) // s + 1
    W_out = (W - f) // s + 1
    patches = H_out * W_out

    X = xp.zeros(X_shape, dtype=cols.dtype)
    for start in range(0, m, batch):
        end = min(start + batch, m)
        cols_batch = cols[:, start * patches:end * patches]
        X_batch = col2im_vectorized(cols_batch, (end - start, C, H, W), f, s)
        X[start:end] = X_batch
    return X


def col2im(cols, X_shape, f, s):
    try:
        return col2im_vectorized(cols, X_shape, f, s)
    except Exception:
        return col2im_safe_batch(cols, X_shape, f, s)

def unscale_grads(layers):
    for layer in layers:
        if hasattr(layer, 'dW'):
            layer.dW = layer.dW.astype(DTYPE) / SCALLING_FACTOR
        if hasattr(layer, 'db'):
            layer.db = layer.db.astype(DTYPE) / SCALLING_FACTOR

def check_for_fusion(activation_name, loss_name):
    if activation_name == 'sigmoid' and loss_name == 'BinaryCrossEntropyLoss':
        return True, 'sigBCE'
    elif activation_name == 'sigmoid' and loss_name == 'MeanSquaredError':
        return True, 'sigMSE' 
    elif activation_name == 'softmax' and loss_name == 'CrossEntropyLoss':
        return True, 'softCCE'
    elif activation_name == 'linear' and loss_name == 'MeanSquaredError':
        return True, 'linMSE'
    elif activation_name == 'linear' and loss_name == 'MeanAbsoluteError':
        return True, 'linMAE'
    elif activation_name == 'TanH' and loss_name == 'MeanSquaredError':
        return True, 'TanMSE'
    else:
        return False, None
    
def fuse(Y, Y_pred, fusion_type):
    if fusion_type == 'sigBCE':
        return sigBCE(Y, Y_pred)
    elif fusion_type == 'sigMSE':
        return sigMSE(Y, Y_pred)
    elif fusion_type == 'softCCE':
        return softCCE(Y, Y_pred)
    elif fusion_type == 'linMSE':
        return linMSE(Y, Y_pred)
    elif fusion_type == 'linMAE':
        return linMAE(Y, Y_pred)
    elif fusion_type == 'TanMSE':
        return TanMSE(Y, Y_pred)

def iou(box1, box2):
    (box1_x1, box1_y1, box1_x2, box1_y2) = box1
    (box2_x1, box2_y1, box2_x2, box2_y2) = box2

    xi1 = max(box1_x1, box2_x1)
    yi1 = max(box1_y1, box2_y1)
    xi2 = min(box1_x2, box2_x2)
    yi2 = min(box1_y2, box2_y2)

    inter_width = max((xi2-xi1), 0)
    inter_height =  max((yi2-yi1), 0)
    inter_area = inter_width * inter_height
    
    box1_area = (box1_x2-box1_x1) * (box1_y2-box1_y1)
    box2_area = (box2_x2-box2_x1) * (box2_y2-box2_y1)
    
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    
    return iou

def serialize_value(val):
    """Convert Python/numpy objects into JSON-safe formats."""
    if isinstance(val, (int, float, str, bool)) or val is None:
        return val
    if isinstance(val, xp.generic):  # e.g. np.float32
        return val.item()
    if isinstance(val, (list, tuple)):
        return [serialize_value(v) for v in val]
    if isinstance(val, xp.ndarray):
        return val.tolist()
    if callable(val):  # activation functions, schedulers etc.
        return {
            "__callable__": True,
            "module": val.__module__,
            "name": val.__name__
        }
    # fallback: stringify (last resort)
    return str(val)

def deserialize_value(val):
    """Convert JSON-safe formats back to Python objects."""
    import importlib

    if isinstance(val, dict) and val.get("__callable__"):
        module = importlib.import_module(val["module"])
        return getattr(module, val["name"])
    return val
    
def load_from_config(config):
    """
    Universal loader for any LunarLearn object (Layer, Loss, Optimizer, Scheduler, NeuralNetwork).
    Expects a dict with: {"module": str, "class": str, "params": dict}
    """

    import importlib

    module_name = config["module"]
    class_name = config["class"]
    params = config.get("params", {})

    # Dynamically import module
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    # Recursively rebuild nested configs
    for key, val in params.items():
        if isinstance(val, dict) and "module" in val and "class" in val:
            params[key] = load_from_config(val)
        elif isinstance(val, list):
            # Handle lists of configs (e.g. layers in NeuralNetwork)
            params[key] = [
                load_from_config(v) if isinstance(v, dict) and "module" in v and "class" in v else v
                for v in val
            ]

    return cls(**params)

def object_from_config(config, **kwargs):
    """
    Create an object from a config dictionary.
    
    Args:
        config (dict): Must have keys "module", "class", "params", optionally "extra".
    
    Returns:
        object: Initialized object with extra attributes set.
    """
    import importlib

    if config is None:
        return None

    # Import class
    module = importlib.import_module(config["module"])
    klass = getattr(module, config["class"])

    # Merge params and extra, giving kwargs highest priority
    init_args = {}
    if "params" in config:
        init_args.update(config["params"])
    if "extra" in config:
        init_args.update(config["extra"])
    init_args.update(kwargs)  # e.g., optimizer link for scheduler

    if hasattr(klass, "from_config"):
        return klass.from_config(config, **kwargs)
    else:
        return klass(**init_args)
    

from LunarLearn.tensor import Tensor

def accuracy(preds: Tensor, targets: Tensor, threshold: float = 0.5) -> float:
    """
    Compute classification accuracy (binary or multi-class).

    This function automatically detects whether the task is binary
    (preds shape (B, 1)) or multi-class (preds shape (B, C)).

    Args:
        preds (Tensor): Model predictions.
            - Binary: shape (B, 1), typically sigmoid output.
            - Multi-class: shape (B, C), typically softmax output.
        targets (Tensor): Ground-truth labels.
            - Binary: shape (B,) or (B, 1).
            - Multi-class: shape (B,) with class indices or (B, C) one-hot.
        threshold (float, optional): Probability threshold for binary
            classification. Default is 0.5.

    Returns:
        float: Accuracy in [0, 1].
    """
    preds_data = preds.data

    # ----- Binary classification -----
    if preds_data.ndim == 2 and preds_data.shape[1] == 1:
        pred_labels = (preds_data > threshold).astype(int).reshape(-1)
        targets = targets.reshape(-1).astype(int)

    # ----- Multi-class classification -----
    else:
        pred_labels = preds_data.argmax(axis=1)
        if targets.ndim == 2:  # one-hot
            targets = targets.argmax(axis=1)

    correct = (pred_labels == targets).sum()
    return (correct / len(targets)).item()