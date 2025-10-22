import LunarLearn.backend as backend

xp = backend.xp
DTYPE = backend.DTYPE

#Mean Squared Error
def MSE(Y, Y_pred):
    diff = Y - Y_pred
    loss = xp.mean(diff * diff, dtype=DTYPE)
    return loss

def MSE_derivative(Y, Y_pred):
    m = Y.shape[0]
    loss_derivative = DTYPE(2/m) * (Y_pred - Y)
    return loss_derivative

#Mean Absolute Error
def MAE(Y, Y_pred):
    loss = xp.mean(xp.abs(Y - Y_pred, dtype=DTYPE), dtype=DTYPE)
    return loss

def MAE_derivative(Y, Y_pred):
    m = Y.shape[0]
    loss_derivative = xp.where(Y_pred > Y, xp.array(1/m, dtype=DTYPE), xp.where(Y_pred < Y, xp.array(-1/m, dtype=DTYPE), 0))
    return loss_derivative

def Huber(Y, Y_pred, delta=1.0):
    delta = xp.array(delta, dtype=DTYPE)
    error = xp.mean(Y - Y_pred, dtype=DTYPE)
    if xp.abs(error) <= delta:
        error *= error
        loss = 0.5 * error
    else:
        loss = delta * (xp.abs(error) - 0.5 * delta)
    return loss

def Huber_derivative(Y, Y_pred, delta=1.0):
    delta = xp.array(delta, dtype=DTYPE)
    error = xp.mean(Y - Y_pred, dtype=DTYPE)
    if xp.abs(error) <= delta:
        loss_derivative = error
    else:
        loss_derivative = delta * xp.sign(error)
    return loss_derivative

#Binary Cross Entropy
def BCE(Y, Y_pred, epsilon=1e-15):
    epsilon = xp.array(epsilon, dtype=DTYPE)
    Y_pred = xp.clip(Y_pred, epsilon, 1-epsilon) #prevent log(0)
    m = Y.shape[0]
    loss = xp.sum(xp.matmul(Y, xp.log(Y_pred, dtype=DTYPE).T) + xp.matmul((1 - Y), xp.log(1 - Y_pred, dtype=DTYPE).T), dtype=DTYPE) * DTYPE(-1/m)
    loss = xp.squeeze(loss)
    return loss

def BCE_derivative(Y, Y_pred, epsilon=1e-15):
    epsilon = xp.array(epsilon, dtype=DTYPE)
    Y_pred = xp.clip(Y_pred, epsilon, 1-epsilon) #prevent log(0)
    inv_Y_pred = 1.0 / Y_pred
    inv_1m_pred = 1.0 / (1 - Y_pred)
    term = (Y * inv_Y_pred) - ((1 - Y) * inv_1m_pred)
    loss_derivative = xp.mean(-term, dtype=DTYPE)
    return loss_derivative

#Categorical Cross-Entropy
def CCE(Y, Y_pred, epsilon=1e-15):
    epsilon = xp.array(epsilon, dtype=DTYPE)
    #loss w.r.t Z while using softmax: y_pred - y
    Y_pred = xp.clip(Y_pred, epsilon, 1-epsilon) #prevent log(0)
    loss = xp.mean(-xp.sum(Y * xp.log(Y_pred, dtype=DTYPE), axis=1), dtype=DTYPE)
    return loss

def CCE_derivative(Y, Y_pred, epsilon=1e-15):
    epsilon = xp.array(epsilon, dtype=DTYPE)
    Y_pred = xp.clip(Y_pred, epsilon, 1-epsilon) #prevent log(0)
    inv_Y_pred = 1.0 / Y_pred
    term = Y * inv_Y_pred
    loss_derivative = xp.mean(-term, dtype=DTYPE)
    return loss_derivative

def CCE_softmax_derivative(Y, Y_pred):
    loss_derivative = Y_pred - Y
    return loss_derivative

#Sparse Categorical Cross-Entropy
def SCCE(Y, Y_pred, epsilon=1e-15):
    epsilon = xp.array(epsilon, dtype=DTYPE)
    Y_pred = xp.clip(Y_pred, epsilon, 1-epsilon) #prevent log(0)
    loss = xp.mean(-xp.log(Y_pred[xp.arange(len(Y), dtype=DTYPE), Y], dtype=DTYPE), dtype=DTYPE)
    return loss

def SCCE_derivative(Y, Y_pred, epsilon=1e-15):
    pass

def KL_div(Y, Y_pred):
    pass

def Reconstruction(Y, Y_pred):
    pass

def minimax(Y, Y_pred):
    pass

def Constrastive(Y, Y_pred):
    pass

def Triplet(Y, Y_pred):
    pass

def L1_loss(Y, layers, lambd):
    lambd = xp.array(lambd, dtype=DTYPE)
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.abs(layers[i].W, dtype=DTYPE), dtype=DTYPE)
    loss = W_sum * (lambd/Y.shape[0])
    return loss

def L2_loss(Y, layers, lambd):
    lambd = xp.array(lambd, dtype=DTYPE)
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.square(layers[i], dtype=DTYPE), dtype=DTYPE)
    loss = W_sum * (1/Y.shape[0]) * (lambd/2)
    return loss

def ElasticNet_loss(Y, layers, lambd1, lambd2):
    lambd1 = xp.array(lambd1, dtype=DTYPE)
    lambd2 = xp.array(lambd2, dtype=DTYPE)
    W_sum = 0
    for i in range(1, len(layers)):
        W_sum += xp.sum(xp.abs(layers[i].W, dtype=DTYPE), dtype=DTYPE) * lambd1 + xp.sum(xp.square(layers[i], dtype=DTYPE), dtype=DTYPE) * lambd2
    loss = W_sum * (1/Y.shape[0])
    return loss