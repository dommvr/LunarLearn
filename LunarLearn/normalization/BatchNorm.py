import numpy as np

class BatchNorm():
    def __init__(self, shape, beta=0.9, epsilon=1e-8):
        self.shape = shape
        self.beta = beta
        self.epsilon = epsilon
        self.gamma = np.ones((1, shape))
        self.beta = np.zeros((1, shape))
        self.running_mean = np.zeros((1, shape))
        self.running_var = np.ones((1, shape))

    def forward_propagation(self, Z, training=True):
        if training:
            # Compute batch mean and variance
            Z_mean = np.mean(Z, axis=0, keepdims=True)
            Z_var = np.var(Z, axis=0, keepdims=True)

            # Normalize
            self.Z_norm = (Z - Z_mean) / np.sqrt(Z_var + self.epsilon)

            # Update running estimates
            self.running_mean = self.beta * self.running_mean + (1 - self.beta) * Z_mean
            self.running_var = self.beta * self.running_var + (1 - self.beta) * Z_var

            # Scale and shift
            Z_norm = self.gamma * self.Z_norm + self.beta
        else:
            # Use running stats during inference
            Z_norm = (Z - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            Z_norm = self.gamma * Z_norm + self.beta

        return Z_norm

    def backward_propagation(self, dZ_norm):
        m = dZ_norm.shape[0]

        # Gradients for scale and shift
        d_gamma = np.sum(dZ_norm * self.Z_norm, axis=0, keepdims=True)
        d_beta = np.sum(dZ_norm, axis=0, keepdims=True)

        # Gradient w.r.t. normalized input
        dZ_norm = dZ_norm * self.gamma

        # Gradient w.r.t. variance
        d_var = np.sum(dZ_norm * self.X_centered, axis=0, keepdims=True) * -0.5 * self.std_inv**3

        # Gradient w.r.t. mean
        d_mean = np.sum(dZ_norm * -self.std_inv, axis=0, keepdims=True) + d_var * np.mean(-2. * self.X_centered, axis=0, keepdims=True)

        # Gradient w.r.t. input
        dZ = dZ_norm * self.std_inv + d_var * 2 * self.X_centered / m + d_mean / m

        # Store gradients
        self.d_gamma = d_gamma
        self.d_beta = d_beta

        return dZ
