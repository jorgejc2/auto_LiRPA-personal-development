import numpy as np
import torch
import math

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

def tanh(x):
    return np.tanh(x)

def torch_tanh(x):
    return torch.tanh(x)

def hardtanh(y):
    return np.minimum(np.maximum(y, -1), 1)

def sech(x):
    return 1/np.cosh(x)

def torch_sech(x):
    return 1/torch.cosh(x)

def exp(x):
    return np.exp(x)

def loglog_dr(x):
    return 1 - exp(-exp(x))

def geluopenai(x):
    return 0.5 * x * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x*x*x)))

def swish(x):
    return x*sigmoid(x)

def sigmoid_tanh(x, y):
    return sigmoid(x) * tanh(y)

def x_sigmoid(x, y):
    return x * sigmoid(y)

def sigmoid_hardtanh(x, y):
    return sigmoid(x) * hardtanh(y)

def loglog_x(x, y):
    return loglog_dr(x)*y

def loglog_tanh(x, y):
    return loglog_dr(x) * tanh(y)

def swish_grad(x):
    return np.expand_dims(sigmoid(x) + x * (1 - sigmoid(x)) * sigmoid(x), axis=-1)

def geluopenai_grad(x):
    grad = 0.5 * tanh(0.0356774 * x**3 + 0.797885 * x) + \
           (0.0535161 * x**3 + 0.398942*x) * sech(0.0356774 * x**3 + 0.797885 * x)**2 + 0.5
    return np.expand_dims(grad, axis=-1)

def loglog_grad(x):
    return np.expand_dims(exp(-exp(x) + x), axis=-1)

def sigmoid_grad(x):
    return np.expand_dims(sigmoid(x) * (1 - sigmoid(x)), axis=-1)

def torch_sigmoid_grad(x):
    return torch.expand_dims(torch_sigmoid(x) * (1 - torch_sigmoid(x)), axis=-1)

def tanh_grad(x):
    return np.expand_dims(sech(x) ** 2, axis=-1)

def torch_tanh_grad(x):
    return torch.expand_dims(torch_sech(x) ** 2, axis=-1)

def sigmoid_tanh_grad(x, y):
    grad_x = (1 - sigmoid(x))*sigmoid(x)*tanh(y)
    grad_y = sigmoid(x)*(sech(y)**2)
    return np.stack([grad_x, grad_y], axis=-1)
    
def x_sigmoid_grad(x, y):
    grad_x = sigmoid(y)
    grad_y = x*(1 - sigmoid(y))*sigmoid(y)
    return np.stack([grad_x, grad_y], axis=-1)

def sigmoid_hardtanh_grad(x, y):
    grad_x = (1-sigmoid(x))*sigmoid(x)*hardtanh(y)
    grad_y = np.where(np.logical_or(y < -1, y > 1), 0, sigmoid(x))
    return np.stack([grad_x, grad_y], axis=-1)

def loglog_x_grad(x, y):
    grad_x = exp(-exp(x) + x) * y
    grad_y = 1 - loglog_dr(x)
    return np.stack([grad_x, grad_y], axis=-1)

def loglog_tanh_grad(x, y):
    grad_x = exp(-exp(x) + x) * tanh(y)
    grad_y = loglog_dr(x) * (sech(y) ** 2)
    return np.stack([grad_x, grad_y], axis=-1)

STANDARD_ACTIVATIONS = {
    'ReLU': relu,
    'sigmoid': sigmoid,
    'tanh': tanh,
}