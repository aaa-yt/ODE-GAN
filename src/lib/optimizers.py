import sys
sys.path.append("../")
from config import Config
import numpy as np


class Optimizer(object):
    def __init__(self, config: Config, network_type):
        self.config = config
        self.mc = config.generator if network_type == "generator" else config.discriminator
        self.tc = config.trainer
    

class SGD(Optimizer):
    def __init__(self, config: Config, network_type):
        super(SGD, self).__init__(config, network_type)
        self.rate = self.tc.rate
    
    def __call__(self, params, g_params):
        return tuple(param - self.rate * g_param for param, g_param in zip(params, g_params))
    

class Momentum(Optimizer):
    def __init__(self, config: Config, network_type):
        super(Momentum, self).__init__(config, network_type)
        self.rate = self.tc.rate
        self.momentum = self.tc.momentum
        alpha_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        beta_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        A_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_out), dtype=np.float32)
        alpha_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_out), dtype=np.float32)
        gamma_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        A_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)

    def __call__(self, params, g_params):
        new_params = tuple(param + self.momentum * (param - v) - self.rate * g_param for param, g_param, v in zip(params, g_params, self.v))
        self.v = params
        return new_params


class AdaGrad(Optimizer):
    def __init__(self, config: Config, network_type):
        super(AdaGrad, self).__init__(config, network_type)
        self.rate = self.tc.rate
        alpha_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        beta_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        A_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_out), dtype=np.float32)
        alpha_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_out), dtype=np.float32)
        gamma_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        A_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((v + np.square(g_param) + self.eps).astype(np.float32))), g_param), v + np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class RMSprop(Optimizer):
    def __init__(self, config: Config, network_type):
        super(RMSprop, self).__init__(config, network_type)
        self.rate = self.tc.rate
        self.decay = self.tc.decay
        alpha_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        beta_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        A_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_out), dtype=np.float32)
        alpha_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_out), dtype=np.float32)
        gamma_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        A_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32))), g_param), self.decay * v + (1. - self.decay) * np.square(g_param)) for param, g_param, v in zip(params, g_params, self.v)])
        self.v = new_v
        return new_params


class AdaDelta(Optimizer):
    def __init__(self, config: Config, network_type):
        super(AdaDelta, self).__init__(config, network_type)
        self.decay = self.tc.decay
        alpha_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        beta_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        A_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_out), dtype=np.float32)
        alpha_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_out), dtype=np.float32)
        gamma_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        A_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.s = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.params_prev = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.eps = 1e-8
    
    def __call__(self, params, g_params):
        new_params, new_v, new_s = zip(*[(param - np.multiply(np.divide(np.sqrt((self.decay * s + (1. - self.decay) * np.square(param - param_prev) + self.eps).astype(np.float32)), np.sqrt((self.decay * v + (1. - self.decay) * np.square(g_param) + self.eps).astype(np.float32))), g_param), self.decay * v + (1. - self.decay) * np.square(g_param), self.decay * s + (1. - self.decay) * np.square(param - param_prev)) for param, g_param, v, s, param_prev in zip(params, g_params, self.v, self.s, self.params_prev)])
        self.param_prev = params
        self.v = new_v
        self.s = new_s
        return new_params


class Adam(Optimizer):
    def __init__(self, config: Config, network_type):
        super(Adam, self).__init__(config, network_type)
        self.rate = self.tc.rate
        self.decay1 = self.tc.decay
        self.decay2 = self.tc.decay2
        alpha_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        beta_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_in), dtype=np.float32)
        gamma_x = np.zeros(shape=(self.mc.division, self.mc.dim_in), dtype=np.float32)
        A_x = np.zeros(shape=(self.mc.division, self.mc.dim_in, self.mc.dim_out), dtype=np.float32)
        alpha_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        beta_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_out), dtype=np.float32)
        gamma_y = np.zeros(shape=(self.mc.division, self.mc.dim_out), dtype=np.float32)
        A_y = np.zeros(shape=(self.mc.division, self.mc.dim_out, self.mc.dim_in), dtype=np.float32)
        self.v = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.s = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.t = 1
        self.eps = 1e-8

    def __call__(self, params, g_params):
        new_params, new_v, new_s = zip(*[(param - np.multiply(np.divide(self.rate, np.sqrt((np.divide(self.decay2 * s + (1. - self.decay2) * np.square(g_param), 1. - self.decay2 ** self.t) + self.eps).astype(np.float32))), np.divide(self.decay1 * v + (1. - self.decay1) * g_param, 1. - self.decay1 ** self.t)), self.decay1 * v + (1. - self.decay1) * g_param, self.decay2 * s + (1. - self.decay2) * np.square(g_param)) for param, g_param, v, s in zip(params, g_params, self.v, self.s)])
        self.v = new_v
        self.s = new_s
        self.t += 1
        return new_params


def get(config: Config, network_type):
    tc = config.trainer
    all_optimizer = {
        "sgd": SGD(config=config, network_type=network_type),
        "momentum": Momentum(config=config, network_type=network_type),
        "adagrad": AdaGrad(config=config, network_type=network_type),
        "rmsprop": RMSprop(config=config, network_type=network_type),
        "adadelta": AdaDelta(config=config, network_type=network_type),
        "adam": Adam(config=config, network_type=network_type)
    }

    optimizer_type = tc.optimizer_type
    if optimizer_type.lower() in all_optimizer:
        optimizer_type = optimizer_type.lower()
        return all_optimizer[optimizer_type]
    else:
        return all_optimizer["sgd"]