import os
import json
from logging import getLogger
import numpy as np

from config import Config
from lib.helper import string_to_function, euler

logger = getLogger(__name__)

class ODEGANModel:
    def __init__(self, config: Config):
        self.config = config
        self.generator = NeuralODEModel(config.generator, config.trainer.regularization)
        self.discriminator = NeuralODEModel(config.discriminator, config.trainer.regularization)
    
    def __call__(self, x0):
        return self.discriminator(self.generator(x0))
    
    def load(self, generator_model_path, discriminator_model_path):
        self.generator.load(generator_model_path)
        self.discriminator.load(discriminator_model_path)
    
    def save(self, config_path, generator_model_path, discriminator_model_path):
        logger.debug("save model config to {}".format(config_path))
        self.config.save_parameter(config_path)
        self.generator.save(generator_model_path)
        self.discriminator.save(discriminator_model_path)


class NeuralODEModel:
    def __init__(self, config, regularization):
        self.config = config
        self.dim_in = config.dim_in
        self.dim_out = config.dim_out
        self.max_time = config.max_time
        self.division = config.division
        self.regularization = regularization
        alpha_x = np.random.uniform(low=-np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in)).astype(np.float32)
        beta_x = np.random.uniform(low=-np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in, self.dim_in)).astype(np.float32)
        gamma_x = np.random.uniform(low=-np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in)).astype(np.float32)
        A_x = np.random.uniform(low=-np.sqrt(3. / self.dim_in), high=np.sqrt(3. / self.dim_in), size=(self.division, self.dim_in, self.dim_out)).astype(np.float32)
        alpha_y = np.random.uniform(low=-np.sqrt(3. / self.dim_out), high=np.sqrt(3. / self.dim_out), size=(self.division, self.dim_out)).astype(np.float32)
        beta_y = np.random.uniform(low=-np.sqrt(3. / self.dim_out), high=np.sqrt(3. / self.dim_out), size=(self.division, self.dim_out, self.dim_out)).astype(np.float32)
        gamma_y = np.random.uniform(low=-np.sqrt(3. / self.dim_out), high=np.sqrt(3. / self.dim_out), size=(self.division, self.dim_out)).astype(np.float32)
        A_y = np.random.uniform(low=-np.sqrt(3. / self.dim_out), high=np.sqrt(3. / self.dim_out), size=(self.division, self.dim_out, self.dim_in)).astype(np.float32)
        self.params = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
        self.function_x, self.d_function_x = string_to_function(config.function_x_type)
        self.function_y, self.d_function_y = string_to_function(config.function_y_type)
        self.t = np.linspace(0., self.max_time, self.division)
    
    def __call__(self, x0):
        def func(x, t, params, function_x, function_y, dim_in, division):
            index = int(t * (division - 1))
            x1 = x[:, :dim_in]
            x2 = x[:, dim_in:]
            y1 = params[0][index] * function_x(np.dot(x1, params[1][index].T) + params[2][index] + np.dot(x2, params[3][index].T))
            y2 = params[4][index] * function_y(np.dot(x2, params[5][index].T) + params[6][index] + np.dot(x1, params[7][index].T))
            return np.hstack([y1, y2])
        
        y0 = np.zeros(shape=(len(x0), self.dim_out), dtype=np.float32)
        x = euler(func, np.hstack([x0, y0]), self.t, args=(self.params, self.function_x, self.function_y, self.dim_in, self.division))
        self.x = x[:, :, :self.dim_in]
        self.y = x[:, :, self.dim_in:]
        return x[-1, :, self.dim_in:]
    
    def gradient(self, aT):
        def func(a, t, params, function_x, function_y, x1, x2, dim_in, division):
            index = int(t * (division - 1))
            a1 = a[:, :dim_in]
            a2 = a[:, dim_in:]
            b1 = -np.dot(a1 * params[0][index] * function_x(np.dot(x1[index], params[1][index].T) + params[2][index] + np.dot(x2[index], params[3][index].T)), params[1][index]) - np.dot(a2 * params[4][index] * function_y(np.dot(x2[index], params[5][index].T) + params[6][index] + np.dot(x1[index], params[7][index].T)), params[7][index])
            b2 = -np.dot(a1 * params[0][index] * function_x(np.dot(x1[index], params[1][index].T) + params[2][index] + np.dot(x2[index], params[3][index].T)), params[3][index]) - np.dot(a2 * params[4][index] * function_y(np.dot(x2[index], params[5][index].T) + params[6][index] + np.dot(x1[index], params[7][index].T)), params[5][index])
            return np.hstack([b1, b2])
        
        a = euler(func, aT, self.t[::-1], args=(self.params, self.d_function_x, self.d_function_y, self.x, self.y, self.dim_in, self.division))
        self.a1 = a[::-1, :, :self.dim_in]
        self.a2 = a[::-1, :, self.dim_in:]
        x1 = np.einsum("ijk,ilk->ilj", self.params[1], self.x) + self.params[2].reshape(self.division, 1, self.dim_in) + np.einsum("ijk,ilk->ilj", self.params[3], self.y)
        x2 = np.einsum("ijk,ilk->ilj", self.params[5], self.y) + self.params[6].reshape(self.division, 1, self.dim_out) + np.einsum("ijk,ilk->ilj", self.params[7], self.x)
        g_alpha_x = np.sum(self.a1 * self.function_x(x1), 1) + self.regularization * self.params[0]
        g_beta_x = np.einsum("ilj,ilk->ijk", self.a1 * self.params[0].reshape(self.division, 1, self.dim_in) * self.d_function_x(x1), self.x) + self.regularization * self.params[1]
        g_gamma_x = np.sum(self.a1 * self.params[0].reshape(self.division, 1, self.dim_in) * self.d_function_x(x1), 1) + self.regularization * self.params[2]
        g_A_x = np.einsum("ilj,ilk->ijk", self.a1 * self.params[0].reshape(self.division, 1, self.dim_in) * self.d_function_x(x1), self.y) + self.regularization * self.params[3]
        g_alpha_y = np.sum(self.a2 * self.function_y(x2), 1) + self.regularization * self.params[4]
        g_beta_y = np.einsum("ilj,ilk->ijk", self.a2 * self.params[4].reshape(self.division, 1, self.dim_out) * self.d_function_y(x2), self.y) + self.regularization * self.params[5]
        g_gamma_y = np.sum(self.a2 * self.params[4].reshape(self.division, 1, self.dim_out) * self.d_function_y(x2), 1) + self.regularization * self.params[6]
        g_A_y = np.einsum("ilj,ilk->ijk", self.a2 * self.params[4].reshape(self.division, 1, self.dim_out) * self.d_function_y(x2), self.x) + self.regularization * self.params[7]
        return (g_alpha_x, g_beta_x, g_gamma_x, g_A_x, g_alpha_y, g_beta_y, g_gamma_y, g_A_y)
    
    def load(self, model_path):
        if os.path.exists(model_path):
            logger.debug("loding model from {}".format(model_path))
            with open(model_path, "rt") as f:
                model_weights = json.load(f)
            alpha_x = np.array(model_weights.get("Alpha_x"))
            beta_x = np.array(model_weights.get("Beta_x"))
            gamma_x = np.array(model_weights.get("Gamma_x"))
            A_x = np.array(model_weights.get("A_x"))
            alpha_y = np.array(model_weights.get("Alpha_y"))
            beta_y = np.array(model_weights.get("Beta_y"))
            gamma_y = np.array(model_weights.get("Gamma_y"))
            A_y = np.array(model_weights.get("A_y"))
            if self.params[0].shape != alpha_x.shape: alpha_x = self.params[0]
            if self.params[1].shape != beta_x.shape: beta_x = self.params[1]
            if self.params[2].shape != gamma_x.shape: gamma_x = self.params[2]
            if self.params[3].shape != A_x.shape: A_x = self.params[3]
            if self.params[4].shape != alpha_y.shape: alpha_y = self.params[4]
            if self.params[5].shape != beta_y.shape: beta_y = self.params[5]
            if self.params[6].shape != gamma_y.shape: gamma_y = self.params[6]
            if self.params[7].shape != A_y.shape: A_y = self.params[7]
            self.params = (alpha_x, beta_x, gamma_x, A_x, alpha_y, beta_y, gamma_y, A_y)
    
    def save(self, model_path):
        logger.debug("save model to {}".format(model_path))
        model_data = {
            "Alpha_x": self.params[0].tolist(),
            "Beta_x": self.params[1].tolist(),
            "Gamma_x": self.params[2].tolist(),
            "A_x": self.params[3].tolist(),
            "Alpha_y": self.params[4].tolist(),
            "Beta_y": self.params[5].tolist(),
            "Gamma_y": self.params[6].tolist(),
            "A_y": self.params[7].tolist()
        }
        with open(model_path, "wt") as f:
            json.dump(model_data, f, indent=4)