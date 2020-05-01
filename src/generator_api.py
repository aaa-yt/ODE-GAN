import os
import json
from datetime import datetime
from logging import getLogger
import numpy as np

from config import Config

logger = getLogger(__name__)

def start(config: Config, n_data=1):
    return GeneratorAPI(config, n_data).start()

class GeneratorAPI:
    def __init__(self, config: Config, n_data):
        self.config = config
        self.model = None
        self.n_data = n_data
    
    def start(self):
        self.model = self.load_model()
        self.data = self.create_input_data()
        generated_point = self.model.generator(self.data)
        self.save_data_generate(generated_point)
    
    def load_model(self):
        from model import ODEGANModel
        model = ODEGANModel(self.config)
        model.load(self.config.resource.generator_model_path, self.config.resource.discriminator_model_path)
        return model
    
    def create_input_data(self):
        dim_out = self.config.model.dim_out
        x = []
        for i in range(dim_out):
            while len(x) < self.n_data * (i + 1) / dim_out:
                xx = np.zeros(shape=dim_out, dtype=np.float32)
                xx[i] = 1
                x.append(xx)
        noise = np.random.normal(loc=0., scale=0.5, size=(self.n_data, self.config.model.dim_noise)).astype(np.float32)
        return np.hstack([noise, x])

    def save_data_generate(self, generated_point):
        rc = self.config.resource
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_generate_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "data_generate.json")
        data_generate = {
            "Input": generated_point.tolist(),
            "Output": self.data[:,self.config.model.dim_noise:].tolist()
        }
        logger.debug("save generated data to {}".format(result_path))
        with open(result_path, "wt") as f:
            json.dump(data_generate, f, indent=4)