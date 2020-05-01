import os
import json
from logging import getLogger
import numpy as np

from config import Config

logger = getLogger(__name__)

def start(config: Config):
    return DataCreator(config).start()

class DataCreator:
    def __init__(self, config: Config):
        self.config = config
        self.dataset = None
    
    def start(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("Dataset already exists")
        else:
            self.dataset = self.create_dataset()
            self.save_dataset(data_path)
    
    def create_dataset(self):
        def get_data(n_data, sigma):
            x, y = [], []
            while len(x) < n_data / 3:
                x1 = np.random.normal(loc=0.5, scale=sigma)
                x2 = np.random.normal(loc=0.75, scale=sigma)
                x.append([x1, x2])
                y.append([1, 0, 0])
            while len(x) < 2 * n_data / 3:
                x1 = np.random.normal(loc=0.25, scale=sigma)
                x2 = np.random.normal(loc=0.25, scale=sigma)
                x.append([x1, x2])
                y.append([0, 1, 0])
            while len(x) < n_data:
                x1 = np.random.normal(loc=0.75, scale=sigma)
                x2 = np.random.normal(loc=0.25, scale=sigma)
                x.append([x1, x2])
                y.append([0, 0, 1])
            return (x, y)

        logger.info("Create a new dataset")
        n_data = 300
        sigma = 0.05
        data = get_data(n_data, sigma)
        dataset = {
            "Input": data[0],
            "Output": data[1]
        }
        return dataset
    
    def save_dataset(self, data_path):
        logger.debug("Save a new dataset")
        with open(data_path, "wt") as f:
            json.dump(self.dataset, f, indent=4)