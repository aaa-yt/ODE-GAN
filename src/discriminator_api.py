import os
import json
from datetime import datetime
from logging import getLogger
import numpy as np

from config import Config

logger = getLogger(__name__)

def start(config: Config):
    return DiscriminatorAPI(config).start()

class DiscriminatorAPI:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
    
    def start(self):
        self.model = self.load_model()
        self.dataset = self.load_dataset()
        y_pred = self.model.discriminator(self.dataset[0])
        self.save_data_predict(y_pred[:,:-1])
    
    def load_model(self):
        from model import ODEGANModel
        model = ODEGANModel(self.config)
        model.load(self.config.resource.generator_model_path, self.config.resource.discriminator_model_path)
        return model

    def load_dataset(self):
        data_path = self.config.resource.data_path
        if os.path.exists(data_path):
            logger.debug("loading data from {}".format(data_path))
            with open(data_path, "rt") as f:
                datasets = json.load(f)
            x = datasets.get("Input")
            y = datasets.get("Output")
            if x is None or y is None:
                raise TypeError("Dataset does not exists in {}".format(data_path))
            if len(x[0]) != self.config.model.dim_in:
                raise ValueError("Input dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_in, len(x[0])))
            if len(y[0]) != self.config.model.dim_out:
                raise ValueError("Output dimensions in config and dataset are not equal: {} != {}".format(self.config.model.dim_out, len(y[0])))
            return (np.array(x, dtype=np.float32), np.array(y,dtype=np.float32))
        else:
            raise FileNotFoundError("Dataset file can not loaded!")
    
    def save_data_predict(self, y_pred):
        rc = self.config.resource
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_predict_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "data_predict.json")
        data_predict = {
            "Input": self.dataset[0].tolist(),
            "Output": y_pred.tolist()
        }
        logger.debug("save prediction data to {}".format(result_path))
        with open(result_path, "wt") as f:
            json.dump(data_predict, f, indent=4)