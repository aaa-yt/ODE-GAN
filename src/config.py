import os
import configparser
from logging import getLogger

logger = getLogger(__name__)

def _project_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _data_dir():
    return os.path.join(_project_dir(), "data")

def _model_dir():
    return os.path.join(_project_dir(), "model")

class Config:
    def __init__(self):
        self.resource = ResourceConfig()
        self.model = ModelConfig()
        self.generator = GeneratorConfig(self.model)
        self.discriminator = DiscriminatorConfig(self.model)
        self.trainer = TrainerConfig()
    
    def load_parameter(self):
        if os.path.exists(self.resource.config_path):
            logger.debug("loading parameter from {}".format(self.resource.config_path))
            config_parser = configparser.ConfigParser()
            config_parser.read(self.resource.config_path, encoding='utf-8')
            read_model = config_parser['MODEL']
            if read_model.get("Input_dimension") is not None: self.model.dim_in = int(read_model.get("Input_dimension"))
            if read_model.get("Output_dimension") is not None: self.model.dim_out = int(read_model.get("Output_dimension"))
            if read_model.get("Noise_dimension") is not None: self.model.dim_noise = int(read_model.get("Noise_dimension"))
            if read_model.get("Maximum_time") is not None: self.model.max_time = float(read_model.get("Maximum_time"))
            if read_model.get("Weights_division") is not None: self.model.division = int(read_model.get("Weights_division"))
            if read_model.get("Generator_function_x_type") is not None: self.model.generator_function_x_type = read_model.get("Generator_function_x_type")
            if read_model.get("Generator_function_y_type") is not None: self.model.generator_function_y_type = read_model.get("Generator_function_y_type")
            if read_model.get("discriminator_function_x_type") is not None: self.model.discriminator_function_x_type = read_model.get("discriminator_function_x_type")
            if read_model.get("discriminator_function_y_type") is not None: self.model.discriminator_function_y_type = read_model.get("discriminator_function_y_type")
            read_trainer = config_parser['TRAINER']
            if read_trainer.get("Optimizer_type") is not None: self.trainer.optimizer_type = read_trainer.get("Optimizer_type")
            if read_trainer.get("Loss_type") is not None: self.trainer.loss_type = read_trainer.get("Loss_type")
            if read_trainer.get("Learning_rate") is not None: self.trainer.rate = float(read_trainer.get("Learning_rate"))
            if read_trainer.get("Momentum") is not None: self.trainer.momentum = float(read_trainer.get("Momentum"))
            if read_trainer.get("Decay") is not None: self.trainer.decay = float(read_trainer.get("Decay"))
            if read_trainer.get("Decay2") is not None: self.trainer.decay2 = float(read_trainer.get("Decay2"))
            if read_trainer.get("Regularization_rate") is not None: self.trainer.regularization = float(read_trainer.get("Regularization_rate"))
            if read_trainer.get("Epoch") is not None: self.trainer.epoch = int(read_trainer.get("Epoch"))
            if read_trainer.get("Batch_size") is not None: self.trainer.batch_size = int(read_trainer.get("Batch_size"))
            if read_trainer.get("Is_visualize") is not None: self.trainer.is_visualize = bool(int(read_trainer.get("Is_visualize")))
            self.generator = GeneratorConfig(self.model)
            self.discriminator = DiscriminatorConfig(self.model)
    
    def save_parameter(self, config_path):
        config_parser = configparser.ConfigParser()
        config_parser["MODEL"] = {
            "Input_dimension": self.model.dim_in,
            "Output_dimension": self.model.dim_out,
            "Noise_dimension": self.model.dim_noise,
            "Maximum_time": self.model.max_time,
            "Weights_division": self.model.division,
            "Generator_function_x_type": self.model.generator_function_x_type,
            "Generator_function_y_type": self.model.generator_function_y_type,
            "discriminator_function_x_type": self.model.discriminator_function_x_type,
            "discriminator_function_y_type": self.model.discriminator_function_y_type
        }
        config_parser["TRAINER"] = {
            "Optimizer_type": self.trainer.optimizer_type,
            "Loss_type": self.trainer.loss_type,
            "Learning_rate": self.trainer.rate,
            "Momentum": self.trainer.momentum,
            "Decay": self.trainer.decay,
            "Decay2": self.trainer.decay2,
            "Regularization_rate": self.trainer.regularization,
            "Epoch": self.trainer.epoch,
            "Batch_size": self.trainer.batch_size,
            "Is_visualize": int(self.trainer.is_visualize)
        }
        with open(config_path, "wt") as f:
            config_parser.write(f)


class ResourceConfig:
    def __init__(self):
        self.project_dir = os.environ.get("PROJECT_DIR", _project_dir())
        self.data_dir = os.environ.get("DATA_DIR", _data_dir())
        self.data_processed_dir = os.path.join(self.data_dir, "processed")
        self.data_path = os.path.join(self.data_processed_dir, "data.json")
        self.model_dir = os.environ.get("MODEL_DIR", _model_dir())
        self.generator_model_path = os.path.join(self.model_dir, "generator.json")
        self.discriminator_model_path = os.path.join(self.model_dir, "discriminator.json")
        self.result_dir = os.path.join(self.data_dir, "result")
        self.log_dir = os.path.join(self.project_dir, "logs")
        self.main_log_path = os.path.join(self.log_dir, "main.log")
        self.config_dir = os.path.join(self.project_dir, "config")
        self.config_path = os.path.join(self.config_dir, "parameter.conf")
    
    def create_directories(self):
        dirs = [self.project_dir, self.data_dir, self.data_processed_dir, self.model_dir, self.result_dir, self.log_dir, self.config_dir]
        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)


class ModelConfig:
    def __init__(self):
        self.dim_in = 1
        self.dim_out = 1
        self.dim_noise = 1
        self.max_time = 1.
        self.division = 100
        self.generator_function_x_type = "relu"
        self.generator_function_y_type = "relu"
        self.discriminator_function_x_type = "relu"
        self.discriminator_function_y_type = "sigmoid"


class GeneratorConfig:
    def __init__(self, config: ModelConfig):
        self.dim_in = config.dim_out + config.dim_noise
        self.dim_out = config.dim_in
        self.max_time = config.max_time
        self.division = config.division
        self.function_x_type = config.generator_function_x_type
        self.function_y_type = config.generator_function_y_type


class DiscriminatorConfig:
    def __init__(self, config: ModelConfig):
        self.dim_in = config.dim_in
        self.dim_out = config.dim_out + 1
        self.max_time = config.max_time
        self.division = config.division
        self.function_x_type = config.discriminator_function_x_type
        self.function_y_type = config.discriminator_function_y_type


class TrainerConfig:
    def __init__(self):
        self.optimizer_type = "RMSprop"
        self.loss_type = "MSE"
        self.rate = 0.01
        self.momentum = 0.9
        self.decay = 0.9
        self.decay2 = 0.999
        self.regularization = 0.0001
        self.epoch = 5
        self.batch_size = 10
        self.is_visualize = True