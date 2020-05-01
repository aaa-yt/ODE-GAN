import os
import json
import time
import csv
from datetime import datetime
from logging import getLogger
from tqdm import tqdm
import numpy as np
from sklearn.utils import shuffle

from config import Config
from lib import optimizers
from lib.losses import Loss, accuracy
from visualize import Visualize

logger = getLogger(__name__)

def start(config: Config):
    return Trainer(config).start()

class Trainer:
    def __init__(self, config: Config):
        self.config = config
        self.model = None
        self.dataset = None
        if config.trainer.is_visualize: self.visualize = Visualize()
    
    def start(self):
        self.model = self.load_model()
        self.training()
    
    def training(self):
        tc = self.config.trainer
        self.compile_model()
        self.dataset = self.load_dataset()
        self.fit(self.dataset[0], self.dataset[1], epochs=tc.epoch, batch_size=tc.batch_size, is_visualize=tc.is_visualize)
        self.save_model()
        self.save_result()
    
    def compile_model(self):
        self.optimizer_generator = optimizers.get(self.config, "generator")
        self.optimizer_discriminator = optimizers.get(self.config, "discriminator")
        self.loss = Loss(self.config).get(self.config.trainer.loss_type)
        self.accuracy = accuracy
    
    def fit(self, x=None, y=None, epochs=1, batch_size=1, is_visualize=False):
        if x is None or y is None:
            raise ValueError("There is no fitting data")
        n_train = len(x)
        self.losses_generator = []
        self.losses_discriminator = []
        self.losses_generator_reg = []
        self.losses_discriminator_reg = []
        self.accuracies = []

        logger.info("training start")
        start_time = time.time()
        for epoch in range(epochs):
            x, y = shuffle(x, y)
            
            #Discriminatorの学習
            noise = np.random.normal(loc=0., scale=0.5, size=(n_train, self.config.model.dim_noise))
            x_fake = self.model.generator(np.hstack([noise, y]))
            y_fake = np.hstack([y, np.zeros(shape=(n_train, 1), dtype=np.float32)])
            y_real = np.hstack([y, np.ones(shape=(n_train, 1), dtype=np.float32)])
            x_train = np.concatenate([x, x_fake])
            y_train = np.concatenate([y_real, y_fake])
            x_train, y_train = shuffle(x_train, y_train)
            with tqdm(range(0, n_train*2, batch_size), desc="[Epoch: {}  Discriminator]".format(epoch+1)) as pbar:
                for i, ch in enumerate(pbar):
                    n_data = len(x_train[i:i+batch_size])
                    y_pred = self.model.discriminator(x_train[i:i+batch_size])
                    aT = np.zeros_like(x[i:i+batch_size])
                    bT = (y_train[i:i+batch_size] / y_pred) / n_data if self.config.trainer.loss_type == "crossentropy" else (y_pred - y_train[i:i+batch_size]) / n_data
                    self.model.discriminator.params = self.optimizer_discriminator(self.model.discriminator.params, self.model.discriminator.gradient(np.hstack([aT, bT])))
            y_pred = self.model.discriminator(x_train)
            error = self.loss(y_pred, y_train, self.model.discriminator.params)
            self.losses_discriminator.append(error[0])
            self.losses_discriminator_reg.append(error[1])
            accuracy = self.accuracy(y_pred[:, :-1], y_train[:, :-1])
            self.accuracies.append(accuracy)
            message1 = "Discriminator Epoch: {}  Loss: {}  Loss_reg: {}  accuracy: {}".format(epoch+1, error[0], error[1], accuracy)
            logger.info(message1)

            #Generatorの学習
            noise = np.random.normal(loc=0., scale=0.5, size=(n_train, self.config.model.dim_noise))
            x_train = np.hstack([noise, y])
            y_train = np.hstack([y, np.ones(shape=(n_train, 1), dtype=np.float32)])
            x_train, y_train = shuffle(x_train, y_train)
            with tqdm(range(0, n_train, batch_size), desc="[Epoch: {}  Generator]".format(epoch+1)) as pbar:
                for i, ch in enumerate(pbar):
                    n_data = len(x_train[i:i+batch_size])
                    y_pred = self.model(x_train[i:i+batch_size])
                    aT = np.zeros(shape=(n_data, self.model.discriminator.dim_in), dtype=np.float32)
                    bT = (y_train[i:i+batch_size] / y_pred) / n_data if self.config.trainer.loss_type == "crossentropy" else (y_pred - y_train[i:i+batch_size]) / n_data
                    self.model.discriminator.gradient(np.hstack([aT, bT]))
                    aT = np.zeros_like(x_train[i:i+batch_size])
                    bT = self.model.discriminator.a1[0]
                    self.model.generator.params = self.optimizer_generator(self.model.generator.params, self.model.generator.gradient(np.hstack([aT, bT])))
            y_pred = self.model(x_train)
            error = self.loss(y_pred, y_train, self.model.generator.params)
            self.losses_generator.append(error[0])
            self.losses_generator_reg.append(error[1])
            message2 = "Generator Epoch: {}  Loss: {}  Loss_reg: {}".format(epoch+1, error[0], error[1])
            logger.info(message2)
            
            if is_visualize:
                noise = np.random.normal(loc=0., scale=0.5, size=(n_train, self.config.model.dim_noise))
                generated_points = self.model.generator(np.hstack([noise, y]))
                self.visualize.plot_realtime([self.losses_generator, self.losses_generator_reg], [self.losses_discriminator, self.losses_discriminator_reg], self.accuracies, x, y, generated_points, y)
        
        interval = time.time() - start_time
        logger.info("end of training")
        logger.info("time: {}".format(interval))
        logger.info(message1)
        logger.info(message2)

    def load_model(self):
        from model import ODEGANModel
        model = ODEGANModel(self.config)
        model.load(self.config.resource.generator_model_path, self.config.resource.discriminator_model_path)
        return model
    
    def save_model(self):
        rc = self.config.resource
        model_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        model_dir = os.path.join(rc.model_dir, "model_{}".format(model_id))
        os.makedirs(model_dir, exist_ok=True)
        config_path = os.path.join(model_dir, "parameter.conf")
        generator_model_path = os.path.join(model_dir, "generator.json")
        discriminator_model_path = os.path.join(model_dir, "discriminator.json")
        self.model.save(config_path, generator_model_path, discriminator_model_path)
    
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
            return (np.array(x), np.array(y))
        else:
            raise FileNotFoundError("Dataset file can not loaded!")
    
    def save_result(self):
        rc = self.config.resource
        tc = self.config.trainer
        result_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_dir = os.path.join(rc.result_dir, "result_train_{}".format(result_id))
        os.makedirs(result_dir, exist_ok=True)
        result_path = os.path.join(result_dir, "learning_curve.csv")
        e = [i for i in range(1, tc.epoch+1)]
        self.visualize.save_plot_loss([self.losses_generator, self.losses_generator_reg], xlabel='Epoch', ylabel='Loss', title='Loss of generator', save_file=os.path.join(result_dir, "loss_generator.png"))
        self.visualize.save_plot_loss([self.losses_discriminator, self.losses_discriminator_reg], xlabel='Epoch', ylabel='Loss', title='Loss of discriminator', save_file=os.path.join(result_dir, "loss_discriminator.png"))
        self.visualize.save_plot_accuracy(self.accuracies, xlabel='Epoch', ylabel='Accuracy', title='Accuracy', save_file=os.path.join(result_dir, "accuracy.png"))
        result_csv = [e, self.losses_generator, self.losses_generator_reg, self.losses_discriminator, self.losses_discriminator_reg, self.accuracies]
        columns = ['epoch', 'loss_generator', 'loss_generator_reg', 'loss_discriminator', 'loss_discriminator_reg', 'accuracy']
        logger.debug("save result to {}".format(result_path))
        with open(result_path, "wt") as f:
            writer = csv.writer(f)
            writer.writerow(columns)
            writer.writerows(list(zip(*result_csv)))
        result_generator_dir = os.path.join(result_dir, "generator")
        result_discriminator_dir = os.path.join(result_dir, "discriminator")
        os.makedirs(result_generator_dir, exist_ok=True)
        os.makedirs(result_discriminator_dir, exist_ok=True)
        save_params_path = [os.path.join(result_generator_dir, "alpha_x.png"), os.path.join(result_generator_dir, "beta_x.png"), os.path.join(result_generator_dir, "gamma_x.png"), os.path.join(result_generator_dir, "A_x.png"), os.path.join(result_generator_dir, "alpha_y.png"), os.path.join(result_generator_dir, "beta_y.png"), os.path.join(result_generator_dir, "gamma_y.png"), os.path.join(result_generator_dir, "A_y.png"), os.path.join(result_generator_dir, "params.png")]
        self.visualize.save_plot_params(self.model.generator.t, self.model.generator.params, save_file=save_params_path)
        save_params_path = [os.path.join(result_discriminator_dir, "alpha_x.png"), os.path.join(result_discriminator_dir, "beta_x.png"), os.path.join(result_discriminator_dir, "gamma_x.png"), os.path.join(result_discriminator_dir, "A_x.png"), os.path.join(result_discriminator_dir, "alpha_y.png"), os.path.join(result_discriminator_dir, "beta_y.png"), os.path.join(result_discriminator_dir, "gamma_y.png"), os.path.join(result_discriminator_dir, "A_y.png"), os.path.join(result_discriminator_dir, "params.png")]
        self.visualize.save_plot_params(self.model.discriminator.t, self.model.discriminator.params, save_file=save_params_path)