import os
import shutil
import json
import configparser
import subprocess
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

def create_config_file(config, config_path):
    config_parser = configparser.ConfigParser()
    config_parser["MODEL"] = {
        "Input_dimension": config["Input_dimension"],
        "Output_dimension": config["Output_dimension"],
        "Noise_dimension": config["Noise_dimension"],
        "Maximum_time": config["Maximum_time"],
        "Weights_division": config["Weights_division"],
        "Generator_function_x_type": config["Generator_function_x_type"],
        "Generator_function_y_type": config["Generator_function_y_type"],
        "discriminator_function_x_type": config["discriminator_function_x_type"],
        "discriminator_function_y_type": config["discriminator_function_y_type"]
    }
    config_parser["TRAINER"] = {
        "Optimizer_type": config["Optimizer_type"],
        "Loss_type": config["Loss_type"],
        "Learning_rate": config["Learning_rate"],
        "Momentum": config["Momentum"],
        "Decay": config["Decay"],
        "Decay2": config["Decay2"],
        "Regularization_rate": config["Regularization_rate"],
        "Epoch": config["Epoch"],
        "Batch_size": config["Batch_size"],
        "Is_visualize": config["Is_visualize"],
    }
    with open(config_path, "wt") as f:
        config_parser.write(f)

def create_data_file(config, data_path):
    def get_data(n_data, sigma):
        x, y = [], []
        while len(x) < n_data / 6:
            x1 = np.random.rand()
            x2 = 0.5 + np.sqrt(0.5*0.5-(x1-0.5)**2)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < n_data / 3:
            x1 = np.random.rand()
            x2 = 0.5 - np.sqrt(0.5*0.5-(x1-0.5)**2)
            x.append([x1, x2])
            y.append([1, 0, 0])
        while len(x) < 2 * n_data / 3:
            x1 = np.random.uniform(low=0.1, high=0.7)
            x2 = 0.4 + 1.6 * np.sqrt(0.3*0.3-(x1-0.4)**2)
            x.append([x1, x2])
            y.append([0, 1, 0])
        while len(x) < n_data:
            x1 = np.random.uniform(low=0.3, high=0.9)
            x2 = 0.6 - 1.6 * np.sqrt(0.3*0.3-(x1-0.6)**2)
            x.append([x1, x2])
            y.append([0, 0, 1])
        return (x, y)
    
    '''
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
    '''
    
    data = get_data(config["N_data_origin"], 0.07)
    dataset = {
        "Input": data[0],
        "Output": data[1]
    }
    with open(data_path, "wt") as f:
        json.dump(dataset, f, indent=4)

def setting_file(path):
    if not os.path.exists(path["Config_dir"]):
        os.makedirs(path["Config_dir"])
    if not os.path.exists(path["Data_dir"]):
        os.makedirs(path["Data_dir"])
    if not os.path.exists(path["Data_processed_dir"]):
        os.makedirs(path["Data_processed_dir"])
    shutil.copy(path["Ex_config_path"], path["Config_path"])
    shutil.copy(path["Ex_data_path"], path["Data_path"])

def setting_model(path):
    model_directries = [d for d in os.listdir(path["Model_dir"]) if os.path.isdir(os.path.join(path["Model_dir"], d))]
    model_dir_old = os.path.join(path["Model_dir"], model_directries[0])
    shutil.copy(os.path.join(model_dir_old, "generator.json"), path["Generator_model_path"])
    shutil.copy(os.path.join(model_dir_old, "discriminator.json"), path["Discriminator_model_path"])
    
def setting_generate_predict(path):
    os.remove(path["Data_path"])
    for p in os.listdir(path["Result_dir"]):
        if "generate" in p:
            data_generate_path = os.path.join(os.path.join(path["Result_dir"], p), "data_generate.json")
            shutil.copy(data_generate_path, path["Data_path"])

def copy_result(path):
    if not os.path.exists(path["Ex_result_dir"]):
        os.makedirs(path["Ex_result_dir"])
    result_predict_dirs = {}
    for p in os.listdir(path["Result_dir"]):
        result_dir = os.path.join(path["Result_dir"], p)
        for pp in os.listdir(result_dir):
            result_path = os.path.join(result_dir, pp)
            if pp == "data_predict.json":
                result_predict_dirs[datetime.strptime(p.strip("result_predict_"), "%Y%m%d-%H%M%S")] = result_dir
                continue
            shutil.move(result_path, path["Ex_result_dir"])
    new_result_path = os.path.join(result_predict_dirs[max(result_predict_dirs)], "data_generate_predict.json")
    shutil.move(os.path.join(result_predict_dirs[max(result_predict_dirs)], "data_predict.json"), new_result_path)
    shutil.move(new_result_path, path["Ex_result_dir"])
    shutil.move(os.path.join(result_predict_dirs[min(result_predict_dirs)], "data_predict.json"), path["Ex_result_dir"])
    shutil.move(path["Generator_model_path"], path["Ex_dir"])
    shutil.move(path["Discriminator_model_path"], path["Ex_dir"])

def plot(path):
    dataset_origin = load_dataset(path["Ex_data_path"], path["Data_predict_path"])
    dataset_generated = load_dataset(path["Data_generate_path"], path["Data_generate_predict_path"])
    fig = plt.figure()
    ax = fig.add_subplot(111)

    fig_origin = plt.figure()
    ax_origin = fig_origin.add_subplot(111)

    fig_origin_predict = plt.figure()
    ax_origin_predict = fig_origin_predict.add_subplot(111)

    fig_generated = plt.figure()
    ax_generated = fig_generated.add_subplot(111)

    fig_generated_predict = plt.figure()
    ax_generated_predict = fig_generated_predict.add_subplot(111)

    fig_origin_generated = plt.figure()
    ax_origin_generated = fig_origin_generated.add_subplot(111)

    fig_predict = plt.figure()
    ax_predict = fig_predict.add_subplot(111)

    x_ori = np.array(dataset_origin[0])
    y_ori_true = np.argmax(dataset_origin[1], 1)
    y_ori_pred = np.argmax(dataset_origin[2], 1)
    x_gen = np.array(dataset_generated[0])
    y_gen_true = np.argmax(dataset_generated[1], 1)
    y_gen_pred = np.argmax(dataset_generated[2], 1)

    ax.scatter(x_ori[np.where((y_ori_pred==1) & (y_ori_true==0))[0]][:, 0], x_ori[np.where((y_ori_pred==1) & (y_ori_true==0))[0]][:, 1], s=10, c='#ffbb00', label=r'$F(\xi)=1,y(T;\xi)=2$')
    ax.scatter(x_ori[np.where((y_ori_pred==2) & (y_ori_true==0))[0]][:, 0], x_ori[np.where((y_ori_pred==2) & (y_ori_true==0))[0]][:, 1], s=10, c='#ff00bb', label=r'$F(\xi)=1,y(T;\xi)=3$')
    ax.scatter(x_ori[np.where((y_ori_pred==0) & (y_ori_true==1))[0]][:, 0], x_ori[np.where((y_ori_pred==0) & (y_ori_true==1))[0]][:, 1], s=10, c='#bbff00', label=r'$F(\xi)=2,y(T;\xi)=1$')
    ax.scatter(x_ori[np.where((y_ori_pred==2) & (y_ori_true==1))[0]][:, 0], x_ori[np.where((y_ori_pred==2) & (y_ori_true==1))[0]][:, 1], s=10, c='#00ffbb', label=r'$F(\xi)=2,y(T;\xi)=3$')
    ax.scatter(x_ori[np.where((y_ori_pred==0) & (y_ori_true==2))[0]][:, 0], x_ori[np.where((y_ori_pred==0) & (y_ori_true==2))[0]][:, 1], s=10, c='#bb00ff', label=r'$F(\xi)=3,y(T;\xi)=1$')
    ax.scatter(x_ori[np.where((y_ori_pred==1) & (y_ori_true==2))[0]][:, 0], x_ori[np.where((y_ori_pred==1) & (y_ori_true==2))[0]][:, 1], s=10, c='#00bbff', label=r'$F(\xi)=3,y(T;\xi)=2$')
    ax.scatter(x_ori[np.where((y_ori_pred==0) & (y_ori_true==0))[0]][:, 0], x_ori[np.where((y_ori_pred==0) & (y_ori_true==0))[0]][:, 1], s=10, c='#ff0000', label=r'$F(\xi)=1,y(T;\xi)=1$')
    ax.scatter(x_ori[np.where((y_ori_pred==1) & (y_ori_true==1))[0]][:, 0], x_ori[np.where((y_ori_pred==1) & (y_ori_true==1))[0]][:, 1], s=10, c='#00ff00', label=r'$F(\xi)=2,y(T;\xi)=2$')
    ax.scatter(x_ori[np.where((y_ori_pred==2) & (y_ori_true==2))[0]][:, 0], x_ori[np.where((y_ori_pred==2) & (y_ori_true==2))[0]][:, 1], s=10, c='#0000ff', label=r'$F(\xi)=3,y(T;\xi)=3$')
    ax.scatter(x_gen[np.where((y_gen_pred==1) & (y_gen_true==0))[0]][:, 0], x_gen[np.where((y_gen_pred==1) & (y_gen_true==0))[0]][:, 1], s=10, c='#ffbb00', label=r'$F(\xi)=1,y(T;\xi)=2$')
    ax.scatter(x_gen[np.where((y_gen_pred==2) & (y_gen_true==0))[0]][:, 0], x_gen[np.where((y_gen_pred==2) & (y_gen_true==0))[0]][:, 1], s=10, c='#ff00bb', label=r'$F(\xi)=1,y(T;\xi)=3$')
    ax.scatter(x_gen[np.where((y_gen_pred==0) & (y_gen_true==1))[0]][:, 0], x_gen[np.where((y_gen_pred==0) & (y_gen_true==1))[0]][:, 1], s=10, c='#bbff00', label=r'$F(\xi)=2,y(T;\xi)=1$')
    ax.scatter(x_gen[np.where((y_gen_pred==2) & (y_gen_true==1))[0]][:, 0], x_gen[np.where((y_gen_pred==2) & (y_gen_true==1))[0]][:, 1], s=10, c='#00ffbb', label=r'$F(\xi)=2,y(T;\xi)=3$')
    ax.scatter(x_gen[np.where((y_gen_pred==0) & (y_gen_true==2))[0]][:, 0], x_gen[np.where((y_gen_pred==0) & (y_gen_true==2))[0]][:, 1], s=10, c='#bb00ff', label=r'$F(\xi)=3,y(T;\xi)=1$')
    ax.scatter(x_gen[np.where((y_gen_pred==1) & (y_gen_true==2))[0]][:, 0], x_gen[np.where((y_gen_pred==1) & (y_gen_true==2))[0]][:, 1], s=10, c='#00bbff', label=r'$F(\xi)=3,y(T;\xi)=2$')
    ax.scatter(x_gen[np.where((y_gen_pred==0) & (y_gen_true==0))[0]][:, 0], x_gen[np.where((y_gen_pred==0) & (y_gen_true==0))[0]][:, 1], s=10, c='#ff0000', label=r'$F(\xi)=1,y(T;\xi)=1$')
    ax.scatter(x_gen[np.where((y_gen_pred==1) & (y_gen_true==1))[0]][:, 0], x_gen[np.where((y_gen_pred==1) & (y_gen_true==1))[0]][:, 1], s=10, c='#00ff00', label=r'$F(\xi)=2,y(T;\xi)=2$')
    ax.scatter(x_gen[np.where((y_gen_pred==2) & (y_gen_true==2))[0]][:, 0], x_gen[np.where((y_gen_pred==2) & (y_gen_true==2))[0]][:, 1], s=10, c='#0000ff', label=r'$F(\xi)=3,y(T;\xi)=3$')

    ax_origin.scatter(x_ori[np.where(y_ori_true==0)[0]][:,0], x_ori[np.where(y_ori_true==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_origin.scatter(x_ori[np.where(y_ori_true==1)[0]][:,0], x_ori[np.where(y_ori_true==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_origin.scatter(x_ori[np.where(y_ori_true==2)[0]][:,0], x_ori[np.where(y_ori_true==2)[0]][:,1], s=10, c='#0000ff', label='3')

    ax_origin_predict.scatter(x_ori[np.where(y_ori_pred==0)[0]][:,0], x_ori[np.where(y_ori_pred==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_origin_predict.scatter(x_ori[np.where(y_ori_pred==1)[0]][:,0], x_ori[np.where(y_ori_pred==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_origin_predict.scatter(x_ori[np.where(y_ori_pred==2)[0]][:,0], x_ori[np.where(y_ori_pred==2)[0]][:,1], s=10, c='#0000ff', label='3')

    ax_generated.scatter(x_gen[np.where(y_gen_true==0)[0]][:,0], x_gen[np.where(y_gen_true==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_generated.scatter(x_gen[np.where(y_gen_true==1)[0]][:,0], x_gen[np.where(y_gen_true==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_generated.scatter(x_gen[np.where(y_gen_true==2)[0]][:,0], x_gen[np.where(y_gen_true==2)[0]][:,1], s=10, c='#0000ff', label='3')

    ax_generated_predict.scatter(x_gen[np.where(y_gen_pred==0)[0]][:,0], x_gen[np.where(y_gen_pred==0)[0]][:,1], s=10, c='#ff0000', label='1')
    ax_generated_predict.scatter(x_gen[np.where(y_gen_pred==1)[0]][:,0], x_gen[np.where(y_gen_pred==1)[0]][:,1], s=10, c='#00ff00', label='2')
    ax_generated_predict.scatter(x_gen[np.where(y_gen_pred==2)[0]][:,0], x_gen[np.where(y_gen_pred==2)[0]][:,1], s=10, c='#0000ff', label='3')

    ax_origin_generated.scatter(x_ori[np.where(y_ori_true==0)[0]][:,0], x_ori[np.where(y_ori_true==0)[0]][:,1], s=10, c='#ff0000', label='original 1')
    ax_origin_generated.scatter(x_ori[np.where(y_ori_true==1)[0]][:,0], x_ori[np.where(y_ori_true==1)[0]][:,1], s=10, c='#00ff00', label='original 2')
    ax_origin_generated.scatter(x_ori[np.where(y_ori_true==2)[0]][:,0], x_ori[np.where(y_ori_true==2)[0]][:,1], s=10, c='#0000ff', label='original 3')
    ax_origin_generated.scatter(x_gen[np.where(y_gen_true==0)[0]][:,0], x_gen[np.where(y_gen_true==0)[0]][:,1], s=10, c='#ff6666', label='generated 1')
    ax_origin_generated.scatter(x_gen[np.where(y_gen_true==1)[0]][:,0], x_gen[np.where(y_gen_true==1)[0]][:,1], s=10, c='#66ff66', label='generated 2')
    ax_origin_generated.scatter(x_gen[np.where(y_gen_true==2)[0]][:,0], x_gen[np.where(y_gen_true==2)[0]][:,1], s=10, c='#6666ff', label='generated 3')

    ax_predict.scatter(x_ori[np.where(y_ori_pred==0)[0]][:,0], x_ori[np.where(y_ori_pred==0)[0]][:,1], s=10, c='#ff0000', label='original 1')
    ax_predict.scatter(x_ori[np.where(y_ori_pred==1)[0]][:,0], x_ori[np.where(y_ori_pred==1)[0]][:,1], s=10, c='#00ff00', label='original 2')
    ax_predict.scatter(x_ori[np.where(y_ori_pred==2)[0]][:,0], x_ori[np.where(y_ori_pred==2)[0]][:,1], s=10, c='#0000ff', label='original 3')
    ax_predict.scatter(x_gen[np.where(y_gen_pred==0)[0]][:,0], x_gen[np.where(y_gen_pred==0)[0]][:,1], s=10, c='#ff6666', label='generated 1')
    ax_predict.scatter(x_gen[np.where(y_gen_pred==1)[0]][:,0], x_gen[np.where(y_gen_pred==1)[0]][:,1], s=10, c='#66ff66', label='generated 2')
    ax_predict.scatter(x_gen[np.where(y_gen_pred==2)[0]][:,0], x_gen[np.where(y_gen_pred==2)[0]][:,1], s=10, c='#6666ff', label='generated 3')

    ax.set_xlabel(r'$\xi_1$')
    ax.set_ylabel(r'$\xi_2$')
    ax.set_aspect('equal')
    ax_origin.set_xlabel(r'$\xi_1$')
    ax_origin.set_ylabel(r'$\xi_2$')
    ax_origin.set_title("Original data")
    ax_origin.set_aspect('equal')
    ax_origin.legend()
    ax_origin_predict.set_xlabel(r'$\xi_1$')
    ax_origin_predict.set_ylabel(r'$\xi_2$')
    ax_origin_predict.set_title("Original data prediction")
    ax_origin_predict.set_aspect('equal')
    ax_origin_predict.legend()
    ax_generated.set_xlabel(r'$\xi_1$')
    ax_generated.set_ylabel(r'$\xi_2$')
    ax_generated.set_title("Generated data")
    ax_generated.set_aspect('equal')
    ax_generated.legend()
    ax_generated_predict.set_xlabel(r'$\xi_1$')
    ax_generated_predict.set_ylabel(r'$\xi_2$')
    ax_generated_predict.set_title("Generated data prediction")
    ax_generated_predict.set_aspect('equal')
    ax_generated_predict.legend()
    ax_origin_generated.set_xlabel(r'$\xi_1$')
    ax_origin_generated.set_ylabel(r'$\xi_2$')
    ax_origin_generated.set_title("Original and Generated data")
    ax_origin_generated.set_aspect('equal')
    ax_origin_generated.legend()
    ax_predict.set_xlabel(r'$\xi_1$')
    ax_predict.set_ylabel(r'$\xi_2$')
    ax_predict.set_title("Original and Generated data prediction")
    ax_predict.set_aspect('equal')
    ax_predict.legend()

    lgnd = ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.15), ncol=2)

    fig.savefig(os.path.join(path["Ex_result_dir"], "data.png"), bbox_extra_artists=(lgnd,), bbox_inches='tight')
    fig_origin.savefig(os.path.join(path["Ex_result_dir"], "data_origin.png"))
    fig_origin_predict.savefig(os.path.join(path["Ex_result_dir"], "data_origin_predict.png"))
    fig_generated.savefig(os.path.join(path["Ex_result_dir"], "data_generated.png"))
    fig_generated_predict.savefig(os.path.join(path["Ex_result_dir"], "data_generated_predict.png"))
    fig_origin_generated.savefig(os.path.join(path["Ex_result_dir"], "data_origin_generated.png"))
    fig_predict.savefig(os.path.join(path["Ex_result_dir"], "data_predict.png"))

def load_dataset(data_path, data_predict_path):
    with open(data_path, "rt") as f:
        data = json.load(f)
    with open(data_predict_path, "rt") as f:
        data_predict = json.load(f)
    x = data.get("Input")
    y = data.get("Output")
    y_pred = data_predict.get("Output")
    return (x, y, y_pred)
    
def clear(path):
    shutil.rmtree(path["Data_dir"])
    shutil.rmtree(path["Config_dir"])
    shutil.rmtree(path["Model_dir"])

if __name__ == "__main__":
    config = {
        "Input_dimension": 2,
        "Output_dimension": 3,
        "Noise_dimension": 7,
        "Maximum_time": 1.,
        "Weights_division": 100,
        "Generator_function_x_type": "relu",
        "Generator_function_y_type": "relu",
        "discriminator_function_x_type": "relu",
        "discriminator_function_y_type": "sigmoid",
        "Optimizer_type": "RMSprop",
        "Loss_type": "MSE",
        "Learning_rate": 0.01,
        "Momentum": 0.9,
        "Decay": 0.9,
        "Decay2": 0.999,
        "Regularization_rate": 0.0001,
        "Epoch": 5000,
        "Batch_size": 10,
        "Is_visualize": 1,
        "N_data_origin": 800,
        "N_data_generate": 5200,
    }

    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    data_dir = os.path.join(project_dir, "data")
    data_processed_dir = os.path.join(data_dir, "processed")
    data_path = os.path.join(data_processed_dir, "data.json")
    config_dir = os.path.join(project_dir, "config")
    config_path = os.path.join(config_dir, "parameter.conf")
    model_dir = os.path.join(project_dir, "model")
    generator_model_path = os.path.join(model_dir, "generator.json")
    discriminator_model_path = os.path.join(model_dir, "discriminator.json")
    result_dir = os.path.join(data_dir, "result")
    program_path = os.path.join(os.path.join(project_dir, "src"), "run.py")
    test_dir = os.path.join(project_dir, "test")
    ex_dir = os.path.join(os.path.join(test_dir, "experiments"), "ex-multi-classification")
    ex_data_path = os.path.join(ex_dir, "data.json")
    ex_config_path = os.path.join(ex_dir, "parameter.conf")
    ex_generator_model_path = os.path.join(ex_dir, "generator.json")
    ex_discriminator_model_path = os.path.join(ex_dir, "discriminator.json")
    ex_result_dir = os.path.join(ex_dir, "result")
    data_generate_path = os.path.join(ex_result_dir, "data_generate.json")
    data_predict_path = os.path.join(ex_result_dir, "data_predict.json")
    data_generate_predict_path = os.path.join(ex_result_dir, "data_generate_predict.json")

    path = {
        "Project_dir": project_dir,
        "Data_dir": data_dir,
        "Data_processed_dir": data_processed_dir,
        "Data_path": data_path,
        "Config_dir": config_dir,
        "Config_path": config_path,
        "Model_dir": model_dir,
        "Generator_model_path": generator_model_path,
        "Discriminator_model_path": discriminator_model_path,
        "Result_dir": result_dir,
        "Program_path": program_path,
        "Test_dir": test_dir,
        "Ex_dir": ex_dir,
        "Ex_data_path": ex_data_path,
        "Ex_config_path": ex_config_path,
        "Ex_generator_model_path": ex_generator_model_path,
        "Ex_discriminator_model_path": ex_discriminator_model_path,
        "Ex_result_dir": ex_result_dir,
        "Data_generate_path": data_generate_path,
        "Data_predict_path": data_predict_path,
        "Data_generate_predict_path": data_generate_predict_path
    }
    
    create_config_file(config, ex_config_path)
    create_data_file(config, ex_data_path)
    setting_file(path)
    subprocess.call(["python", program_path, "train"])
    setting_model(path)
    subprocess.call(["python", program_path, "predict"])
    subprocess.call(["python", program_path, "generate", "--number", str(config["N_data_generate"])])
    setting_generate_predict(path)
    subprocess.call(["python", program_path, "predict"])
    copy_result(path)
    
    plot(path)
    clear(path)