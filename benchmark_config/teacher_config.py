# This is the config file, if you do not want to run automl. It is recommended to use this config, because automl will
# take a long time and have a limited test range

# Mandelbrot
mandel_dict = {"Neuron": [4, 2, 1],  # change length of list and/or just neurons/units
               "Epoch": 2,  # Early stepping is activated
               "batch_size": 4096,  # the dataset is large, if you have enough memory this value should be this high
               "file_path": "./datasets/Mandelbrot.csv",  # change of your path is different
               "type": "normal"}

# Amazon dataset not functional yet

amazon_dict = {"max_tokens": 5000,
               "embedding_dim": 256,
               "dropout_rate_embedding": 0.5,
               "learning_rate": 0.001,
               "layers": [[]],
               "file_path": "./datasets/office_rating.csv",
               "type": "text"}


def get_mandel_dict():
    return mandel_dict


def get_amazon_dict():
    return amazon_dict
