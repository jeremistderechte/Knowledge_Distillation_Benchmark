# This has to be configured, because zhe implemented AutoML just searches for the best

# Mandelbrot
mandel_dict = {"Neuron": [16, 8, 4],  # change length of list and/or just neurons/units
               "Epoch": 2,  # Early stepping is activated
               "batch_size": 4096,  # the dataset is large, if you have enough memory this value can be this high
               "file_path": "./datasets/Mandelbrot.csv",  # change of your path is different
               "temperature": 3,  # hyperparam for knowledge distillation
               "alpha": 0.001}  # hyperparam for knowledge distillation

# Amazon dataset

amazon_dict = {"max_tokens": 5000,
               "embedding_dim": 256,
               "dropout_rate_embedding": 0.5,
               "learning_rate": 0.001,
               "layers": [[]],
               "batch_size": 512,
               "Epoch": 2,
               "file_path": "./datasets/office_rating.csv"}


def get_mandel_dict():
    return mandel_dict


def get_amazon_dict():
    return amazon_dict
