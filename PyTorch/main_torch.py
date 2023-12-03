import torch
import os
import BayesianSearchFNN, BayesianSearchLSTM
from FNN import ReluNet
from LSTM import LSTMModel



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_students():


    # Change these values, for other hyperparameters (just values)
    # Bayesian search with given bounds
    pbounds_mandelbrot = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (4097, 10000)}
    pbounds_amazon = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (256, 1000)}

    bayesian_mandelbrot = BayesianSearchFNN.BayesianModelSearch()
    bayesian_mandelbrot.run(pbounds_mandelbrot)

    bayesian_amazon = BayesianSearchLSTM.BayesianModelSearch()
    bayesian_amazon.run(pbounds_amazon)


def main():
    print("Using torch backend!")

    mandelbrot_exists = False
    amazon_exists = False

    # Searching if teacher models are already trained
    try:
        os.listdir("./teacher_models/mandelbrot_teacher")
        mandelbrot_exists = True
        print("Teacher model for mandelbrot found, it will be used!")
    except FileNotFoundError:
        print("No Teacher model for mandelbrot found!")
    except:
        print("Unhandled exception while looking for teacher model")

    try:
        os.listdir("./teacher_models/amazon_teacher")
        amazon_exists = True
        print("Teacher model for amazon found, it will be used!")
    except FileNotFoundError:
        print("No Teacher model for amazon found!")
    except:
        print("Unhandled exception while looking for teacher model")

    # Handling if all or one/none teacher are trained
    if not mandelbrot_exists and not amazon_exists:
        import TeacherTorch
        TeacherTorch.train_teacher(mandelbrot=True, amazon=True)

    elif mandelbrot_exists and not amazon_exists:
        import TeacherTorch
        TeacherTorch.train_teacher(mandelbrot=False, amazon=True)
    elif not mandelbrot_exists and amazon_exists:
        import TeacherTorch
        TeacherTorch.train_teacher(mandelbrot=True, amazon=False)

    # All teachers should now be trained, bayesian search for best student NN started or best student will be trained
    train_students()



main()
