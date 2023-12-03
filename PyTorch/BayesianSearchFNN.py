import torch
import wandb
from FNN_Distilled import ReluKnowledgeDistilled, train_model, test_model
from FNN import ReluNet
from bayes_opt import BayesianOptimization
from benchmark_config.student_config import get_mandel_dict
from BatchedDataHandler import get_data



device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

def login(key):
    wandb.login(key=key)


def closest_power_of_2(floating_point):
    if floating_point <= 0:
        raise ValueError("Input must be a positive number")

    closest_pow = 1
    while closest_pow * 2 <= floating_point:
        closest_pow *= 2

    return closest_pow


class BayesianModelSearch:
    def __init__(self):
        self.train_loader = None
        self.test_loader = None
        self.val_loader = None

        self.project = None
        self.entity = None

        self.teacher_model = None

    def create_model(self, temperature, alpha, batch_size, save_model=False, student_dict=None):

        hyperparams = {
            "temperature": temperature,
            "alpha": alpha,
            "batch_size": batch_size,
        }

        run = wandb.init(project=self.project, entity=self.entity, config=hyperparams)

        student_model = ReluKnowledgeDistilled().to(device)

        optimizer = torch.optim.Adam(student_model.parameters())
        loss_fn = torch.nn.CrossEntropyLoss()

        train_loader, test_loader, val_loader = get_data("../datasets/Mandelbrot.csv", ["X", "Y"],
                                                         ["divergend"], batch_size)

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader

        train_model(self.train_loader, self.val_loader, student_model, self.teacher_model,
                    optimizer, loss_fn, temperature, alpha)

        metrics = test_model(self.test_loader, student_model, loss_fn, wandb_logging=False)

        test_accuracy = metrics["Accuracy"]

        run.finish()

        if save_model:
            torch.save(student_model, "./student_models/student_mandelbrot.pth")

        return test_accuracy

    def start_model(self, temperature, alpha, batch_size):
        temperature = int(temperature)

        batch_size = closest_power_of_2(batch_size)

        return self.create_model(temperature, alpha, batch_size)

    def run(self, pbounds):
        try:
            if device == "cpu":
                # Fix, if one want to run it on CPU
                map_location = torch.device('cpu')
                self.teacher_model = torch.load('./teacher_models/mandelbrot_teacher/teacher_mandelbrot.pth',
                                                map_location=map_location)
            else:
                self.teacher_model = torch.load('./teacher_models/mandelbrot_teacher/teacher_mandelbrot.pth')
        except:
            print("Error loading model, teacher model was trained?")
            return 1

        key = input("Please enter your wandb API key (leave blank, if you already logged in): ")

        if key == "":
            print("No key entered!")
        else:
            login(key)

        self.project = input("Please enter your project name: ")
        self.entity = input("Please enter you entity: ")

        user_select = input("Do you want to run the the student_config? (one model for each dataset will be trained)" +
                            " (y/n): ")

        if not user_select.lower() == "y":
            init_points = int(input("How many init_points do you want: "))
            n_iter = int(input("How many iterations do you want: "))
            optimizer = BayesianOptimization(
                f=self.start_model,
                pbounds=pbounds,
                random_state=1,
            )

            optimizer.maximize(
                init_points=init_points,
                n_iter=n_iter,
            )
        else:
            student_dict = get_mandel_dict()
            print("Training student model with config")
            temperature = student_dict["temperature"]
            alpha = student_dict["alpha"]
            batch_size = student_dict["batch_size"]
            self.create_model(temperature, alpha, batch_size, save_model=True, student_dict=student_dict)
