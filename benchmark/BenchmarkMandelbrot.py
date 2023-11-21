from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense
import keras
import tensorflow as tf
from keras.utils import to_categorical
from benchmark.KnowledgeDistillation import Distiller
import wandb
from bayes_opt import BayesianOptimization
from benchmark_config import student_config


# Fix for crashes due to cuda
if len(tf.config.list_physical_devices('GPU')) >= 1:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)


def login(key):
    wandb.login(key=key)


def closest_power_of_2(floating_point):
    if floating_point <= 0:
        raise ValueError("Input must be a positive number")

    closest_pow = 1
    while closest_pow * 2 <= floating_point:
        closest_pow *= 2

    return closest_pow


class HypParamOpt:
    def __init__(self, X, Y):
        self.X_train = None  # Training: 80%
        self.y_train = None
        self.X_val = None  # Validation: 10%
        self.y_val = None
        self.X_test = None  # Test: 10%
        self.y_test = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)
        self.y_train = to_categorical(self.y_train)
        self.y_test = to_categorical(self.y_test)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=42)

        self.project = None
        self.entity = None

    def create_model(self, temperature, alpha, batch_size, save_model=False, student_dict=None):



        hyperparams = {
            "temperature" : temperature,
            "alpha" : alpha,
            "batch_size" : batch_size,
        }

        run = wandb.init(project=self.project, entity=self.entity, config=hyperparams)

        #Converting dataframe to tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train)).batch(batch_size)

        model = Sequential()

        neuron_list = [16, 8, 4]

        for layer in range(len(neuron_list)):
            if layer == 0:
                model.add(Dense(neuron_list[layer], activation='relu', input_shape=(self.X_train.shape[1],)))
            else:
                model.add(Dense(neuron_list[layer], activation='relu'))

        model.add(Dense(2, activation='softmax'))

        try:
            teacher_model = keras.models.load_model("./teacher_models/mandelbrot_teacher")
        except:
            print("Error loading model, teacher model was trained?")
            return None

        distiller = Distiller(student=model, teacher=teacher_model)

        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.CategoricalAccuracy()],
            student_loss_fn=keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=alpha,
            temperature=temperature)


        #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0)


        if student_dict is not None:
            distiller.train(train_dataset, verbose=1, epochs=student_dict["Epoch"])
        else:
            distiller.train(train_dataset, verbose=1)

        #history = distiller.fit(self.X_train, self.y_train, epochs=20, batch_size=batch_size, validation_data=(self.X_val, self.y_val), verbose=0, callbacks=[callback])


        with tf.device('/CPU:0'):  # Inference a bit quicker on CPU on a small architecture (highly sequential)
            val_accuracy, val_loss = distiller.evaluate(self.X_val, self.y_val, verbose=0)

        run.finish()

        if save_model:
            distiller.save_weights("./student_models/mandelbrot_student_weights")

        return val_accuracy

    def start_model(self, temperature, alpha, batch_size):
        temperature = int(temperature)

        batch_size = closest_power_of_2(batch_size)

        return self.create_model(temperature, alpha, batch_size)

    def run(self, pbounds):
        key = input("Please enter your wandb API key (leave blank, if you already logged in):")

        if key == "":
            print("No key entered!")
        else:
            self.project = input("Please enter your project name")
            self.entity = input("Please enter you entity")
            login(key)

        #pbounds = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (1024, 20000), "same_topology": (0, 1)}

        user_select = input("Do you want to run the the student_config? (one model for each dataset will be trained)" +
                            " (y/n)")

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
            student_dict = student_config.get_mandel_dict()
            print("Training student model with config")
            temperature = 6
            alpha = 0.3
            batch_size = student_dict["batch_size"]
            self.create_model(temperature, alpha, batch_size, save_model=True, student_dict=student_dict)