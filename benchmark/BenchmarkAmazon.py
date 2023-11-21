from sklearn.model_selection import train_test_split
import keras
import tensorflow as tf
from keras.utils import to_categorical
from benchmark.KnowledgeDistillation import Distiller
from bayes_opt import BayesianOptimization
import keras.layers as layers
from keras.optimizers import Adam
from benchmark_config import student_config
from keras.preprocessing.sequence import pad_sequences
import wandb



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

        Y[Y < 3] = 0
        Y[Y == 3] = 1
        Y[Y > 3] = 2

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

        input_length = 100  # Example length

        # Model configuration
        max_tokens = 5000
        embedding_dim = 128
        dropout_rate_embedding = 0.5
        learning_rate = 0.001

        # Model construction
        input_layer = layers.Input(shape=(input_length,), dtype="int32")
        x = layers.Embedding(max_tokens, embedding_dim, input_length=input_length)(input_layer)
        x = layers.Dropout(dropout_rate_embedding)(x)

        # Convolutional block 1
        x = layers.Conv1D(32, 3, activation='relu')(x)
        x = layers.Conv1D(16, 3, activation='relu')(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Flatten()(x)

        # Dense layers
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dense(32, activation='relu')(x)

        output_layer = layers.Dense(3, activation='softmax')(x)  # Assuming binary classification

        # Compile the model
        student = keras.Model(inputs=input_layer, outputs=output_layer)
        student.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                        metrics=['accuracy'])

        # Model summary
        student.summary()

        run = wandb.init(project="test21", entity="jeremy-barenkamp", config=hyperparams)

        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_valid_seq = tokenizer.texts_to_sequences(self.X_val)
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_valid_pad = pad_sequences(X_valid_seq, maxlen=100)

        #Converting dataframe to tf dataset
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_pad, self.y_train)).batch(batch_size)



        try:
            teacher_model = keras.models.load_model("./teacher_models/amazon_teacher")
        except:
            print("Error loading model, teacher model was trained?")
            return None

        distiller = Distiller(student=student, teacher=teacher_model)

        distiller.compile(
            optimizer=keras.optimizers.Adam(),
            metrics=[keras.metrics.CategoricalAccuracy()],
            student_loss_fn=keras.losses.CategoricalCrossentropy(),
            distillation_loss_fn=keras.losses.KLDivergence(),
            alpha=alpha,
            temperature=temperature)

        if student_dict is None:
            distiller.train(train_dataset, verbose=1)
        else:
            distiller.train(train_dataset, verbose=1, epochs=student_dict["Epoch"])

        val_accuracy, val_loss = distiller.evaluate(self.X_val, self.y_val, verbose=0)

        run.finish()

        if save_model:
            distiller.save_weights("./student_models/amazon_student_weights")

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

        # pbounds = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (1024, 20000), "same_topology": (0, 1)}

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
            student_dict = student_config.get_amazon_dict()
            print("Training student model with config")
            temperature = 6
            alpha = 0.3
            batch_size = student_dict["batch_size"]
            self.create_model(temperature, alpha, batch_size, save_model=True)
