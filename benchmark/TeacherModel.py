from keras.models import Sequential
from sklearn.model_selection import train_test_split
import keras.layers as layers
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import keras
from benchmark import AutoML
from keras.optimizers import Adam


class Teacher:
    def __init__(self, X, Y, teacher_param):
        self.teacher_param = teacher_param

        self.X = X
        self.y = Y
        self.X_train = None  # Training: 80%
        self.y_train = None
        self.X_val = None  # Validation: 10%
        self.y_val = None
        self.X_test = None  # Test: 10%
        self.y_test = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.1, random_state=42, shuffle=True)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1, random_state=42)

    def train_teacher(self):
        if self.teacher_param["type"].lower() == "normal":
            self.normal_model(self.teacher_param)
        elif self.teacher_param["type"].lower() == "text":
            self.text_model(self.teacher_param)

    def train_best_teacher(self):
        if self.teacher_param["type"].lower() == "normal":
            best_model_normal = AutoML.BestModel()
            best_model_normal.train(self.X, self.y, "normal")
            best_model_normal.save_model()
        elif self.teacher_param["type"].lower() == "text":
            best_model_text = AutoML.BestModel()
            best_model_text.train(self.X, self.y, "text")
            best_model_text.save_model()

    def normal_model(self, param_dict):
        model = Sequential()

        neuron_list = param_dict["Neuron"]

        for layer in range(len(neuron_list)):
            if layer == 0:
                model.add(layers.Dense(neuron_list[layer], activation='relu', input_shape=(self.X_train.shape[1],)))
            else:
                model.add(layers.Dense(neuron_list[layer], activation='relu'))

        model.add(layers.Dense(2, activation='softmax'))

        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.X_train, to_categorical(self.y_train), epochs=param_dict["Epoch"], batch_size=param_dict["batch_size"],
                  validation_data=(self.X_val, to_categorical(self.y_val)))
        model.save('./teacher_models/mandelbrot_teacher')

    def text_model(self, param_dict, use_lstm=False):

        self.y[self.y < 3] = 0
        self.y[self.y == 3] = 1
        self.y[self.y > 3] = 2

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42,
                                                                                shuffle=True)

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train, test_size=0.1,
                                                                              random_state=42)

        # Vectorization - let's say we are using TF's Tokenizer and pad_sequences
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(self.X_train)
        X_train_seq = tokenizer.texts_to_sequences(self.X_train)
        X_valid_seq = tokenizer.texts_to_sequences(self.X_val)
        X_train_pad = pad_sequences(X_train_seq, maxlen=100)
        X_valid_pad = pad_sequences(X_valid_seq, maxlen=100)

        input_length = 100


        max_tokens = 5000
        embedding_dim = 256
        dropout_rate_embedding = 0.5
        learning_rate = 0.001

        if not use_lstm:

            input_layer = layers.Input(shape=(input_length,), dtype="int32")
            x = layers.Embedding(max_tokens, embedding_dim, input_length=input_length)(input_layer)
            x = layers.Dropout(dropout_rate_embedding)(x)

            # Convolutional block 1
            x = layers.Conv1D(512, 3, activation='relu')(x)
            x = layers.Conv1D(64, 3, activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=2)(x)

            # Convolutional block 2
            x = layers.Conv1D(128, 3, activation='relu')(x)
            x = layers.Conv1D(64, 3, activation='relu')(x)
            x = layers.MaxPooling1D(pool_size=2)(x)

            # Convolutional block 3 (extra block not in original specification)
            x = layers.Conv1D(16, 3, activation='relu')(x)
            x = layers.Conv1D(512, 3, activation='relu')(x)

            x = layers.Flatten()(x)

            # Dense layers
            x = layers.Dense(32, activation='relu')(x)
            x = layers.Dense(32, activation='relu')(x)

            output_layer = layers.Dense(3, activation='softmax')(x)  # Assuming binary classification

            # Compile the model
            model = keras.Model(inputs=input_layer, outputs=output_layer)
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='categorical_crossentropy',
                          metrics=['accuracy'])
        else:

            # not ready yet
            model = Sequential()
            model.add(layers.Embedding(max_tokens, embedding_dim, input_length=input_length))
            model.add(layers.LSTM(512))
            model.add(layers.Dropout(0.3))
            model.add(layers.LSTM(256))
            model.add(layers.Dropout(0.1))
            model.add(layers.Dense(3, activation="softmax"))

        model.fit(X_train_pad, to_categorical(self.y_train), batch_size=4096, epochs=25,
                  validation_data=(X_valid_pad, to_categorical(self.y_val)))

        model.save('./teacher_models/amazon_teacher')

