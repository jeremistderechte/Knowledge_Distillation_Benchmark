from sklearn.model_selection import train_test_split
import autokeras as ak
import tensorflow as tf
import keras
import wandb

class WandbAutoKerasCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        wandb.init(project='AMAZON_GPU_2', entity='jeremy-barenkamp', reinit=True)

    def on_train_end(self, logs=None):
        wandb.finish()

    def on_epoch_end(self, epoch, logs=None):
        wandb.log(logs)


class BestModel:
    def __init__(self):
        self.X_train = None  # Training: 80%
        self.y_train = None
        self.X_val = None  # Validation: 10%
        self.y_val = None
        self.X_test = None  # Test: 10%
        self.y_test = None
        self.classification_type = None
        self.model = None

    def _normal(self):
        input_node = ak.StructuredDataInput()
        output_node = ak.DenseBlock()(input_node)
        output_node = ak.ClassificationHead()(output_node)

        nrml_clf = ak.AutoModel(
            inputs=input_node, outputs=output_node, overwrite=False, max_trials=100
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

        nrml_clf.fit(self.X_train, self.y_train, epochs=20, batch_size=8192,
                       callbacks=[callback, WandbAutoKerasCallback()],
                       validation_data=(self.X_val, self.y_val))

        return nrml_clf

    def _text_classification(self):
        input_node = ak.TextInput()
        output_node = ak.TextBlock(block_type="vanilla")(input_node)
        output_node = ak.ClassificationHead()(output_node)

        txt_clf = ak.AutoModel(
            inputs=input_node, outputs=output_node, overwrite=True, max_trials=100
        )

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        txt_clf.fit(self.X_train, self.y_train, epochs=20, batch_size=4096, validation_data=(self.X_val, self.y_val),
                       callbacks=[callback, WandbAutoKerasCallback()])
        return txt_clf

    def _preprocess_data(self, x, y):
        y = y.astype('int')
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.1,
                                                                                random_state=42, shuffle=True)

    def train(self, x, y, classification_type):
        self.classification_type = classification_type
        self._preprocess_data(x, y)

        if self.classification_type.lower() == "normal":
            clf = self._normal()
        elif self.classification_type.lower() == "text":
            clf = self._text_classification()
        else:
            print("Not defined classificator")
            return 0

        self.model = clf.export_model()

    def save_model(self):
        if self.model is not None:
            try:
                self.model.save(self.classification_type.lower(), save_format="tf")
            except Exception:
                self.model.save(self.classification_type.lower() + ".h5")
        else:
            print("Pleases define a model before")
