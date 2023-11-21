import tensorflow as tf
import keras
import wandb
import numpy as np


class Distiller(keras.Model): #Destiller class for knowledge destillation
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):

        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        x, y = data

        teacher_predictions = self.teacher(x, training=False)

        with tf.GradientTape() as tape:
            # Forward pass of student (is a tensor)
            student_predictions = self.student(x, training=True)


            student_loss = self.student_loss_fn(y, student_predictions)


            distillation_loss = (
                self.distillation_loss_fn(
                    tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                    tf.nn.softmax(student_predictions / self.temperature, axis=1),
                )
                * self.temperature**2
            )

            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        #gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        #weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        #metrics
        self.compiled_metrics.update_state(y, student_predictions)
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):

        x, y = data
        #predictions
        y_prediction = self.student(x, training=False)

        #loss
        student_loss = self.student_loss_fn(y, y_prediction)

        #metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        results = {m.name: m.result() for m in self.metrics}
        results.update({"loss": student_loss})
        return results

    def train(self, train_dataset, val_dataset=None, epochs=20, verbose=1, patience=3):
        best_loss = float('inf')
        wait = 0  # Counter for patience

        for epoch in range(epochs):
            if verbose == 1:
                print(f"Epoch {epoch + 1}/{epochs}")

            epoch_metrics = {}

            # Training phase
            for step, data in enumerate(train_dataset):
                results = self.train_step(data)

                for k, v in results.items():
                    if k not in epoch_metrics:
                        epoch_metrics[k] = []
                    epoch_metrics[k].append(v.numpy())

            epoch_metrics_avg = {k: np.mean(v) for k, v in epoch_metrics.items()}
            wandb.log(epoch_metrics_avg)

            if verbose == 1:
                print(f"Metrics for epoch {epoch + 1}: ",
                      " ".join([f"{k}: {v:.4f}" for k, v in epoch_metrics_avg.items()]))

            # Validation phase
            if val_dataset is not None:
                val_loss = self.helper(val_dataset)

                if verbose == 1:
                    print(f"Validation loss for epoch {epoch + 1}: {val_loss:.4f}")

                # Early stopping check
                if val_loss < best_loss:
                    best_loss = val_loss
                    wait = 0  # Reset counter
                else:
                    wait += 1  # Increment counter if no improvement

                if wait >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break  # Stop training if patience exceeded

        return best_loss

    def call(self, x):
        return self.student(x, training=False)

    def helper(self, val_dataset):
        val_loss = 0
        for step, data in enumerate(val_dataset):
            results = self.test_step(data)
            val_loss += results['loss']
        val_loss /= (step + 1)
        return val_loss
