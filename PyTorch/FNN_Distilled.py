import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as f
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import wandb
from EarlyStopping import EarlyStopping

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_model(train_loader, val_loader, student_model, teacher_model, optimizer, loss_fn, temperature, alpha,
                epochs=1, verbose=0):

    VERBOSE = True if verbose >= 1 else False
    loss = None
    distillation_loss_fn = nn.KLDivLoss(reduction='batchmean')
    teacher_model.eval()

    early_stopping = EarlyStopping(patience=3, min_delta=0.001)

    if VERBOSE:
        print("Training started")

    for epoch in range(epochs):
        student_model.train()
        for data in train_loader:
            x, y = data

            # Forward pass
            student_predictions = student_model(x)
            student_loss = loss_fn(student_predictions, y)

            with torch.inference_mode():
                teacher_predictions = teacher_model(x)

            distillation_loss = (
                    distillation_loss_fn(
                        f.log_softmax(student_predictions / temperature, dim=1),
                        f.softmax(teacher_predictions / temperature, dim=1)) * temperature ** 2
            )

            loss = alpha * student_loss + (1 - alpha) * distillation_loss

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Need to be calculated every epoch, even if verbose is False, for logging in wandb
        metrics = test_model(val_loader, student_model, loss_fn)
        early_stopping(metrics["val-loss"])
        if early_stopping.early_stop:
            print("Early stopping")
            break

        if VERBOSE:
            print(f"Epoch[{(epoch + 1)}/{epochs}], Loss: {round(loss.item(), 4)}, Metrics: {metrics}")


def test_model(test_loader, model, loss_fn, wandb_logging=True):
    test_loss = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1score = 0.0

    batches = len(test_loader)
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            pred = f.softmax(model(x), dim=1)
            test_loss += loss_fn(pred, y).item()
            pred_class = []

            for prediction in pred.cpu().numpy():
                max_value = np.argmax(prediction)
                pred_class.append(max_value)

            true_class = y.cpu().numpy()

            accuracy += accuracy_score(true_class, pred_class)
            precision += precision_score(true_class, pred_class, zero_division=0.0, average="weighted")
            recall += recall_score(true_class, pred_class, zero_division=0.0, average="weighted")
            f1score += f1_score(true_class, pred_class, zero_division=0.0, average="weighted")

    metrics = {"Accuracy": round(accuracy / batches, 4),
               "Precision": round(precision / batches, 4),
               "Recall": round(recall / batches, 4),
               "F1-Score": round(f1score / batches, 4),
               "val-loss": round(test_loss / batches, 4)}

    if wandb_logging:
        wandb.log(metrics)

    return metrics


class ReluKnowledgeDistilled(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2, 16)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(8, 4)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(4, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x = self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        x = self.dense4(x)

        return x
    