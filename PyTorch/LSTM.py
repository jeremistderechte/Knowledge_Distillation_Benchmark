import torch.nn as nn
import torch
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import torch.nn.functional as f
import numpy as np
import torch.utils.data as data_utils

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def train_model(train_loader, test_loader, model, optimizer, loss_fn, epochs=15, verbose=0):
    VERBOSE = True if verbose >= 1 else False
    loss = None

    if VERBOSE:
        print("Training started")

    for epoch in range(epochs):
        model.train()
        for data in train_loader:
            x, y = data
            x = x.long()

            # Forward pass
            outputs = model(x)
            loss = loss_fn(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if VERBOSE:
            metrics = _test_model(test_loader, model, loss_fn)
            print(f"Epoch[{(epoch + 1)}/{epochs}], Loss: {round(loss.item(), 4)}, Metrics: {metrics}")


def _test_model(test_loader, model, loss_fn):
    test_loss = 0.0
    accuracy = 0.0
    precision = 0.0
    recall = 0.0
    f1score = 0.0

    batches = len(test_loader)
    with torch.inference_mode():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)

            x = x.long()

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
               "test-loss": round(test_loss / batches, 4)}

    return metrics


class LSTMModel(nn.Module):
    def __init__(self, max_tokens, embedding_dim, input_length):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(max_tokens, embedding_dim)
        self.lstm1 = nn.LSTM(embedding_dim, 512, batch_first=True)
        self.dropout1 = nn.Dropout(0.3)
        self.lstm2 = nn.LSTM(512, 256, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        self.fc = nn.Linear(256, 3)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x = self.dropout1(x)
        x, _ = self.lstm2(x)
        x = self.dropout2(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x