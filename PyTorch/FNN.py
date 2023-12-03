import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
import wandb


def train_model(train_loader, test_loader, model, optimizer, loss_fn, epochs=15, verbose=0):
    VERBOSE = True if verbose >= 1 else False

    if VERBOSE:
        print("Training started")

    model.train() # Model to training mode -> Saves gradients
    for epoch in range(epochs):
        for data in train_loader:
            x, y = data

            # Forward through the RelU
            output = model(x)
            loss = loss_fn(output, y)

            #Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if VERBOSE:
            metrics = __test(test_loader, model)
            print(f"Epoch[{(epoch+1)}/{epochs}], Loss: {round(loss.item(), 4)}, Metrics: {metrics}")



def __test(test_loader, model):
    with torch.inference_mode():
        for data in test_loader:
            x,y = data
            predictions = f.softmax(model(x), dim=1)

            pred_high = []

            # Softmax as function gives output from 0 to 1, the "possibilities" are written in a tensor of the
            # shape of the output-layer, we will convert them back to a list in which the class with the highest
            # float value is chosen
            for prediction in predictions:
                max_value = np.argmax(prediction)
                pred_high.append(max_value)

            true = y.cpu().numpy()

            metrics = {"Accuracy": accuracy_score(true, pred_high),
                       "Precision": precision_score(true, pred_high, zero_division=0.0),
                       "F1-Score": f1_score(true, pred_high, zero_division=0.0)}

            return metrics

class ReluNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(2, 4096)
        self.relu1 = nn.ReLU()
        self.dense2 = nn.Linear(4096, 2048)
        self.relu2 = nn.ReLU()
        self.dense3 = nn.Linear(2048, 1024)
        self.relu3 = nn.ReLU()
        self.dense4 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.relu1(x)
        x= self.dense2(x)
        x = self.relu2(x)
        x = self.dense3(x)
        x = self.relu3(x)
        logits = self.dense4(x)

        return logits

