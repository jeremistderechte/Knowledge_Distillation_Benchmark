"""
import FNN
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data as data_utils

batch_size = 4096
df = pd.read_csv("../datasets/Mandelbrot.csv").sample(100000)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


X_Train, X_test, y_train, y_test = train_test_split(df[["X", "Y"]], df['divergend'], shuffle=True,
                                                    random_state=42, train_size=0.8)



train = data_utils.TensorDataset(torch.tensor(X_Train.values, dtype=torch.float32, device=device),
                                 torch.tensor(y_train.values, dtype=torch.long, device=device))
test = data_utils.TensorDataset(torch.tensor(torch.tensor(X_test.values, dtype=torch.float32, device=device)),
                                torch.tensor(y_test.values, dtype=torch.long, device=device))

train_loader = data_utils.DataLoader(train, batch_size=batch_size)
test_loader = data_utils.DataLoader(test, batch_size=batch_size)

loss_fn = torch.nn.CrossEntropyLoss()

model_0 = ReluNet.ReluNet()

opt = torch.optim.Adam(model_0.parameters())

ReluNet.train(train_loader, test_loader, model_0, opt, loss_fn, epochs=100, verbose=1)
"""