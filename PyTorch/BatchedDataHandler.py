import pandas as pd
import torch.utils.data as data_utils
import torch
from sklearn.model_selection import train_test_split

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def get_data(csv_path, x_string_list, y_string_list, batch_size, x_dtype=torch.float32, y_dtype=torch.long,
             train_split=0.8, tokenize_params=None):

    df = pd.read_csv(csv_path)

    if tokenize_params is not None:
        import keras
        from keras.preprocessing.sequence import pad_sequences

        df = df[["overall", "reviewText", "summary", "image"]]
        df.loc[df['overall'] < 3, 'overall'] = 0
        df.loc[df['overall'] == 3, 'overall'] = 1
        df.loc[df['overall'] > 3, 'overall'] = 2
        df.loc[df["image"].isna() == False, "image"] = 1
        df.loc[df["image"].isna(), "image"] = 0
        df = df[df["summary"].isna() == False]
        df = df[df["reviewText"].isna() == False]
        df = df.sample(frac=1).reset_index(drop=True)
        df = df.drop(df[df['overall'] > 1].sample(frac=.9).index)
        df = df.drop(df[df['overall'] == 0.0].sample(frac=.4).index)

        tokenizer = keras.preprocessing.text.Tokenizer(num_words=5000)
        tokenizer.fit_on_texts(df["reviewText"])
        X = tokenizer.texts_to_sequences(df["reviewText"])
        X_pad = pad_sequences(X, maxlen=500)

        y = df["overall"].values
        x = X_pad

    #del df
    else:
        x = df[x_string_list].values
        y = df[y_string_list].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, shuffle=True,
                                                        random_state=69, train_size=train_split)

    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, shuffle=True,
                                                    random_state=69, train_size=0.5)

    del x, y

    train = data_utils.TensorDataset(torch.tensor(x_train, dtype=x_dtype, device=device),
                                     torch.tensor(y_train, dtype=y_dtype, device=device).squeeze())

    test = data_utils.TensorDataset(torch.tensor(torch.tensor(x_test, dtype=x_dtype, device=device)),
                                    torch.tensor(y_test, dtype=y_dtype, device=device).squeeze())

    val = data_utils.TensorDataset(torch.tensor(torch.tensor(x_val, dtype=x_dtype, device=device)),
                                   torch.tensor(y_val, dtype=y_dtype, device=device).squeeze())

    train_loader = data_utils.DataLoader(train, batch_size=batch_size)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size)
    val_loader = data_utils.DataLoader(val, batch_size=batch_size)

    return train_loader, test_loader, val_loader
