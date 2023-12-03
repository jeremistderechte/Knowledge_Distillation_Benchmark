import BatchedDataHandler
import LSTM, FNN
import torch

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def save_model(model, path):
    torch.save(model, path)


def train_teacher(use_automl=False, mandelbrot=False, amazon=False):
    # Handling if User input for mandelbrot and amazon is false --> AutoML will be executed
    use_automl = not mandelbrot and not amazon
    if not use_automl:
        if mandelbrot:
            train_loader_mandelbrot, test_loader_mandelbrot = BatchedDataHandler.get_data("../datasets/Mandelbrot.csv",
                                                                                          ["X", "Y"],
                                                                                          ["divergend"],
                                                                                          4096)
            model_0 = FNN.ReluNet().to(device)

            optimizer = torch.optim.Adam(model_0.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()

            FNN.train_model(train_loader_mandelbrot, test_loader_mandelbrot, model_0,optimizer, loss_fn, epochs=15)

            save_model(model_0, "./teacher_models/mandelbrot_teacher/teacher_mandelbrot.pth")

            # delete referenced for garbage collection for systems with less ram nobody knows how the python gc works XD
            del model_0
            del train_loader_mandelbrot, test_loader_mandelbrot
            torch.cuda.empty_cache()

        if amazon:
            train_loader_amazon, test_loader_amazon = BatchedDataHandler.get_data("../datasets/office_rating.csv"
                                                                                  , ["reviewText"],
                                                                                    ["overall"], 512)

            max_tokens = 10000
            embedding_dim = 100
            input_length = 500

            model_1 = LSTM.LSTMModel(max_tokens, embedding_dim, input_length).to(device)

            optimizer = torch.optim.Adam(model_1.parameters())
            loss_fn = torch.nn.CrossEntropyLoss()

            FNN.train_model(train_loader_amazon, test_loader_amazon, model_1, optimizer, loss_fn, epochs=15)

            save_model(model_1, "./teacher_models/amazon_teacher/teacher_amazon.pth")

    else:
        # AutoML for PyTorch, library have to be found
        print()