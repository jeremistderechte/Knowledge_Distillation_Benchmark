import os
import sys
from benchmark_config import teacher_config
from benchmark_config.teacher_config import get_mandel_dict, get_amazon_dict

def check_library(library_name):
    try:
        __import__(library_name)
        print(f"{library_name} is installed.")
        return True
    except ImportError:
        print(f"{library_name} is not installed.")
        return False


def check_gpu():
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        if len(tf.config.list_physical_devices('GPU')) >= 1:
            print("GPU(s) found!")
            return True
        else:
            print("No GPU found! :-(")
            return False
    except:
        print("\nUnexpected error or package(s) missing!\n")


def get_data(custom_path=None, mandelbrot=False, amazon=False):
    from Helper import CsvFileHandler

    data = None

    if mandelbrot:
        mandelbrot_params = teacher_config.get_mandel_dict()
        data = CsvFileHandler.data(mandelbrot_params["file_path"])

    elif amazon:
        amazon_params = teacher_config.get_amazon_dict()
        data = CsvFileHandler.data(amazon_params["file_path"])

    elif custom_path is not None:
        path = input("Please type in the filepath to the dataset (.csv): ")
        data = CsvFileHandler.data(path)

    return data.fetch_data()


def start_benchmark():
    from benchmark import BenchmarkMandelbrot
    from benchmark import BenchmarkAmazon
    import benchmark_config
    # getting data
    df_mandelbrot = get_data(mandelbrot=True)
    df_amazon = get_data(amazon=True)

    df_amazon = df_amazon[df_amazon["reviewText"].isnull() == False]
    df_amazon = df_amazon.astype({'reviewText': 'string'})
    df_amazon = df_amazon.drop(df_amazon[df_amazon['overall'] > 4].sample(frac=.9).index)

    # Creating object for benchmark
    benchmark_mandelbrot = BenchmarkMandelbrot.HypParamOpt(df_mandelbrot[["X", "Y"]], df_mandelbrot["divergend"])
    benchmark_amazon = BenchmarkAmazon.HypParamOpt(df_amazon["reviewText"], df_amazon["overall"].astype('int'))

    mandelbrot_exists = False
    amazon_exists = False

    # training teacher

    try:
        os.listdir("./teacher_models/mandelbrot_teacher")
        mandelbrot_exists = True
        print("Teacher model for mandelbrot found, it will be used!")
    except FileNotFoundError:
        print("No Teacher model for mandelbrot found!")
    except:
        print("Unhandled exception while looking for teacher model")

    try:
        os.listdir("./teacher_models/amazon_teacher")
        amazon_exists = True
        print("Teacher model for amazon found, it will be used!")
    except FileNotFoundError:
        print("No Teacher model for amazon found!")
    except:
        print("Unhandled exception while looking for teacher model")

    pbounds_mandelbrot = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (1024, 20000), "same_topology": (0, 1)}
    pbounds_amazon = {'temperature': (1, 20), 'alpha': (0, 1), "batch_size": (1024, 20000), "same_topology": (0, 1)}

    from benchmark import TeacherModel
    if mandelbrot_exists and amazon_exists:
        # Change these values, for other hyperparameters (just values)
        benchmark_mandelbrot.run(pbounds_mandelbrot)
        benchmark_amazon.run(pbounds_amazon)

    elif mandelbrot_exists and not amazon_exists:
        teacher_model_amazon = TeacherModel.Teacher(df_amazon["reviewText"], df_amazon["overall"].astype('int'),
                                                    get_amazon_dict())

        try:
            teacher_model_amazon.train_teacher()
        except:
            teacher_model_amazon.train_best_teacher()


    elif amazon_exists and not mandelbrot_exists:
        teacher_model_mandelbrot = TeacherModel.Teacher(df_mandelbrot[["X", "Y"]], df_mandelbrot["divergend"],
                                                        get_mandel_dict())

        try:
            teacher_model_mandelbrot.train_teacher()
        except:
            teacher_model_mandelbrot.train_best_teacher()

    else:
        # Training teacher
        teacher_model_mandelbrot = TeacherModel.Teacher(df_mandelbrot[["X", "Y"]], df_mandelbrot["divergend"],
                                                        get_mandel_dict())
        teacher_model_amazon = TeacherModel.Teacher(df_amazon["reviewText"], df_amazon["overall"].astype('int'),
                                                    get_amazon_dict())
        try:
            teacher_model_mandelbrot.train_teacher()
            teacher_model_amazon.train_teacher()
        except:
            teacher_model_mandelbrot.train_best_teacher()
            teacher_model_amazon.train_best_teacher()

    benchmark_mandelbrot.run(pbounds_mandelbrot)
    benchmark_amazon.run(pbounds_amazon)


def main():
    print("\n" * 3)
    print("#" * 60)
    print("Welcome to the knowledge Distillation benchmark tool by Jeremy Barenkamp!\n")

    action_dict = {1: "Start Benchmark (GPU strongly recommended)",
                   2: "Check if GPU is available and all packages are installed",
                   3: "Exit Benchmark"
                   }

    for key, action in zip(action_dict.keys(), action_dict.values()):
        print(f"({key}) {action}")

    legit_select = False

    while not legit_select:
        try:
            user_selection = int(input("\nYour selection: "))

            if user_selection in action_dict.keys():
                legit_select = True

        except:
            print("Please enter a number that is displayed in the selection screen")

    if user_selection == 1:
        start_benchmark()

    elif user_selection == 2:

        prerequisites = ["sklearn", "keras", "tensorflow", "pandas", "numpy", "pickle",
                         "matplotlib.pyplot", "wandb"]
        all_packages = []
        for library in prerequisites:
            all_packages.append(check_library(library))

        if False not in all_packages:
            print("\nAll packages are installed\n")
        else:
            print(f"\n!!! {all_packages.count(False)} package(s) are missing !!!\n")

        gpu = check_gpu()

        print("\n" * 2)
        print("#" * 60)
        print("Summary")
        print(f"Packages: {all_packages.count(False)} packages are missing")
        print(f"GPU: Yes") if gpu else print(f"GPU: No")
        print("#" * 60)
        print("\n" * 2)

        should_continue = input("Do you want to restart the script (y/n): \n")

        if should_continue.upper() == "Y":
            python = sys.executable
            os.execl(python, python, *sys.argv)
        else:
            exit()

    elif (user_selection == 3):
        exit()


if __name__ == "__main__":
    main()
