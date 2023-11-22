# Benchmark to test the capabilities of knowledge distillation

The neurlan networks of this benchmark are written with tensorflow, a version ported to PyTorch follows soon. The benchmark will test the performance of different teacher and studentmodels and log the results. The Benchmarks trains a simple ReluNet on the Mandelbrot Dataset and a CNN and RNN with an amazon review dataset. 

Links to the datasets:
- Mandelbrot: (build own code)
- Amazon: 


## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
- [Usage](#usage)


## Features

- Automatically searches for the best teachermodel (AutoKeras)
- Automatically searches for the best hyperparams of the studentmodel (bayesian search)
- Trains teacher and student with given values (config.py)
- Logging via wandb for good visualisation

## Getting Started

### Prerequisites

Before you start, ensure you have the following prerequisites:

- Cuda compatible GPU recommended, but CPU is also supported
  - Minimum Cuda compatibility: 3.5 (Kepler), Ampere or newer recommended
  - Your GPU needs at least 10GB of VRAM for the default batch_size while training a textmodel (CNN and RNN (LSTM)), you can decrease the batch_size to use GPUs with less VRAM
    
- Python Libraries:
  - wandb
  - tensorflow
  - pandas
  - scikit-learn
  - bayesian-optimization

### Usage

Clone the repository and run the notebooks:

```bash
git clone https://github.com/jeremistderechte/Knowledge_Distillation_Benchmark.git
cd Knowledge_Distillation_Benchmark
python main.py
