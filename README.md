# neuro-evolution
[Neuro-evelution]('https://en.wikipedia.org/wiki/Neuroevolution') for Neural Network hyper parameter tuning


This repository is wrapper for the code provided by [Matt Harvey]('https://github.com/harvitronix')
repository: https://github.com/harvitronix/neural-network-genetic-algorithm

installation: ```pip install git+https://github.com/subpath/neuro-evolution.git```

Example of usage:

1. Create dictionary with parameters
```python
from neuro_evolution import evolution

params = {
    "epochs": [10, 20, 35],
    "batch_size": [10, 20, 40],
    "n_layers": [1, 2, 3, 4],
    "n_neurons": [20, 40, 60],
    "dropout": [0.1, 0.2, 0.5],
    "optimizers": ["nadam", "adam"],
    "activations": ["relu", "sigmoid"],
    "last_layer_activations": ["sigmoid"],
    "losses": ["binary_crossentropy"],
    "metrics": ["accuracy"]
}

```

```python
# x_train, y_train, x_test, y_test - prepared data

search = evolution.NeuroEvolution(generations = 10, population = 10, params=params)

search.evolve(x_train, y_train, x_test, y_test)


100%|██████████| 10/10 [05:37<00:00, 29.58s/it]
100%|██████████| 10/10 [03:55<00:00, 25.55s/it]
100%|██████████| 10/10 [02:05<00:00, 15.05s/it]
100%|██████████| 10/10 [01:37<00:00, 14.03s/it]
100%|██████████| 10/10 [02:49<00:00, 22.53s/it]
100%|██████████| 10/10 [02:37<00:00, 23.14s/it]
100%|██████████| 10/10 [02:36<00:00, 21.37s/it]
100%|██████████| 10/10 [01:57<00:00, 18.56s/it]
100%|██████████| 10/10 [02:42<00:00, 25.29s/it]

"best accuracy: 0.79, best params: {'epochs': 35, 'batch_size': 40, 'n_layers': 2, 'n_neurons': 20, 'dropout': 0.1, 'optimizers': 'nadam', 'activations': 'relu', 'last_layer_activations': 'sigmoid', 'losses': 'binary_crossentropy', 'metrics': 'accuracy'}"

# or you can call it with

search.best_params

```

