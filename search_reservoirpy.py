import os
import json
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli
from reservoirpy.hyper import research
from reservoirpy.hyper import plot_hyperopt_report

from hyperopt import STATUS_OK


def calculate_f1_score(series_a, series_b):
    # Convert to lists if not already
    series_a = list(series_a)
    series_b = list(series_b)

    # Initialize counts
    TP = TN = FP = FN = 0

    # Calculate TP, TN, FP, FN
    for a, b in zip(series_a, series_b):
        if a == 1 and b == 1:
            TP += 1
        elif a == 0 and b == 0:
            TN += 1
        elif a == 0 and b == 1:
            FP += 1
        elif a == 1 and b == 0:
            FN += 1

    # Calculate Precision and Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Calculate F1 Score
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    return f1_score



# Define the objective function for optimization
def objective(dataset, config, *, N, sr, mu, lr, input_scaling, connectivity, activation, ridge, epochs, warmup, seed):
    print(f"Objective parameters: N={N}, sr={sr}, mu={mu}, activation={activation}")
    x_train, y_train, x_test, y_test = dataset
    instances = config["instances_per_trial"]
    variable_seed = seed

    accs, f1s = [], []
    for _ in range(instances):
        # Build the reservoir
        reservoir = IPReservoir(
            units=N,
            sr=sr,
            mu=mu,
            learning_rate=lr,
            input_scaling=input_scaling,
            W=uniform(high=1.0, low=-1.0),
            Win=bernoulli,
            rc_connectivity=connectivity,
            input_connectivity=connectivity,
            activation=activation,
            epochs=epochs,
            seed=variable_seed
        )
        readout = Ridge(ridge=ridge)

        # Train and evaluate the model
        train_states = reservoir.run(x_train, reset=True)
        readout = readout.fit(train_states, y_train, warmup=warmup)

        test_states = reservoir.run(x_test, reset=False)
        raw_predictions = readout.run(test_states)

        predictions = (raw_predictions > 0.5).astype(int)

        # Flatten arrays for metric calculation
        target_test1_flat = target_test1.flatten()
        predictions_flat = predictions.flatten()

        # Calculate metrics
        accuracy = accuracy_score(target_test1_flat, predictions_flat)
        f1 = calculate_f1_score(target_test1_flat, predictions_flat)

        accs.append(accuracy)
        f1s.append(f1)
        variable_seed += 1

    return {'loss': -np.mean(f1), 'status': STATUS_OK, 'f1': np.mean(f1s), 'acc': np.mean(accs),}

# Define the hyperparameter search configuration
hyperopt_config = {
    "exp": "hyperopt-multiscroll",  # Experiment name
    "hp_max_evals": 20,            # Number of hyperparameter trials
    "hp_method": "random",         # Random sampling method
    "seed": 42,                    # Random seed for reproducibility
    "instances_per_trial": 5,      # Number of instances per trial
    "hp_space": {
        "N": ["choice", 5, 50, 500, 5000],            # Reservoir units
        "sr": ["loguniform", 1e-2, 1e1],           # Spectral radius
        "mu": ["uniform", 0, 1],                    # Leaking rate
        "input_scaling": ["loguniform", 1e-5, 2e2],         # Input scaling
        "lr": ["loguniform", 1e-5, 1e-2],            # Learning rate
        "connectivity": ["uniform", 0.1, 0.5],          # Connectivity
        "activation": ["choice", "sigmoid", "tanh"],            # Reservoir units
        "ridge": ["loguniform", 1e-8, 1e1],       # Ridge regularization
        "epochs": ["choice", 100],                # Number of epochs
        "warmup": ["choice", 100],                # Warmup steps
        "seed": ["choice", 12345],                 # Reservoir seed
    },
}

with open(f"{hyperopt_config['exp']}.config.json", "w+") as f:
    json.dump(hyperopt_config, f)

training_data_location = 'occupancy_detection/datatraining.txt'
testing_data_location1 = 'occupancy_detection/datatest.txt'
testing_data_location2 = 'occupancy_detection/datatest2.txt'

# read data from txt file datatrainin.txt
training_data = pd.read_csv(training_data_location, sep=',', header=0)
testing_data1 = pd.read_csv(testing_data_location1, sep=',', header=0)
testing_data2 = pd.read_csv(testing_data_location2, sep=',', header=0)

# Step 1: Extract features and target
features_train = training_data[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_train = training_data['Occupancy'].values  # True/False or 1/0 for occupancy

features_test1 = testing_data1[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test1 = testing_data1['Occupancy'].values  # True/False or 1/0 for occupancy

features_test2 = testing_data2[['Temperature', 'Humidity', 'Light', 'CO2', 'HumidityRatio']].values
target_test2 = testing_data2['Occupancy'].values  # True/False or 1/0 for occupancy

# Step 2: Normalize features
scaler = MinMaxScaler()
features_scaled_train = scaler.fit_transform(features_train)
features_scaled_test1 = scaler.transform(features_test1)
features_scaled_test2 = scaler.transform(features_test2)

target_train = target_train[:, np.newaxis] 
target_test1 = target_test1[:, np.newaxis] 
target_test2 = target_test2[:, np.newaxis] 

# Perform the hyperparameter search
dataset = (features_scaled_train, target_train, features_scaled_test1, target_test1)
start = time.time()
best = research(objective, dataset, f"{hyperopt_config['exp']}.config.json", ".")
search_time = time.time() - start

# fig = plot_hyperopt_report(hyperopt_config["exp"], ("lr", "sr", "ridge"), metric="f1")
# fig.savefig("hyperopt_report.pdf")

best_N = hyperopt_config["hp_space"]["N"][1:][best[0]["N"]]
best_sr = best[0]["sr"]
best_mu = best[0]["mu"]
best_lr = best[0]["lr"]
best_connectivity = best[0]["connectivity"]
best_activation = hyperopt_config["hp_space"]["activation"][1:][best[0]["activation"]]
best_ridge = best[0]["ridge"]
best_warmup = hyperopt_config["hp_space"]["warmup"][1:][best[0]["warmup"]]
best_epochs = hyperopt_config["hp_space"]["epochs"][1:][best[0]["epochs"]]
best_input_scaling = best[0]["input_scaling"]

f1s = []
accs = []
n_instances = hyperopt_config["instances_per_trial"]

variable_seed = hyperopt_config["hp_space"]["seed"][1]
for _ in range(n_instances):
    best_model = IPReservoir(
        units=best_N,
        sr=best_sr,
        mu=best_mu,
        learning_rate=best_lr,
        input_scaling= best_input_scaling,
        W=uniform(high=1.0, low=-1.0),
        Win=bernoulli,
        rc_connectivity=best_connectivity,
        input_connectivity=best_connectivity,
        activation=best_activation,
        epochs=100,
        seed=variable_seed
    )

    readout = Ridge(ridge=best_ridge)

    train_states = best_model.run(features_scaled_train, reset=True)
    readout = readout.fit(train_states, target_train, warmup=100)

    test_states = best_model.run(features_scaled_test1, reset=False)
    raw_predictions = readout.run(test_states)

    predictions = (raw_predictions > 0.5).astype(int)

    # Flatten arrays for metric calculation
    target_test1_flat = target_test1.flatten()
    predictions_flat = predictions.flatten()

    # Calculate metrics
    accuracy = accuracy_score(target_test1_flat, predictions_flat)
    f1 = calculate_f1_score(target_test1_flat, predictions_flat)

    accs.append(accuracy)
    f1s.append(f1)
    variable_seed += 1

print("best param", best)
print("acc:", np.mean(accs), np.std(accs))
print("f1:", np.mean(f1s), np.std(f1s))
print("Search time", search_time)
