import os
import json
import time
import pandas as pd
import numpy as np

import optuna

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli

from hyperopt import STATUS_OK


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

hyperparameters_search_space = {
    "units": [5, 50, 500, 5000],
    "sr": [1e-2, 1e1],
    "mu": [0, 1],
    "input_scaling": [1e-5, 2e2],
    "learning_rate": [1e-5, 1e-2],
    "connectivity": [0.1, 0.5],
    "activation": ["sigmoid", "tanh"],
    "ridge": [1e-8, 1e1],
    "seed": 12345,
    "n_instances": 5,
    "epochs": 100,
    "warmup": 100
}

def objective(trial):
    # Sample hyperparameters
    units = trial.suggest_categorical("units", hyperparameters_search_space["units"])
    sr = trial.suggest_float("sr", hyperparameters_search_space["sr"][0], hyperparameters_search_space["sr"][1], log=True)
    mu = trial.suggest_float("mu", hyperparameters_search_space["mu"][0], hyperparameters_search_space["mu"][1])
    input_scaling = trial.suggest_float("input_scaling", hyperparameters_search_space["input_scaling"][0], hyperparameters_search_space["input_scaling"][1], log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    connectivity = trial.suggest_float("connectivity", hyperparameters_search_space["connectivity"][0], hyperparameters_search_space["connectivity"][1])
    activation = trial.suggest_categorical("activation", hyperparameters_search_space["activation"])
    ridge = trial.suggest_float("ridge", 1e-8, 1e1, log=True)
    epochs = hyperparameters_search_space["epochs"]
    warmup = hyperparameters_search_space["warmup"]
    # Define weight distributions
    W_dist = uniform(high=1.0, low=-1.0)
    Win_dist = bernoulli

    # Placeholder for performance scores
    scores = []

    variable_seed = hyperparameters_search_space["seed"]

    # Run multiple instances
    n_instances = hyperparameters_search_space["n_instances"]
    for _ in range(n_instances):
        # Create the reservoir model
        reservoir = IPReservoir(
            units=units,
            sr=sr,
            mu=mu,
            learning_rate=learning_rate,
            input_scaling=input_scaling,
            W=W_dist,
            Win=Win_dist,
            rc_connectivity=connectivity,
            input_connectivity=connectivity,
            activation=activation,
            epochs=epochs,
            seed=variable_seed
        )

        readout = Ridge(ridge=ridge)

        train_states = reservoir.run(features_scaled_train, reset=True)
        readout = readout.fit(train_states, target_train, warmup=warmup)

        test_states = reservoir.run(features_scaled_test1, reset=False)
        raw_predictions = readout.run(test_states)

        predictions = (raw_predictions > 0.5).astype(int)

        # Flatten arrays for metric calculation
        target_test1_flat = target_test1.flatten()
        predictions_flat = predictions.flatten()

        # Calculate metrics
        f1 = calculate_f1_score(target_test1_flat, predictions_flat)

        scores.append(f1)

        variable_seed += 1
    
    return np.mean(scores)



n_train_iter = 100
study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.HyperbandPruner(
        min_resource=1, max_resource=n_train_iter, reduction_factor=3
    ),
)

start = time.time()
study.optimize(objective, n_trials=20)
search_time = time.time() - start

# Optimization history plot
fig = optuna.visualization.plot_optimization_history(study)
fig.write_image("./hyperband_plots/optimization_history.pdf")  # Save as pdf

# Optimization plot_intermediate_values
fig = optuna.visualization.plot_intermediate_values(study)
fig.write_image("./hyperband_plots/intermediate_values.pdf")  # Save as pdf

# Parameter importances plot
fig = optuna.visualization.plot_param_importances(study)
fig.write_image("./hyperband_plots/param_importances.pdf")  # Save as pdf

# Parallel coordinates plot
fig = optuna.visualization.plot_parallel_coordinate(study)
fig.write_image("./hyperband_plots/parallel_coordinates.pdf")  # Save as pdf

# Contour plot
fig = optuna.visualization.plot_contour(study)
fig.write_image("./hyperband_plots/contour_plot.pdf")  # Save as pdf

# Slice plot
fig = optuna.visualization.plot_slice(study)
fig.write_image("./hyperband_plots/slice_plot.pdf")  # Save as pdf

# Hyperparameter distributions plot
fig = optuna.visualization.plot_param_importances(
    study, target=lambda t: t.duration.total_seconds(), target_name="duration"
)
fig.write_image("./hyperband_plots/param_importances_duration.pdf")  # Save as pdf

# timeline
fig = optuna.visualization.plot_timeline(study)
fig.write_image("./hyperband_plots/timeline.pdf")  # Save as pdf


f1s = []
accs = []
n_instances = hyperparameters_search_space["n_instances"]

variable_seed = hyperparameters_search_space["seed"]
for _ in range(n_instances):
    best_model = IPReservoir(
        units=study.best_params["units"],
        sr=study.best_params["sr"],
        mu=study.best_params["mu"],
        learning_rate=study.best_params["learning_rate"],
        input_scaling=study.best_params["input_scaling"],
        W=uniform(high=1.0, low=-1.0),
        Win=bernoulli,
        rc_connectivity=study.best_params["connectivity"],
        input_connectivity=study.best_params["connectivity"],
        activation=study.best_params["activation"],
        epochs=100,
        seed=variable_seed
    )

    readout = Ridge(ridge=study.best_params["ridge"])

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

print("Best Hyperparameters:", study.best_params)
print("Best Value:", study.best_value)
print("-------------------------------------")
print("acc:", np.mean(accs), np.std(accs))
print("f1:", np.mean(f1s), np.std(f1s))
print("Search time", search_time)
