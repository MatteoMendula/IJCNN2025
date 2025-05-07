import os
import time
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from reservoirpy.nodes import IPReservoir
from reservoirpy.mat_gen import uniform, bernoulli


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

# RLS Update Function
def rls_update(r, y, w, P):
    """Recursive Least Squares update."""
    k = P @ r
    rPr = r.T @ k
    c = 1.0 / (1.0 + rPr)
    P = P - c * np.outer(k, k)
    e = y - w.T @ r  # Prediction error
    dw = c * e * k
    w = w + dw
    return w, P

# Hyperparameter configuration
hyperopt_config = {
    "exp": "hyperopt-multiscroll",
    "hp_max_evals": 20,
    "hp_method": "random",
    "seed": 42,
    "instances_per_trial": 5,
    "hp_space": {
        "N": ["choice", 5, 50, 500, 5000],
        "sr": ["loguniform", 1e-2, 1e1],
        "mu": ["uniform", 0, 1],
        "input_scaling": ["loguniform", 1e-5, 2e2],
        "lr": ["loguniform", 1e-5, 1e-2],
        "connectivity": ["uniform", 0.1, 0.5],
        "activation": ["choice", "sigmoid", "tanh"],
        "ridge": ["loguniform", 1e-8, 1e1],
        "epochs": ["choice", 100],
        "warmup": ["choice", 100],
        "seed": ["choice", 12345],
    },
}

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
best_params_optuna_py = {'units': 500, 'sr': 0.11142895906469773, 'mu': 0.16706462778855036, 'input_scaling': 0.007330807163595258, 'learning_rate': 0.0003256447331920584, 'connectivity': 0.10212406685271117, 'activation': 'sigmoid', 'ridge': 3.628665379631813e-05}

best_N = best_params_optuna_py["units"]
best_sr = best_params_optuna_py["sr"]
best_mu = best_params_optuna_py["mu"]
best_input_scaling = best_params_optuna_py["input_scaling"]
best_lr = best_params_optuna_py["learning_rate"]
best_connectivity = best_params_optuna_py["connectivity"]
best_activation = best_params_optuna_py["activation"]
best_ridge = best_params_optuna_py["ridge"]

best_warmup = hyperopt_config["hp_space"]["warmup"][1]
best_epochs = hyperopt_config["hp_space"]["epochs"][1]
best_seed = hyperopt_config["hp_space"]["seed"][1]

# RLS parameters
lambda_factor = 0.99  # Forgetting factor
alpha = best_ridge
variable_seed = best_seed
train_times = []
test_times = []

f1s = []
accs = []

for _ in range(hyperopt_config["instances_per_trial"]):

    # Initialize reservoir
    reservoir = IPReservoir(
        units=best_N,
        sr=best_sr,
        mu=best_mu,
        input_scaling=best_input_scaling,
        W=uniform(high=1.0, low=-1.0),
        Win=bernoulli,
        rc_connectivity=best_connectivity,
        input_connectivity=best_connectivity,
        activation=best_activation,
        seed=variable_seed,
    )

    P = np.eye(reservoir.units) / alpha  # Initialize inverse covariance matrix
    w = np.zeros((reservoir.units, 1))  # Initialize weights

    # Train RLS
    for i, t in enumerate(range(features_test2.shape[0])):
        start_time = time.time()
        r_t = reservoir.run(features_test2[t:t+1], reset=(t == 0))  # Single input
        r_t = r_t.T  # Ensure it's a column vector
        y_t = target_test2[t]  # Corresponding target
        w, P = rls_update(r_t, y_t, w, P)
        train_times += [time.time() - start_time]

    # Test the model sequentially
    predictions = []
    for t in range(features_test1.shape[0]):
        start_time = time.time()
        r_t = reservoir.run(features_test1[t:t+1], reset=False)
        r_t = r_t.T  # Ensure it's a column vector

        # Generate prediction using the updated weights
        y_pred = w.T @ r_t
        raw_prediction = y_pred[0, 0]
        prediction = (raw_prediction > 0.5).astype(int)
        predictions.append(prediction)
        test_times += [time.time() - start_time]

    # Evaluate the model
    target_test2_flat = target_test1.flatten()
    acc = accuracy_score(target_test2_flat, predictions)
    f1 = calculate_f1_score(target_test2_flat, predictions)

    f1s += [f1]
    accs += [acc]

    variable_seed += 1 

print("f1:", np.mean(f1s), np.std(f1s))
print("accuracy:", np.mean(accs), np.std(accs))
print("train_times: ", np.mean(train_times), np.std(train_times))
print("test_times: ", np.mean(test_times), np.std(test_times))

# f1: 0.970632185811916 0.000526013837617052
# accuracy: 0.9780112570356472 0.0003826656295379204
# train_times:  0.002928117800189199 0.0009062705034207872
# test_times:  0.000789678387525605 5.750364514169518e-05