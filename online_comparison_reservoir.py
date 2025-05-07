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

# best_params_reservoir_py = ({'N': 0, 'activation': 0, 'connectivity': 0.37942316327125414, 'epochs': 0, 'input_scaling': 0.5832996914534163, 'lr': 0.0012953948636839738, 'mu': 0.4349979536508063, 'ridge': 0.00021135596946478938, 'seed': 0, 'sr': 0.09442721440119983, 'warmup': 0}, None)

# best_N = hyperopt_config["hp_space"]["N"][1:][best_params_reservoir_py[0]["N"]]
# best_sr = best_params_reservoir_py[0]["sr"]
# best_mu = best_params_reservoir_py[0]["mu"]
# best_lr = best_params_reservoir_py[0]["lr"]
# best_connectivity = best_params_reservoir_py[0]["connectivity"]
# best_activation = hyperopt_config["hp_space"]["activation"][1:][best_params_reservoir_py[0]["activation"]]
# best_ridge = best_params_reservoir_py[0]["ridge"]
# best_warmup = hyperopt_config["hp_space"]["warmup"][1:][best_params_reservoir_py[0]["warmup"]]
# best_epochs = hyperopt_config["hp_space"]["epochs"][1:][best_params_reservoir_py[0]["epochs"]]
# best_input_scaling = best_params_reservoir_py[0]["input_scaling"]
# best_seed = hyperopt_config["hp_space"]["seed"][1:][best_params_reservoir_py[0]["seed"]]

best_params_reservoir_py = {
    "N": 500,
    "activation": "sigmoid",
    "connectivity": 0.3442498557981145,
    "epochs": 100,
    "input_scaling": 8.03437943624333,
    "lr": 4.4731043285836344e-05,
    "mu": 0.6091041490883193,
    "ridge": 0.03718402541903902,
    "seed": 12345,
    "sr": 0.07071032186253305,
    "warmup": 100
  }

print(best_params_reservoir_py)

best_N = best_params_reservoir_py["N"]
best_sr = best_params_reservoir_py["sr"]
best_mu = best_params_reservoir_py["mu"]
best_lr = best_params_reservoir_py["lr"]
best_connectivity = best_params_reservoir_py["connectivity"]
best_activation = best_params_reservoir_py["activation"]
best_ridge = best_params_reservoir_py["ridge"]
best_warmup = best_params_reservoir_py["warmup"]
best_epochs = best_params_reservoir_py["epochs"]
best_input_scaling = best_params_reservoir_py["input_scaling"]
best_seed = 12345


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
    target_test1_flat = target_test1.flatten()
    acc = accuracy_score(target_test1_flat, predictions)
    f1 = calculate_f1_score(target_test1_flat, predictions)

    f1s += [f1]
    accs += [acc]

    variable_seed += 1 

print("f1:", np.mean(f1s), np.std(f1s))
print("accuracy:", np.mean(accs), np.std(accs))
print("train_times: ", np.mean(train_times), np.std(train_times))
print("test_times: ", np.mean(test_times), np.std(test_times))

# f1: 0.7429034962461227 0.3721949667395966
# accuracy: 0.887579737335835 0.12948986149670108
# train_times:  0.0003134963842959948 4.147194649049778e-05
# test_times:  0.0002924310601898251 3.163898527415378e-05