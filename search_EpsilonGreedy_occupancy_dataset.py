import numpy as np
from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

import random
from collections import deque
import os
import pandas as pd

from scipy.stats import loguniform
from scipy.stats import spearmanr

import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

import time

from reservoir_hp_visualizer import visualize_search
from utils import calculate_f1_score
from base_smart_reservoir_search import BaseEpsilonGreedyReservoirHPSearch

class EpsilonGreedyReservoirHPSearch_F1(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):
        f1s = []
        variable_seed = params["seed"]
        
        for _ in range(params["n_instances"]):
            reservoir = IPReservoir(
                units=params['units'],
                sr=params['sr'],
                mu=params['mu'],
                input_scaling=params['input_scaling'],
                learning_rate=params['learning_rate'],
                W=uniform(high=1.0, low=-1.0),
                Win=bernoulli,
                rc_connectivity=params['connectivity'],
                input_connectivity=params['connectivity'],
                activation=params['activation'],
                epochs=params['epochs'],
                seed=variable_seed
            )
            
            readout = Ridge(ridge=params['ridge'])
            
            try:
                train_states = reservoir.run(self.X_train, reset=True)
                readout = readout.fit(train_states, self.y_train, warmup=params['warmup'])
                
                test_states = reservoir.run(self.X_test, reset=False)
                raw_predictions = readout.run(test_states)

                predictions = (raw_predictions > 0.5).astype(int)

                # Flatten arrays for metric calculation
                target_test1_flat = target_test1.flatten()
                predictions_flat = predictions.flatten()

                # Calculate metrics
                f1 = calculate_f1_score(target_test1_flat, predictions_flat)
                f1s += [f1]
            
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                return -np.inf

            variable_seed += 1

        return np.mean(f1s)

if __name__ == '__main__':

    training_data_location = './datasets/occupancy_detection/datatraining.txt'
    testing_data_location1 = './datasets/occupancy_detection/datatest.txt'
    testing_data_location2 = './datasets/occupancy_detection/datatest2.txt'

    results_path = './testing_results/occupancy_detection'

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

    n_iterations = 1
    searcher = EpsilonGreedyReservoirHPSearch_F1(features_scaled_train, target_train, features_scaled_test1, target_test1, n_iterations, 0.3)
    start_time = time.time()
    best_params, best_score = searcher.search(n_iterations=n_iterations)
    train_time = time.time() - start_time
    visualize_search(searcher, results_path)

    f1s = []
    accs = []
    variable_seed = searcher.search_space["seed"]

    for _ in range(searcher.search_space["n_instances"]):

        reservoir = IPReservoir(
            units=best_params['units'],
            sr=best_params['sr'],
            mu=best_params['mu'],
            input_scaling=best_params['input_scaling'],
            learning_rate=best_params['learning_rate'],
            W=uniform(high=1.0, low=-1.0),
            Win=bernoulli,
            rc_connectivity=best_params['connectivity'],
            input_connectivity=best_params['connectivity'],
            activation=best_params['activation'],
            epochs=best_params['epochs'],
            seed=variable_seed
        )
                
        readout = Ridge(ridge=best_params['ridge'])

        train_states = reservoir.run(searcher.X_train, reset=True)
        readout = readout.fit(train_states, searcher.y_train, warmup=best_params['warmup'])
        
        test_states = reservoir.run(searcher.X_test, reset=False)
        raw_predictions = readout.run(test_states)

        predictions = (raw_predictions > 0.5).astype(int)

        # Flatten arrays for metric calculation
        target_test1_flat = target_test1.flatten()
        predictions_flat = predictions.flatten()

        # Calculate metrics
        f1 = calculate_f1_score(target_test1_flat, predictions_flat)
        acc = accuracy_score(target_test1_flat, predictions_flat)

        f1s += [f1]
        accs += [acc]

        variable_seed += 1

    print("best_params", best_params)
    print("history", searcher.history)
    print("f1:", np.mean(f1s), np.std(f1s))
    print("acc:", np.mean(accs), np.std(accs))
    print("train time", train_time)
