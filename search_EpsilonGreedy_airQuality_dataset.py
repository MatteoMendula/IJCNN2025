import numpy as np
from reservoirpy.nodes import IPReservoir, Ridge
from reservoirpy.mat_gen import uniform, bernoulli
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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
from base_smart_reservoir_search import BaseEpsilonGreedyReservoirHPSearch

class EpsilonGreedyReservoirHPSearch_R2(BaseEpsilonGreedyReservoirHPSearch):
    def evaluate(self, params):
        rs2 = []
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
                predictions = readout.run(test_states)
                
                r2score = r2_score(self.y_test, predictions)
                rs2 += [r2score]
            
            except Exception as e:
                print(f"Error during evaluation: {str(e)}")
                return -np.inf

            variable_seed += 1

        return np.mean(rs2)

if __name__ == '__main__':

    list_of_files = os.listdir('./datasets/airQualityBeijing')
    results_path = './testing_results/airQualityBeijing'

    file = f'./datasets/airQualityBeijing/{list_of_files[0]}'
    df = pd.read_csv(file)

    X = df.drop('PM2.5',axis=1)
    y = df['PM2.5']
    xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=.2, shuffle=False)

    # trasnform the data to numpy array
    xtrain = xtrain.values
    xtest = xtest.values
    ytrain = ytrain.values
    ytest = ytest.values

    # add 1 dimension to ytrain and ytest
    ytrain = ytrain[:,np.newaxis]
    ytest = ytest[:,np.newaxis]

    # scale the data
    scaler = MinMaxScaler()
    xtrain = scaler.fit_transform(xtrain)
    xtest = scaler.transform(xtest)

    n_iterations = 20
    searcher = EpsilonGreedyReservoirHPSearch_R2(xtrain, ytrain, xtest, ytest, n_iterations)
    start_time = time.time()
    best_params, best_score = searcher.search(n_iterations)
    train_time = time.time() - start_time
    visualize_search(searcher, results_path)

    r2s = []
    mses = []
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
        predictions = readout.run(test_states)
        
        r2score = r2_score(searcher.y_test, predictions)
        mse = mean_squared_error(searcher.y_test, predictions)
        r2s += [r2score]
        mses += [mse]

        variable_seed += 1

    print("best_params", best_params)
    print("history", searcher.history)
    print("R2:", np.mean(r2s), np.std(r2s))
    print("MSE:", np.mean(mses), np.std(mses))
    print("train time", train_time)
