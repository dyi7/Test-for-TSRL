import numpy as np
import tensorflow as tf
from PureJumpSolver import BSDESolver
import BarrierEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd
import random
# import wandb
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)
 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 40
    strike = 1.0
    lamb = 0.3
    r = 0.04
    sigma = 0.25
    aver_jump = 0.5
    var_jump = 0.25
    x_init = 1.0
    barrier = 1.3
    barrier_direction = 'Up' # Down or Up
    optionality = 'Out'  # Out = Knock Out Option   In = Knock In Option  ###
    config = {
                "eqn_config": {
                    "_comment": "barrier call option",
                    "eqn_name": "BarrierCallOption",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "lamb":lamb,
                    "aver_jump":aver_jump,
                    "var_jump":var_jump,
                    "x_init":x_init,
                    "barrier": barrier,
                    "barrier_direction": barrier_direction,
                    "optionality": optionality
                },
                "net_config": {
                    "num_hiddens": [dim+20,dim+20],
                    "lr_values": [5e-5, 5e-7],
                    "lr_boundaries": [20000],
                    "num_iterations": 25000, 
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    # track hyperparameters and run metadata
    # wandb.init(project="Morgan Project", config = config)
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train() 
    
       
    #Simulate the BSDE after training - MtM scenarios
    samples = bsde.sample(P)
    
    """
    Monte Carlo Price
    """
    stock = samples[0]
    if barrier_direction == 'Up':
        if optionality == "Out": #up-and-out call
            mcprice = np.exp(-r * total_time) * (np.maximum(stock[:, 0, -1] - strike, 0) @ (np.max(stock, axis=-1) < barrier)) / P
        if optionality == "In": #up-and-in call
            mcprice = np.exp(-r * total_time) * (np.maximum(stock[:, 0, -1] - strike, 0) @ (np.max(stock, axis=-1) > barrier)) / P
    if barrier_direction == 'Down':
        if optionality == "Out": #down-and-out call
            mcprice = np.exp(-r * total_time) * (np.maximum(stock[:, 0, -1] - strike, 0) @ (np.min(stock, axis=-1) > barrier)) / P
        if optionality == "In": #down-and-in call
            mcprice = np.exp(-r * total_time) * (np.maximum(stock[:, 0, -1] - strike, 0) @ (np.min(stock, axis=-1) < barrier)) / P

    print('mcprice is', mcprice[0]) 

    os.makedirs("Barrier/", exist_ok=True)
    np.save(os.path.join("Barrier/", "training_history_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_call"+"_S0_new_"+str(x_init), training_history) 
    np.save(os.path.join("Barrier/", "mcprice_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_call"+"_S0_new_"+str(x_init), mcprice) 
            
    # np.save("Barrier/training_history_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_call"+"_S0_new_"+str(x_init),training_history)
    # np.save("Barrier/mcprice_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_call"+"_S0_new_"+str(x_init), mcprice)
