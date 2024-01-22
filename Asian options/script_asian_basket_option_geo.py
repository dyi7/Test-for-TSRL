import numpy as np
import tensorflow as tf
from PureJumpSolver_v_avg import LUSolver
import AsianEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd
import random
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0' # specify which GPU(s) to be used
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)


############## Geometric Mean ###############


if __name__ == "__main__":
    dim = 100 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 32
    total_time = 1.0
    num_time_interval = 50
    strike = 0.9
    lamb = 0.3
    r = 0.04
    sigma = 0.25
    aver_jump = 0.4
    var_jump = 0.25
    x_init = 1.0
    config = {
                "eqn_config": {
                    "_comment": "a basket of asian options",
                    "eqn_name": "AsianBasketCallOption",
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
                    "avg_way":'Geometric', #'Arithmetic',
                    "p": 1 #Discrete Sampling Frequency
                },
                "net_config": {
                    "block_num": 5,                 
                    "num_hiddens": [dim+10, dim+10],  
                    "lr_boundaries": [5000, 10000, 15000, 20000],    
                    "lr_values": [5e-4, 1e-4, 2e-5, 4e-7, 8e-7],             
                    "num_iterations": 20000,          
                    "batch_size": batch_size,
                    "valid_size": 1000,            #change M trajectories
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "weight": 1,
                    "verbose": True
                }
                }
    # track hyperparameters and run metadata
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = LUSolver(config, bsde)
    training_history = bsde_solver.train()    
    
       
    #Simulate the BSDE after training - MtM scenarios
    samples = bsde.sample(P)  
    
    """
    Monte Carlo Price
    """
    stock = samples[0]
    """"""

    if config.eqn_config.avg_way == "Arithmetic":
        stock_avg = np.sum(stock, -1) / (num_time_interval + 1)  
    if config.eqn_config.avg_way == "Geometric":
        stock_avg = np.exp(np.sum(np.log(stock), -1) / (num_time_interval + 1))  
    mcprice = np.exp(-r* total_time)*np.average(np.maximum(np.sum(stock_avg, 1) - dim * strike, 0)) 

    print('mcprice is', mcprice)

    os.makedirs("Asian/", exist_ok=True)
    np.save(os.path.join("Asian/", "training_history_dim_"+str(dim)+"_seed_v1_"+str(seeds)+str(config.eqn_config.avg_way), training_history) 
    np.save(os.path.join("Asian/", "mcprice_dim_"+str(dim)+"_seed_v1_"+str(seeds)+str(config.eqn_config.avg_way), mcprice) 
            
    # np.save("Asian/training_history_dim_"+str(dim)+"_seed_v1_"+str(seeds)+str(config.eqn_config.avg_way),training_history) #loss
    # np.save("Asian/mcprice_dim_"+str(dim)+"_seed_v1_"+str(seeds)+str(config.eqn_config.avg_way), mcprice)

