import numpy as np
import tensorflow as tf
from PureJumpSolver_v import LUSolver   
import PureJumpEquation as eqn
import munch
from scipy.stats import norm
import pandas as pd
import random
# import wandb
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '1' # specify which GPU(s) to be used
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)


####### Experiments in Lu (2023), Section 3.2. Robustness, 1)  Trajectories and intervals
####### iterations = 250 
####### N = 20
####### M = 500


if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 32
    total_time = 1.0
    num_time_interval = 20  #change intervals 
    strike = 0.9
    lamb = 0.3
    r = 0.25
    sigma = 0.0
    aver_jump = 0.4   
    var_jump = 0.25
    x_init = 1
    config = {
                "eqn_config": {
                    "_comment": "Equation 2 Trajectories and intervals",  
                    "eqn_name": "Equation2",    
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
                },
                "net_config": {
                    "block_num": 5,                  
                    "num_hiddens": [25*dim, 25*dim],  
                    "lr_boundaries": [5000, 10000, 15000, 20000],    
                    "lr_values": [5e-4, 1e-4, 2e-5, 4e-7, 8e-7],                  
                    "num_iterations": 5000,          #change 5000 = 250 *20 (num_time_interval)
                    "batch_size": batch_size,
                    "valid_size": 500,            #change M trajectories
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
    bsde_solver = LUSolver(config, bsde)
    training_history = bsde_solver.train()  
    
   
    #Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))

    os.makedirs("robustness/", exist_ok=True)
    np.save(os.path.join("robustness/", "training_history_dim_"+str(dim)+"_M_"+str(config.net_config.valid_size)+"_N_"+str(num_time_interval)+"_iteration_"+str(config.net_config.num_iterations/num_time_interval), training_history) 
    #np.save("robustness/training_history_dim_"+str(dim)+"_M_"+str(config.net_config.valid_size)+"_N_"+str(num_time_interval)+"_iteration_"+str(config.net_config.num_iterations/num_time_interval),training_history)  #loss
    
    
   
