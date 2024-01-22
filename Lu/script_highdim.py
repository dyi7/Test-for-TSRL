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


####### Experiments in Lu (2023), Section 3.3. High dimensional problems.


if __name__ == "__main__":
    dim = 100 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 32
    total_time = 1.0
    num_time_interval = 50   
    #strike = 0.9
    lamb = 0.3
    r = 0.0
    sigma = 0.3  
    aver_jump = 0.1  
    # var_jump = 0.25
    x_init = 1.0
    config = {
                "eqn_config": {
                    "_comment": "High dimensional problems", 
                    "eqn_name": "HighDimEquation",   #Equations in (3.9)
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    #"strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "lamb":lamb,
                    "aver_jump":aver_jump,
                    # "var_jump":var_jump,
                    "x_init":x_init,

                },
                "net_config": {
                    "block_num": 5,                 
                    "num_hiddens": [dim+10, dim+10],  
                    "lr_boundaries": [10000, 20000], 
                    "lr_values": [5e-4, 1e-4, 2e-5],           
                    "num_iterations": 30000,          
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

    os.makedirs("Highdim/", exist_ok=True)
    np.save(os.path.join("Highdim/", "training_history_dim_"+str(dim)+"_seed_"+str(seeds)+"_highdim"), training_history)
    #np.save("Highdim/training_history_dim_"+str(dim)+"_seed_"+str(seeds)+"_highdim",training_history)  #loss
    
   
