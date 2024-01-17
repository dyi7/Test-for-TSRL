import numpy as np
import tensorflow as tf
from PureJumpSolver_v import LUSolver
import AmericanEquation as eqn
import munch
from scipy.stats import norm
import random
import wandb
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)



if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048*8 #number of outer Monte Carlo Loops
    batch_size = 32
    total_time = 1.0
    num_time_interval = 50
    strike = 1.0
    lamb = 0.3
    r = 0.04
    sigma = 0.25
    aver_jump = 0.4
    var_jump = 0.25
    x_init = 1.0
    barrier = 1.3
    barrier_direction = 'Up' # Down or Up
    optionality = 'Out'  # Out = Knock Out Option   In = Knock In Option  ###
    config = {
                "eqn_config": {
                    "_comment": "American-barrier put option",
                    "eqn_name": "AmericanBarrierPutOption",  # American-style up-and-out put options
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
                    "block_num": 5, 
                    "num_hiddens": [25, 25],  
                    "lr_values": [5e-4, 1e-4, 2e-5, 4e-6],  
                    "lr_boundaries": [10000, 15000, 20000],  
                    "num_iterations": 25000,  
                    "batch_size": batch_size,
                    "valid_size": 1000,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "weight": 1,
                    "verbose": True
                }
                }
    # track hyperparameters and run metadata
    wandb.init(project="Morgan Project", config = config)
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
    H = np.squeeze(stock) 
    if barrier_direction == 'Up':
        if optionality == "Out": #up-and-out put
            mcprice = np.exp(-r * total_time) * (np.maximum(strike - stock[:, 0, -1], 0) @ (np.max(stock, axis=-1) < barrier)) / P
            H_sample = np.where(np.max(H, -1, keepdims=True) < barrier, np.maximum(strike - H, 0), 0)

    print('European-style price is', mcprice[0])  #0.15595926

    def LSM(H):
        V = np.zeros_like(H)  # value matrix
        V[:, -1] = H[:, -1]
        # Valuation by LS Method
        for t in range(num_time_interval - 1, 0, -1):  
            df = np.exp(-r * bsde.delta_t) # discount factor
            good_paths = H[:, t] > 0  # paths where the intrinsic value is positive
            # the regression is performed only on these paths
            rg = np.polyfit(H[good_paths, t], V[good_paths, t + 1] * df, 2)  # polynomial regression
            C = np.polyval(rg, H[good_paths, t])  # evaluation of regression
            exercise = np.zeros(len(good_paths), dtype=bool)  # initialize
            exercise[good_paths] = H[good_paths, t] > C  # paths where it is optimal to exercise
            V[exercise, t] = H[exercise, t]  # set V equal to H where it is optimal to exercise
            V[exercise, t + 1 :] = 0  # set future cash flows, for that path, equal to zero
            discount_path = V[:, t] == 0  # paths where we didn't exercise
            V[discount_path, t] = V[discount_path, t + 1] * df  # set V[t] in continuation region
        val = np.expand_dims(V[:, 1] * df, -1)
        V0 = np.mean(V[:, 1]) * df  # discounted expectation of V[t=1]
        return val, V0
    
    _, ame_price = LSM(H_sample)

    print('American-style price is', ame_price) #[0.15598534]

    np.save("Barrier/American/training_history_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_put_ameri_v12",training_history)
    np.save("Barrier/American/stock_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_put_ameri_v12", stock)
    np.save("Barrier/American/european_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_put_ameri_v12", mcprice)
    np.save("Barrier/American/american_dim_"+str(dim)+"_"+str(barrier_direction)+"_"+str(optionality)+"_put_ameri_v12", ame_price)
