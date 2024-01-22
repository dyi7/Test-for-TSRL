import unittest

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PureJumpSolver_v import LUSolver  
import BarrierEquation as eqn
import munch
import pandas as pd
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # specify which GPU(s) to be used
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)



class TestIntegUpOutPut(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Integration test for the up-and-out put options
        """
        super(TestIntegUpOutPut, self).__init__(*args, **kwargs)
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
                        "_comment": "barrier put option",  #put options
                        "eqn_name": "BarrierPutOption",
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
                        "optionality": optionality,
                    },
                    "net_config": {
                        "block_num": 5, 
                        "num_hiddens": [25, 25],  
                        "lr_values": [5e-4, 1e-4, 2e-5, 4e-7],  
                        "lr_boundaries":[10000, 15000, 20000] ,
                        "num_iterations": 25000, 
                        "batch_size": batch_size,
                        "valid_size": 1000,
                        "logging_frequency": 100,
                        "dtype": "float64",
                        "weight": 1,
                        "verbose": True
                    }
                    }
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)
        self.valid_data = self.bsde.sample(self.config.net_config.valid_size)
        self.x, self.Poisson, self.jumps, self.dw = self.valid_data

    def test_error(self):
        """
        Test if the relative error is within the order of 10^(-4) in pricing of Up-and-out put option 
        """
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.backend.set_floatx(self.config.net_config.dtype)

         
        model = LUSolver(self.config, self.bsde).model
        model.load_weights('model_zoo/upoutput/Weights')
        print('Model Loaded!')
        # training_history = bsde_solver.train() 
    
        valid_data = self.bsde.sample(self.config.net_config.valid_size)
        print('Predicting...')
        y_init = model.simulate_path(valid_data)[0, 0, 0]   
        mcprice = np.load('model_zoo/mcprice_dim_1_Up_Out_put__S0_1.0.npy',allow_pickle=True)
        print()
        print('*'*80)
        print('The fitted value is', y_init)
        print('The MC value is', mcprice[0])
        abs_error = abs(y_init - mcprice[0]) 
        print("The absolute error is", abs_error)
        print()

        self.assertTrue(abs_error < 1e-3), "Absolute error should be in the order of O(10^(-4))"
        print("Under the error tolerence in the order of O(10^(-4)), we consider the fitted value is the same with the solution to Up-and-Out put option")

if __name__ == '__main__':
    unittest.main()



