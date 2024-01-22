import unittest

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from PureJumpSolver_v import LUSolver  
import PureJumpEquation as eqn
import munch
import pandas as pd
import random


os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # specify which GPU(s) to be used
seeds = 2023
random.seed = seeds  
np.random.seed(seeds)  
tf.random.set_seed(seeds)

####### Experiments in Lu (2023), Section 3.2. Robustness, 1)  Trajectories and intervals
####### iterations = 250 
####### N = 20
####### M = 125


class TestIntegOnePure(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Integrated test for robust experiment
        when iteration = 250, M = 125 and N = 20
        """
        super(TestIntegOnePure, self).__init__(*args, **kwargs)
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
                        "valid_size": 125,            #change M trajectories
                        "logging_frequency": 100,
                        "dtype": "float64",
                        "verbose": True
                    }
                    }
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)
        self.valid_data = self.bsde.sample(self.config.net_config.valid_size)
        self.x, self.Poisson, self.jumps, self.dw = self.valid_data

    def test_error(self):
        """
        Test if the relative error is within the order of 10^(-3) when change different M and N
        """
        gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.backend.set_floatx(self.config.net_config.dtype)

         
        model = LUSolver(self.config, self.bsde).model
        model.load_weights('model_zoo/robust/Weights')
        print('Model Loaded!')
        # training_history = bsde_solver.train() 
    
        valid_data = self.bsde.sample(self.config.net_config.valid_size)
        print('Predicting...')
        y_init = model.simulate_path(valid_data)[0, 0, 0]  # 1.0001029828280779
        print()
        print('*'*80)
        print('The fitted value is', y_init)
        rela_error = (y_init - self.config.eqn_config.x_init)/self.config.eqn_config.x_init
        print("The relative error is ", rela_error)

        self.assertTrue(rela_error < 1e-2), "Relative error be in the order of O(10^(-3))"
        print("Under the error tolerence in the order of O(10^(-3)), we consider the fitted value is the same with the solution when M=125, N=20 on 250 iterations.")

if __name__ == '__main__':
    unittest.main()

    


    
   
