import unittest

import munch
import random
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import PureJumpEquation as eqn
DELTA_CLIP = 50.0



class TestEqn2Func(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for each component of Lu's equation (3.5), in robustness experiment
        """
        super(TestEqn2Func, self).__init__(*args, **kwargs)
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
        

    def test_f_tf(self):
        """
        Test the f(t, x, y, z) function equals -self.r * x
        """
        r = 0.25
        y = None
        z = None
        t = 0.2
        x_t = 10
        result = self.bsde.f_tf(t, x_t, y, z)  
        self.assertEqual(result, -2.5) , "Should be -2.5"

    def test_g_tf(self):
        """
        Test u(T, x) as u(T, x) = x
        """
        total_time = tf.constant(1.0, tf.float64)
        x_T = tf.constant(78.0, tf.float64)
        result = self.bsde.g_tf(total_time, x_T)
        self.assertEqual(result, x_T), "Should be 78"

            
    def test_getFsdeDiffusion(self):
        """
        Test coeff of the gradient of u(t,x) is sigma
        """
        sigma = 0.0
        total_time = 1.0
        x_T = 12
        result = self.bsde.getFsdeDiffusion(total_time, x_T)
        self.assertEqual(result, sigma), "Should be 0.0"
        


class TestTempDiff(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for conponents of temporal difference learning  
        """
        super(TestTempDiff, self).__init__(*args, **kwargs)
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

    def test_reward(self):
        """
        Test calculation of Reward in Equation(2.18), Lu(2023)
        """
        delta_t = 0.2
        f_tf = tf.constant([-0.25, -0.5,  -0.75], tf.float64)
        getFsdeDiffusion = 0.0
        N1_grad = tf.constant([4, 5, 6], tf.float64)
        dw_t = tf.constant([1, 2, 3], tf.float64)
        N1_N2_Jump = 5
        N1_grad =  tf.constant([1, 2, 3], tf.float64)
        
        reward = -delta_t * (f_tf + tf.reduce_sum(N1_grad * getFsdeDiffusion * dw_t)) + N1_N2_Jump
        self.assertEqual(sum(reward), sum(tf.constant([5.05, 5.1 , 5.15], tf.float64))), "Should be [5.05, 5.1 , 5.15]"

    def test_td_loss(self):
        """
        Test calculation of Loss 1 (TD_error) in Equation(2.20), Lu(2023) 
        """
        N1_init = tf.constant([1, 1, 1], tf.float64)
        N1 = tf.constant([2, 2, 2], tf.float64)
        reward = tf.constant([5, 5, 5], tf.float64)
        td_error = reward +  N1_init - N1
        loss1 = tf.reduce_mean(tf.square(td_error))
        self.assertEqual(loss1, 16), "Should be 16"

    
    def test_value_func_update(self):
        """
        Test update of the value function in Equation(2.19), Lu(2023) 
        """
        N1_init = tf.constant([1, 1, 1], tf.float64)
        learning_rate = tf.constant([0.1, 0.1, 0.1], tf.float64)
        td_error = tf.constant([5, 5, 5], tf.float64)
        N1_init = N1_init + learning_rate * td_error 
        self.assertEqual(sum(N1_init), sum(tf.constant([1.5, 1.5, 1.5], tf.float64))), "Should be [1.5, 1.5, 1.5]"


    def test_loss4(self):
        """
        Test calculation of Loss 4 of Lu(2023) 
        """
        N1_N2_Jump = tf.constant([5, 5, 5], tf.float64)
        loss4 = tf.reduce_mean(tf.abs(N1_N2_Jump))
        self.assertEqual(loss4, 5), "Should be 5"



class TestLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestLoss, self).__init__(*args, **kwargs)
        """
        Unit test for the training loss of Lu(2023), Equation(2.25)
        """
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
    

    def test_loss2(self):
        """
        Test calculation of loss 2 in Lu(2023)
        """
        diff_2 = tf.constant([1, 8, 17], tf.float64)
        loss2 = tf.reduce_mean(tf.where(tf.abs(diff_2) < DELTA_CLIP, tf.square(diff_2),
                                    2 * DELTA_CLIP * tf.abs(diff_2) - DELTA_CLIP ** 2))
        self.assertEqual(loss2, 118), "Should be 118"


    def test_loss3(self):
        """
        Test calculation of loss 3 in Lu(2023)
        """
        diff_3 = tf.constant([1, 8, 17], tf.float64)
        loss3 = tf.reduce_mean(tf.where(tf.abs(diff_3) < DELTA_CLIP, tf.square(diff_3),
                                    2 * DELTA_CLIP * tf.abs(diff_3) - DELTA_CLIP ** 2))
        self.assertEqual(loss3, 118), "Should be 118"

    
    def test_total_loss(self):
        """
        Test calculation of the total loss in Lu(2023)
        """
        td_loss = tf.constant([2, 9, 19], tf.float64) 
        loss2 = tf.constant([2, 9, 19], tf.float64)
        loss3 = tf.constant([2, 9, 19], tf.float64)
        loss4 = tf.constant([2, 9, 19], tf.float64)
        loss = td_loss + loss2 + loss3 + loss4
        self.assertEqual(sum(loss), sum(tf.constant([8., 36., 76.], tf.float64))), "Should be [8., 36., 76.]"



class TestPriceSimulation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for components in the process of simulating option prices 
        """
        super(TestPriceSimulation, self).__init__(*args, **kwargs)
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

    def test_possion(self):
        """
        Test Poisson with intensity in simulating option prices 
        """
        lamb = 0.5
        delta_t = 0.2
        Poisson =  lamb * delta_t 
        self.assertEqual(Poisson, 0.1) , "Should be 0.1"

    def test_jump(self):
        """
        Test calculation of jumps in simulating option prices 
        """
        eta = 0.5
        Poisson = 0.01
        aver_jump = 0.4
        var_jump = 0.25
        
        sqrt_operator = np.sqrt(var_jump)
        self.assertEqual(sqrt_operator, 0.5) , "Should be 0.5"
        
        mul_operator1 = np.multiply(sqrt_operator, eta)
        self.assertEqual(mul_operator1, 0.25) , "Should be 0.25"
        
        mul_operator2 = np.multiply(Poisson, aver_jump)
        self.assertEqual(mul_operator2, 0.004) , "Should be 0.004"
        

        add_operator = mul_operator1 + mul_operator2
        jumps = add_operator
        self.assertEqual(jumps, 0.254) , "Should be 0.254"
        

    def test_price_sim(self):
        """
        Test the process of simulating option prices 
        """
        r = 0.0
        sigma = 0.25
        delta_t = 0.2

        coeff = (r-(sigma**2)/2) * delta_t
        self.assertEqual(coeff, -0.00625) , "Should be -0.00625"

        z = 1.0
        G_z = np.exp(z)-1
        self.assertEqual(round(G_z,3), 1.718), "Should be 1.718"



if __name__ == '__main__':
    unittest.main()




