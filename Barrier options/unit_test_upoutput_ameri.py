import unittest

import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import AmericanEquation as eqn
import munch
import random
import tensorflow as tf
import numpy as np
DELTA_CLIP = 50.0




class TestUpOutPutFunc(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for each component of Equation for Up-and-out put options  
        """
        super(TestUpOutPutFunc, self).__init__(*args, **kwargs)
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
                        "optionality": optionality,
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
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)
        

    def test_f_tf(self):
        """
        Test the f(t, x, y, z) function should be -self.r * y
        """
        r = 0.04
        t = 0.2
        x_t = 54
        y = 8
        z = None
        result = self.bsde.f_tf(t, x_t, y, z) #0
        self.assertEqual(result, -r*y), "Should be -0.32"  


    def test_g_tf(self):
        """
        Test u(T, x)
        """
        total_time = 1.0
        num_time_interval = 50
        M = 1000
        dim = 1
        strike = 1.0
        barrier = 1.3
        x_T = tf.ones([M, dim, (num_time_interval+1)], tf.float64)

        
        max_value = tf.reduce_max(tf.squeeze(x_T), axis=-1, keepdims = True)  
        min_value = tf.reduce_min(tf.squeeze(x_T), axis=-1, keepdims = True)  
        if self.config.eqn_config.barrier_direction == 'Up':
            if self.config.eqn_config.optionality == "Out":
                H_sample = tf.reduce_sum((tf.maximum(strike - tf.squeeze(x_T), 0) * tf.cast(max_value < barrier, dtype=tf.float64))) 
        
        self.assertEqual(H_sample, 0.0), "Should be the same"

            
    def test_getFsdeDiffusion(self):
        """
        Test coeff of the gradient of u(t,x) is sigma*x
        """
        sigma = 0.25
        total_time = 1.0
        x_T = 4
        result = self.bsde.getFsdeDiffusion(total_time, x_T)
        self.assertEqual(result, sigma*x_T), "Should be 1.0"

    
        
class TestLSM(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for conponents of Longstaff-Schwartz Method  
        """
        super(TestLSM, self).__init__(*args, **kwargs)
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
                        "optionality": optionality,
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
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)
        
    def test_discount_factor(self):
        """
        Test discount factor
        """
        r = 0.04
        delta_t = 0.02
        df = np.exp(-r * delta_t)
        self.assertEqual(round(df, 4), 0.9992), "Should be 0.9992"


    def test_least_square(self):
        """
        Test polynomial regression for coefficients (beta) estimation
        """
        x = np.array([1.0])
        y = np.array([0.6])
        rg = np.polyfit(x, y, 2)
        self.assertEqual(y[0], round(rg[2]*x[0]**2 + rg[1]*x[0] + rg[0], 1))

        

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
                        "optionality": optionality,
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
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)

    def test_reward(self):
        """
        Test calculation of Reward in Equation(2.18), Lu(2023)
        """
        r = 0.04
        time_t =  tf.constant([4, 5, 6], tf.float64)
        x_t =  tf.constant([1, 2, 3], tf.float64)
        N1_grad =  tf.constant([1, 2, 3], tf.float64)
        N1_init = tf.constant([1, 1, 1], tf.float64)
        self.assertEqual(sum(self.bsde.f_tf(time_t, x_t, N1_init, N1_grad)), sum(tf.constant([-0.04, -0.04, -0.04], tf.float64))), "Should be [-0.04, -0.04, -0.04]"
        
        N1_N2_Jump = 5
        dw_t =  tf.constant([0.1, 0.1, 0.1], tf.float64)
        reward = - self.bsde.delta_t * (self.bsde.f_tf(time_t, x_t, N1_init, N1_grad)) + tf.reduce_sum(N1_grad * self.bsde.getFsdeDiffusion(time_t, x_t) * dw_t) + N1_N2_Jump
        self.assertEqual(sum(reward), sum(tf.constant([5.3508, 5.3508, 5.3508], tf.float64))), "Should be [5.3508 5.3508 5.3508]"

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
                        "optionality": optionality,
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
        Unit test for the process of simulating option prices 
        """
        super(TestPriceSimulation, self).__init__(*args, **kwargs)
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
                        "optionality": optionality,
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

