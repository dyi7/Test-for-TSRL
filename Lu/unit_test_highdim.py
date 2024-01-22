import unittest

import munch
import random
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import PureJumpEquation as eqn
DELTA_CLIP = 50.0



class TestHighDimFunc(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for each component of Lu's equation (3.9), in 100-dimension experiment
        """
        super(TestHighDimFunc, self).__init__(*args, **kwargs)
        dim = 100 #dimension of brownian motion
        P = 2048*8 #number of outer Monte Carlo Loops
        batch_size = 32
        total_time = 1.0
        num_time_interval = 50   
        strike = 0.9
        lamb = 0.3
        r = 0.0
        sigma = 0.3  
        aver_jump = 0.1  
        var_jump = 0.25
        x_init = 1.0
        config = {
                    "eqn_config": {
                        "_comment": "High dimensional problems", 
                        "eqn_name": "HighDimEquation",   #Equations in (3.9)
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
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)
        self.valid_data = self.bsde.sample(self.config.net_config.valid_size)
        self.x, self.Poisson, self.jumps, self.dw = self.valid_data
        

    def test_f_tf(self):
        """
        Test the f(t, x, y, z) function equals -self.lamb*self.aver_jump**2 - self.sigma**2 + self.r*x**2
        """
        lamb = 0.3
        aver_jump = 0.1
        r = 0.0
        sigma = 0.3
        y = None
        z = None
        t = 0.2
        x_t = 54
        result = self.bsde.f_tf(t, x_t, y, z)  #-self.lamb*self.aver_jump**2 - self.sigma**2 
        self.assertEqual(result, -0.093) , "Should be -0.093"

    def test_g_tf(self):
        """
        Test u(T, x) as tf.reduce_sum(tf.pow(x,2), axis=1, keepdims=True)/self.dim #(M, dim)
        """
        total_time = tf.constant(1.0, tf.float64)
        dim = 100
        x_T = tf.ones([self.config.net_config.valid_size, 100], tf.float64)
        
        res =  tf.reduce_sum(tf.pow(x_T,2), axis=1, keepdims=True)/dim
        
        res2 = self.bsde.g_tf(total_time, x_T)
        result = tf.equal(sum(res), sum(res2))#tf.constant(500., tf.float64))
        self.assertTrue(result)

            
    def test_getFsdeDiffusion(self):
        """
        Test coeff of the gradient of u(t,x) is sigma
        """
        sigma = 0.3
        total_time = 1.0
        x_T = 12
        result = self.bsde.getFsdeDiffusion(total_time, x_T)
        self.assertEqual(result, sigma), "Should be 0.3"
        


class TestTempDiff(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        """
        Unit test for conponents of temporal difference learning  
        """
        super(TestTempDiff, self).__init__(*args, **kwargs)
        dim = 100 #dimension of brownian motion
        P = 2048*8 #number of outer Monte Carlo Loops
        batch_size = 32
        total_time = 1.0
        num_time_interval = 50   
        strike = 0.9
        lamb = 0.3
        r = 0.0
        sigma = 0.3  
        aver_jump = 0.1  
        var_jump = 0.25
        x_init = 1.0
        config = {
                    "eqn_config": {
                        "_comment": "High dimensional problems", 
                        "eqn_name": "HighDimEquation",   #Equations in (3.9)
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
        self.config = munch.munchify(config)
        self.bsde = getattr(eqn, self.config.eqn_config.eqn_name)(self.config.eqn_config)

    def test_reward(self):
        """
        Test calculation of Reward in Equation(2.18), Lu(2023)
        """
        time_t =  tf.constant([4, 5, 6], tf.float64)
        x_t =  tf.constant([1, 2, 3], tf.float64)
        dw_t =  tf.constant([0.1, 0.1, 0.1], tf.float64)
        N1_init = tf.constant([1, 1, 1], tf.float64)
        N1_N2_Jump = 5
        N1_grad =  tf.constant([1, 2, 3], tf.float64)
        reward = - self.bsde.delta_t * (self.bsde.f_tf(time_t, x_t, N1_init, N1_grad)) + tf.reduce_sum(N1_grad * self.bsde.getFsdeDiffusion(time_t, x_t) * dw_t) + N1_N2_Jump
        self.assertEqual(self.bsde.f_tf(time_t, x_t, N1_init, N1_grad), -0.093), "Should be -0.093"
        self.assertEqual(reward, 5.18186), "Should be 5.18186"

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



class TestOnePureLoss(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOnePureLoss, self).__init__(*args, **kwargs)
        """
        Unit test for the training loss of Lu(2023), Equation(2.25)
        """
        dim = 100 #dimension of brownian motion
        P = 2048*8 #number of outer Monte Carlo Loops
        batch_size = 32
        total_time = 1.0
        num_time_interval = 50   
        strike = 0.9
        lamb = 0.3
        r = 0.0
        sigma = 0.3  
        aver_jump = 0.1  
        var_jump = 0.25
        x_init = 1.0
        config = {
                    "eqn_config": {
                        "_comment": "High dimensional problems", 
                        "eqn_name": "HighDimEquation",   #Equations in (3.9)
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
        dim = 100 #dimension of brownian motion
        P = 2048*8 #number of outer Monte Carlo Loops
        batch_size = 32
        total_time = 1.0
        num_time_interval = 50   
        strike = 0.9
        lamb = 0.3
        r = 0.0
        sigma = 0.3  
        aver_jump = 0.1  
        var_jump = 0.25
        x_init = 1.0
        config = {
                    "eqn_config": {
                        "_comment": "High dimensional problems", 
                        "eqn_name": "HighDimEquation",   #Equations in (3.9)
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




