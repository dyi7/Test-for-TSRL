from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal


### Barrier put options ###

class BarrierPutOption(Equation):
    def __init__(self, eqn_config):
        super(BarrierPutOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike  
        self.x_init = np.ones(self.dim) * eqn_config.x_init  
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump     
        self.useExplict = True #whether to use explict formula to evaluate dyanamics of x
        self.barrier = eqn_config.barrier
        self.barrier_direction = eqn_config.barrier_direction  # Down or Up
        self.optionality = eqn_config.optionality  #Out = Knock Out Option   In = Knock In Option
        

    def sample(self, num_sample):  
        
        # Brownian simulation
        
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t  
               
        if self.dim==1:  
            dw_sample = np.expand_dims(dw_sample,axis=0) 
            dw_sample = np.swapaxes(dw_sample,0,1) 
        
        # Jump simulation
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])  
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval]) # eta ~ N(0,1)
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])  
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson), eta)
        
        # forward trajectory

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])  
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  
        
        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]

        return x_sample, Poisson, jumps, dw_sample
    
    def f_tf(self, t, x, y, z):
        return -self.r * y
    
    def g_tf(self, t, x):
        max_value = tf.reduce_max(x, axis=-1)  
        min_value = tf.reduce_min(x, axis=-1) 
        if self.barrier_direction == 'Up':
            if self.optionality == "Out":
                payoff = tf.maximum(self.strike - x[:, :, -1], 0) * tf.cast(max_value < self.barrier, dtype=tf.float64)
            if self.optionality == "In":
                payoff = tf.maximum(self.strike - x[:, :, -1], 0) * tf.cast(max_value > self.barrier, dtype=tf.float64)
        if self.barrier_direction == 'Down':
            if self.optionality == "Out":
                payoff = tf.maximum(self.strike - x[:, :, -1], 0) * tf.cast(min_value > self.barrier, dtype=tf.float64)
            if self.optionality == "In":
                payoff = tf.maximum(self.strike - x[:, :, -1], 0) * tf.cast(min_value < self.barrier, dtype=tf.float64)
        return payoff 

    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
        dg_x = tape.gradient(g, x_tensor)
        del tape
        return dg_x    
    
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x   



### Barrier call options ###
    
class BarrierCallOption(Equation):
    def __init__(self, eqn_config):
        super(BarrierCallOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike  
        self.x_init = np.ones(self.dim) * eqn_config.x_init  
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump     
        self.useExplict = True # whether to use explict formula to evaluate dyanamics of x
        self.barrier = eqn_config.barrier
        self.barrier_direction = eqn_config.barrier_direction  # Down or Up
        self.optionality = eqn_config.optionality  #Out = Knock Out Option   In = Knock In Option
        

    def sample(self, num_sample):  
        
        # Brownian simulation
        
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t  
               
        if self.dim==1:  
            dw_sample = np.expand_dims(dw_sample,axis=0)  
            dw_sample = np.swapaxes(dw_sample,0,1)  
        
        # Jump simulation
    
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])  
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])  
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])  
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson), eta)
        
        # forward trajectory

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1])  
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  
        
        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
                
        return x_sample, Poisson, jumps, dw_sample
    
    def f_tf(self, t, x, y, z):
        return -self.r * y

    def g_tf(self, t, x):
        max_value = tf.reduce_max(x, axis=-1)  
        min_value = tf.reduce_min(x, axis=-1)  
        if self.barrier_direction == 'Up':
            if self.optionality == "Out":
                payoff = tf.maximum(x[:, :, -1] - self.strike, 0) * tf.cast(max_value < self.barrier, dtype=tf.float64)
            if self.optionality == "In":
                payoff = tf.maximum(x[:, :, -1] - self.strike, 0) * tf.cast(max_value > self.barrier, dtype=tf.float64)
        if self.barrier_direction == 'Down':
            if self.optionality == "Out":
                payoff = tf.maximum(x[:, :, -1] - self.strike, 0) * tf.cast(min_value > self.barrier, dtype=tf.float64)
            if self.optionality == "In":
                payoff = tf.maximum(x[:, :, -1] - self.strike, 0) * tf.cast(min_value < self.barrier, dtype=tf.float64)
        return payoff  
    
    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
        dg_x = tape.gradient(g, x_tensor)
        del tape
        return dg_x
        
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x   






