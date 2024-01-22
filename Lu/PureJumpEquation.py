from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal 
import random
    

class OnePure(Equation):
    ### Lu(23), Equation (3.2)
    def __init__(self,eqn_config):
        super(OnePure, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = True # whether to use explicit formula to evaluate dynamics of x
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump
        
    def sample(self, num_sample):  ## = 1000
        
        # Brownian simulation
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t
        if num_sample==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # Jump simulation
        eta = normal.rvs(mean=0.0 ,cov=1.0, size = [num_sample, self.dim, self.num_time_interval])
        eta = np.reshape(eta,[num_sample, self.dim, self.num_time_interval])
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        # forward trajectory
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
  
    def f_tf(self, t, x, y, z):
         return 0
   
    def g_tf(self, t, x):
        return x   
    
    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
        dg_x = tape.gradient(g, x_tensor)
        del tape
        return dg_x

    def getFsdeDiffusion(self, t, x):
        return 0



class Equation2(Equation):
    ### Lu(23), Equation (3.5)
    def __init__(self,eqn_config):
        super(Equation2, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = True # whether to use explicit formula to evaluate dynamics of x
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        self.var_jump = eqn_config.var_jump
        
    def sample(self, num_sample):  ## = 1000
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
        jumps = np.multiply(Poisson, self.aver_jump) + np.sqrt(self.var_jump)*np.multiply(np.sqrt(Poisson),eta)
        
        # forward trajectory
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(np.exp(self.aver_jump + 0.5*self.var_jump)-1)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
  
    def f_tf(self, t, x, y, z):
         return -self.r * x
   
    def g_tf(self, t, x):
        return x   
    
    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
        dg_x = tape.gradient(g, x_tensor)
        del tape
        return dg_x

    def getFsdeDiffusion(self, t, x):
        return self.sigma  #0

    
class HighDimEquation(Equation): #Equation 3
    ### Lu(23), Equation (3.9)
    def __init__(self,eqn_config):
        super(HighDimEquation, self).__init__(eqn_config)
        #self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x 
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = True # whether to use explicit formula to evaluate dynamics of x
        self.lamb = eqn_config.lamb
        self.aver_jump = eqn_config.aver_jump
        #self.var_jump = eqn_config.var_jump
        
    def sample(self, num_sample):  ## = 1000
        # Brownian simulation
        dw_sample = normal.rvs(size=[num_sample,     
                                      self.dim,
                                      self.num_time_interval]) * self.sqrt_delta_t  
        if num_sample==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)
        
        # Jump simulation
        eta = self.aver_jump * np.ones((num_sample, self.dim, self.num_time_interval))  # 0.1
        Poisson = np.random.poisson(self.lamb * self.delta_t, [num_sample, self.dim , self.num_time_interval])
        jumps = np.multiply(Poisson, eta)
        
        # forward trajectory
        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: 
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t - self.lamb*(self.aver_jump)*self.delta_t)
            for i in range(self.num_time_interval): 
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i]) * np.exp(jumps[:, :, i])) * x_sample[:, :, i]
        return x_sample, Poisson, jumps, dw_sample  
  
    def f_tf(self, t, x, y, z):
         return -self.lamb*self.aver_jump**2 - self.sigma**2 
   
    def g_tf(self, t, x):
        return tf.reduce_sum(tf.pow(x,2), axis=1, keepdims=True)/self.dim #(M, dim)

    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
            #g_tensor = tf.convert_to_tensor(g, dtype=tf.float64)
        dg_x = tape.gradient(g, x_tensor)
        del tape
        return dg_x

    def getFsdeDiffusion(self, t, x):
        return self.sigma     





        
        
