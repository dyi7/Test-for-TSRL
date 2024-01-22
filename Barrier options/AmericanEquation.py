from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal
#import tensorflow.experimental.numpy as tnp


##################### use LSM_tensorflow_1

tf.config.experimental_run_functions_eagerly(True)

class AmericanBarrierPutOption(Equation):
    def __init__(self, eqn_config):
        super(AmericanBarrierPutOption, self).__init__(eqn_config)
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

    def LSM(self, H):
        V = np.zeros_like(H)  # value matrix
        V[:, -1] = H[:, -1]
        
        # Valuation by LS Method
        for t in range(self.num_time_interval - 1, 0, -1):  
            df = np.exp(-self.r * self.delta_t) # discount factor
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
        val = tf.convert_to_tensor(np.expand_dims(V[:, 1] * df, -1),dtype=tf.float64)
        V0 = np.mean(V[:, 1]) * df  # discounted expectation of V[t=1]
        return val, V0
    
    def f_tf(self, t, x, y, z):
        return -self.r * y
    
    def g_tf(self, t, x):
        
        max_value = tf.reduce_max(tf.squeeze(x), axis=-1, keepdims = True)  
        min_value = tf.reduce_min(tf.squeeze(x), axis=-1, keepdims = True)  
        if self.barrier_direction == 'Up':
            if self.optionality == "Out":
                H_sample = (tf.maximum(self.strike - tf.squeeze(x), 0) * tf.cast(max_value < self.barrier, dtype=tf.float64)) 
                H_sample1 = tf.make_ndarray(tf.make_tensor_proto(H_sample))

        payoff, _ = self.LSM(H_sample1)
        
        return payoff  
   

    def g_grad_tf(self, t, x):
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        with tf.GradientTape() as tape:
            tape.watch(x_tensor)
            g = self.g_tf(t, x_tensor)
            dg_x = tape.gradient(g, x_tensor, unconnected_gradients=tf.UnconnectedGradients.ZERO)  
        del tape
        return dg_x    
    
    def getFsdeDiffusion(self, t, x):
        return self.sigma * x 


