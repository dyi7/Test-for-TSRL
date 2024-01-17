import logging
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l2
from scipy.stats import multivariate_normal as normal
import wandb
import random
DELTA_CLIP = 50.0



class LUSolver(object):
    """The fully connected neural network model."""
    def __init__(self, config, bsde):
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde

        try:
            lr_schedule = config.net_config.lr_schedule
        except AttributeError:
            lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.net_config.lr_boundaries, self.net_config.lr_values)     
            
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, epsilon=1e-8)
        
        self.lrate = tf.keras.backend.get_value(self.optimizer.lr(self.optimizer.iterations))

        self.model = NonsharedModel(config, bsde, self.lrate)

    def train(self):
        start_time = time.time()
        training_history = []
        valid_data = self.bsde.sample(self.net_config.valid_size)

        # begin sgd iteration
        for step in tqdm(range(self.net_config.num_iterations+1)):
            if step % self.net_config.logging_frequency == 0:
                loss, tdloss, loss_2, loss_3, loss_4 = self.loss_fn(valid_data, training=False)
                loss = loss.numpy() 
                y_pred = self.model.simulate_path(valid_data) 
                y_init = self.model.simulate_path(valid_data)[0,0,0] # or y_init = self.model.subnet[0].call(valid_data[0][:,:,0],False)[0].numpy()[0]
                relative_error = tf.abs(y_init-self.bsde.x_init)/self.bsde.x_init
                elapsed_time = time.time() - start_time
                training_history.append([step, loss, y_init, y_pred, elapsed_time])
                # log metrics to wandb
                wandb.log({"loss": loss, "Y0": y_init,"relative_error": relative_error, "td_loss": tdloss.numpy(), "loss2": loss_2.numpy(), "loss3": loss_3.numpy(), "loss4": loss_4.numpy()})
                if self.net_config.verbose:
                    print("step: %5u,    loss: %.4e, Y0: %.4e,  lrate: %5e,  elapsed time: %3u" % (
                        step, loss, y_init, self.lrate, elapsed_time))
            #print('self.bsde.sample(self.net_config.batch_size)', self.bsde.sample(self.net_config.batch_size)[1].dtype)
            self.train_step(self.bsde.sample(self.net_config.batch_size))    #int64
            
        wandb.finish()
        return np.array(training_history)
   
    def loss_fn(self, inputs, training):  
        x, Poisson, jumps, dw = inputs
        """update"""
        N1_terminal, N1_t_average, td_loss, loss4 = self.model(inputs, training)
        N1_terminal_grad = self.model.grad(inputs)  
        # #print('    N1_terminal_grad', N1_terminal_grad) 
        diff_2 = N1_t_average - self.bsde.g_tf(self.bsde.total_time, x)
        loss2 = tf.reduce_mean(tf.where(tf.abs(diff_2) < DELTA_CLIP, tf.square(diff_2),
                                    2 * DELTA_CLIP * tf.abs(diff_2) - DELTA_CLIP ** 2))
        diff_3 = N1_terminal_grad - self.bsde.g_grad_tf(self.bsde.total_time, x)  
        #print(N1_terminal_grad.shape, self.bsde.g_grad_tf(self.bsde.total_time, x[:, :, -1]).shape)
        loss3 = tf.reduce_mean(tf.where(tf.abs(diff_3) < DELTA_CLIP, tf.square(diff_3),
                                    2 * DELTA_CLIP * tf.abs(diff_3) - DELTA_CLIP ** 2))  

        loss = td_loss + loss2 + loss3 + loss4
        print('*'*80)
        print("td_loss, loss2, loss3, loss4:", td_loss, loss2, loss3, loss4)
        print('*'*80)
        return loss, td_loss, loss2, loss3, loss4
    
    def grad(self, inputs, training):
        with tf.GradientTape(persistent=True) as tape:
            loss,_,_,_,_ = self.loss_fn(inputs, training) 
            
        grad = tape.gradient(loss, self.model.trainable_variables)
        del tape
        return grad
    
    @tf.function
    def train_step(self, train_data):
        grad = self.grad(train_data, training=True)
        self.optimizer.apply_gradients(zip(grad, self.model.trainable_variables))     
        


class NonsharedModel(tf.keras.Model):
    def __init__(self, config, bsde, rate):
        super(NonsharedModel, self).__init__()
        self.config = config
        self.eqn_config = config.eqn_config
        self.net_config = config.net_config
        self.bsde = bsde       
        self.dim = bsde.dim       
        self.subnet = [ResNet(config, bsde.dim) for _ in range(self.bsde.num_time_interval+1)]  
        self.learning_rate = rate

    
    @tf.function  
    def call(self, inputs, training):   #Training = True
        x, Poisson, jumps, dw = inputs
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        N1_init = all_one_vec * self.subnet[0].call(all_one_vec * 0, all_one_vec * self.bsde.x_init , training)[0]
        
        history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
        history = history.write(0,N1_init)
        
        """add loss_1, loss_4"""
        loss_1 = 0.0  
        loss_4 = 0.0
        
        """ Following the pseudo-code of page 8, t-1"""
        for t in range(0, self.bsde.num_time_interval):  
            
            """input: x[:, :, t ], t"""
            input_t = all_one_vec * t   
            
            N1, N2 = self.subnet[t].call(input_t, x[:, :, t], training=True)

            N1_grad = self.subnet[t].grad(input_t, x[:, :, t])[0] * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim  

            N1_N2_Jump = self.subnet[t].call(input_t, x[:, :, t ] * tf.math.exp(jumps[:, :, t]), training=True)[0] - N1 - N2   #(Lu(2023), equation(2.12)) 
            loss_4 = loss_4 + (tf.reduce_mean(tf.abs(N1_N2_Jump)))

            """change N1 to N1_init"""
            reward = - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], N1_init, N1_grad)) + \
               tf.reduce_sum(N1_grad * self.bsde.getFsdeDiffusion(time_stamp[t], x[:, :, t]) * dw[:, :, t], 1, keepdims=True) + N1_N2_Jump

            ####################### 
            """NEWLY ADDED"""
            td_error = reward +  N1_init - N1   
            loss_1 = loss_1 + (tf.reduce_mean(tf.square(td_error)))

            N1_init = N1_init + self.learning_rate * td_error 

            if t == self.bsde.num_time_interval-1: 
                
                N1_init = N1  
                
                """Update the previous terminal state """
                x_pt = tf.random.normal(shape=x[:, :, -2].shape, mean=0.0, stddev=1.0, dtype=self.net_config.dtype)
                
                N1_pt, N2_pt = self.subnet[t].call(input_t, x_pt, training=True)
                
                N1_grad_pt = self.subnet[t].grad(input_t, x_pt)[0] * self.bsde.getFsdeDiffusion(t, x_pt)/ self.bsde.dim   
                
                N1_N2_Jump_pt = self.subnet[t].call(input_t, x_pt * tf.math.exp(jumps[:, :, t]), training=True)[0] - N1_pt - N2_pt  
                loss_4 = loss_4 + (tf.reduce_mean(tf.abs(N1_N2_Jump_pt)))
                
                reward = - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x_pt, N1_init, N1_grad_pt)) + tf.reduce_sum(N1_grad_pt * self.bsde.getFsdeDiffusion(time_stamp[t], x_pt) * dw[:, :, t], 1, keepdims=True) + N1_N2_Jump_pt
                
                td_error = reward + N1_init - N1_pt   
                loss_1 = loss_1 + (tf.reduce_mean(tf.square(td_error)))
                
                N1_pt = N1_init + self.learning_rate * td_error


                ######### Terminal time ######
                N1_T_init = N1_pt 
                
                x_T = tf.random.normal(shape=x[:, :, -1].shape, mean=0.0, stddev=1.0, dtype=self.net_config.dtype)
                
                N1_TT, N2_TT = self.subnet[t].call(input_t, x_T, training=True)
                
                N1_grad_TT = self.subnet[t].grad(input_t, x_T)[0] * self.bsde.getFsdeDiffusion(t, x_T)/ self.bsde.dim   
                
                N1_N2_Jump_TT = self.subnet[t].call(input_t, x_T * tf.math.exp(jumps[:, :, t]), training=True)[0] - N1_TT - N2_TT  
                loss_4 = loss_4 + (tf.reduce_mean(tf.abs(N1_N2_Jump_TT)))
                
                reward = - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x_T, N1_T_init, N1_grad_TT)) + tf.reduce_sum(N1_grad_TT * self.bsde.getFsdeDiffusion(time_stamp[t], x_T) * dw[:, :, t], 1, keepdims=True) + N1_N2_Jump_TT
                
                td_error = reward + N1_T_init - N1_TT   
                loss_1 = loss_1 + (tf.reduce_mean(tf.square(td_error)))
                
                N1_T = N1_T_init + self.learning_rate * td_error

                history = history.write(t+1, N1_T)
                
            else:
                history = history.write(t+1,N1_init)
                N1_init = N1  # move to next state 

        history = tf.transpose(history.stack(),perm=[1,2,0]) 
        N1_T_avg = tf.reduce_mean(history, -1) 

        return N1_T, N1_T_avg, loss_1, loss_4

    
    def grad(self, inputs):
        x, Poissons, jumpss, dws = inputs
        x_tensor = tf.convert_to_tensor(x, dtype=tf.float64)
        Poisson  = tf.convert_to_tensor(Poissons, dtype=tf.int64)
        jumps  = tf.convert_to_tensor(jumpss, dtype=tf.float64)
        dw  = tf.convert_to_tensor(dws, dtype=tf.float64)
        inp = [x_tensor, Poisson, jumps, dw]
        with tf.GradientTape(watch_accessed_variables=True) as t:
            t.watch(x_tensor)
            #out, _, _ = self.call(inp, training=False)
            _, out, _, _ = self.call(inp, training=False)
        grad = t.gradient(out, x_tensor)  
        del t
        return grad 

    
    @tf.function
    def predict_step(self, data):  #Training = False
        x, Poisson, jumps, dw = data[0] 
        
        time_stamp = np.arange(0, self.eqn_config.num_time_interval) * self.bsde.delta_t
        all_one_vec = tf.ones(shape=tf.stack([tf.shape(Poisson)[0], 1]), dtype=self.net_config.dtype)
        
        N1_init = all_one_vec * self.subnet[0].call(all_one_vec * 0, all_one_vec * self.bsde.x_init, training=False)[0]
       
        history = tf.TensorArray(self.net_config.dtype,size=self.bsde.num_time_interval+1)     
        history = history.write(0, N1_init)
        
        
        for t in range(0, self.bsde.num_time_interval):
            
            """input: x[:, :, t ], t"""
            input_t = all_one_vec * t  
            
            N1, N2 = self.subnet[t].call(input_t, x[:, :, t], training=False)  
                            
            N1_grad = self.subnet[t].grad(input_t, x[:, :, t])[0] * self.bsde.getFsdeDiffusion(t,x[:, :, t])/ self.bsde.dim  

            N1_N2_Jump = self.subnet[t].call(input_t, x[:, :, t ] * tf.math.exp(jumps[:, :, t]), training=False)[0] - N1 - N2   #(Lu(2023), equation(2.12)) 
            
            reward = - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x[:, :, t], N1_init, N1_grad)) + \
               tf.reduce_sum(N1_grad * self.bsde.getFsdeDiffusion(time_stamp[t], x[:, :, t]) * dw[:, :, t], 1, keepdims=True) + N1_N2_Jump


            """NEWLY ADDED"""
            td_error = reward +  N1_init - N1  

            #print('N1 is ', N1)  
            N1_init = N1_init + self.learning_rate * td_error  

            if t == self.bsde.num_time_interval-1: 

                N1_init = N1  
                
                """Update the previous terminal state """
                x_pt = tf.random.normal(tf.shape(x[:, :, -2]), mean=0.0, stddev=1.0, dtype=self.net_config.dtype)
                
                N1_pt, N2_pt = self.subnet[t].call(input_t, x_pt, training=True)
                
                N1_grad_pt = self.subnet[t].grad(input_t, x_pt)[0] * self.bsde.getFsdeDiffusion(t, x_pt)/ self.bsde.dim   
                
                N1_N2_Jump_pt = self.subnet[t].call(input_t, x_pt * tf.math.exp(jumps[:, :, t]), training=True)[0] - N1_pt - N2_pt  
                
                reward = - self.bsde.delta_t * (self.bsde.f_tf(time_stamp[t], x_pt, N1_init, N1_grad_pt)) + tf.reduce_sum(N1_grad_pt * self.bsde.getFsdeDiffusion(time_stamp[t], x_pt) * dw[:, :, t], 1, keepdims=True) + N1_N2_Jump_pt
                
                td_error = reward + N1_init - N1_pt   
                
                N1_T = N1_init + self.learning_rate * td_error

                history = history.write(t+1, N1_T)
            else:
                history = history.write(t+1, N1_init)
                N1_init = N1  # move to next state 
            
        history = tf.transpose(history.stack(),perm=[1,2,0])
        return Poisson, jumps, x, history, dw  
    
    
    def simulate_path(self, num_sample):
        return self.predict(num_sample)[3]  
    



class ResNet(tf.keras.Model):   
    def __init__(self, config, dim):
        super(ResNet, self).__init__()
        block_num = config.net_config.block_num
        #hidden_layer = config.net_config.dense_hidden_num  
        num_hiddens = config.net_config.num_hiddens
        self.conc = tf.keras.layers.Concatenate() 
        self.dense_layer1 = tf.keras.layers.Dense(num_hiddens[0], activation=None, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
        self.dense_layer2 = tf.keras.layers.Dense(dim*2, activation=None, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
        self.blocks = [ResidualBlock(config) for _ in range(block_num)]  
        self.reshape = tf.keras.layers.Reshape((dim, 2))
        self.bn_layer1 = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-6, 
                                                            beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                                                            gamma_initializer=tf.random_uniform_initializer(0.1, 0.5))

    def call(self, t, inps, training):   #x.shape = (256, 5), input_t.shape = (256, 1) --> (input_t, x):(256, 6)--> (256, 5, 2)  
        x = self.conc([t, inps])  #shape=[256, 6]  
        #x = self.bn_layer1(x, training) #(256, 6)-->(256, 6)
        x = self.dense_layer1(x)   #(256, 6)--> (256, hidden_layer) 
        for i in range(len(self.blocks)):
            x = self.blocks[i](x, training)  
        x = self.dense_layer2(x)  
        x = self.reshape(x)  
        
        out1 = x[:,:,0]   
        out2 = x[:,:,1]   
        
        """ Need 2-dim outputs """
        return out1, out2
    
    def grad(self, t, inps):
        x_tensor = tf.convert_to_tensor(inps, dtype=tf.float64)
        with tf.GradientTape(watch_accessed_variables=True) as tapes:
            tapes.watch(x_tensor)
            out1, _ = self.call(t, x_tensor, training=False)
        grad = tapes.gradient(out1, x_tensor)   
        del tapes
        return grad
    
    

class ResidualBlock(tf.keras.Model):   
    def __init__(self, config):
        super(ResidualBlock, self).__init__()  
        num_hiddens = config.net_config.num_hiddens
        self.bn_layers = [tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-6,
                beta_initializer=tf.random_normal_initializer(0.0, stddev=0.1),
                gamma_initializer=tf.random_uniform_initializer(0.1, 0.5)
            )
            for _ in range(len(num_hiddens) + 2)]
        self.dense_layers = [tf.keras.layers.Dense(num_hiddens[i],
                                                   use_bias=False,
                                                   activation=None,
                                                   kernel_regularizer=l2(0.01), 
                                                   bias_regularizer=l2(0.01),)
                             for i in range(len(num_hiddens))]  
        self.tanh = tf.keras.activations.tanh

    def call(self, x0, training):   
        """structure: dense -> 5 residual blocks -> dense """
        """block structure: (bn --> dense -> tanh) * len(num_hiddens)  --> residual connection """
        x = x0
        for i in range(len(self.dense_layers) - 1): 
            x = self.dense_layers[i](x)  
            x = self.tanh(x) #tf.nn.tanh(x)
        return x + x0    
    
    

