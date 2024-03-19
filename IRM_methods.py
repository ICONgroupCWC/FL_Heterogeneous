import tensorflow as tf
import numpy as np
import argparse
import IPython.display as display
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import pandas as pd
import tensorflow as tf
tf.compat.v1.enable_eager_execution()
# import cProfile
import copy as cp
from sklearn.model_selection import KFold


class variable_irm_game_model:
    def __init__(self, model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start):
        
        self.model_list        = model_list          # list of models for the environments and representation learner
        self.num_epochs        = num_epochs          # number of epochs
        self.batch_size        = batch_size          # batch size for each gradient update
        self.termination_acc   = termination_acc     # threshold on accuracy below which we terminate
        self.warm_start        = warm_start          # minimum number of steps before terminating
        self.learning_rate     = learning_rate       # learning rate for Adam optimizer
    
    def fit(self, data_tuple_list):
        
        n_e  = len(data_tuple_list) # number of environments

        # cross entropy loss
        def loss_comb(model_list, x, y):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            n_e = len(model_list)-1
            y_ = tf.zeros_like(y, dtype=tf.float32)
            # pass the data from the representation learner
            z = model_list[n_e](x) 
            # pass the output from the representation learner into the environments and aggregate them
            for i in range(n_e):
                model_i = model_list[i]
                y_ = y_ + 0.5*model_i(z)

            return loss_object(y_true=y, y_pred=y_)

        # gradient of cross entropy loss for environment e
        def grad_comb(model_list, inputs, targets, e):
            with tf.GradientTape() as tape:
                loss_value = loss_comb(model_list, inputs, targets)
            return loss_value, tape.gradient(loss_value, model_list[e].trainable_variables)


        model_list = self.model_list
        learning_rate = self.learning_rate
        
        # initialize optimizers for all the environments and representation learner and store it in a list
        optimizer_list = []
        for e in range(n_e):
            if (e<n_e-1):
                optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate))
            if (e==n_e-1):
                optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate*0.1))


        # training
        print('starting training')
        train_accuracy_results_0 = [] # list to store training accuracy
        flag = 'false'
        num_epochs      = self.num_epochs
        batch_size      = self.batch_size
        num_examples    = data_tuple_list[0][0].shape[0]
        period          = n_e-1
        termination_acc = self.termination_acc
        warm_start      = self.warm_start
        steps           = 0
        for epoch in range(num_epochs):
            print ("Epoch: " + str(epoch))
            datat_list = []
            for e in range(n_e):
                
                x_e = data_tuple_list[e][0]
                y_e = data_tuple_list[e][1]
                datat_list.append(shuffle(x_e,y_e))
                
            count = 0
            for offset in range(0, num_examples, batch_size):
                end = offset + batch_size
                batch_x_list = [] # list to store batches for each environment
                batch_y_list = [] # list to store batches of labels for each environment
                loss_value_list = [] # list to store loss values
                grads_list      = [] # list to store gradients
                countp = period- 1- (count % period)  # countp decides the index of the model which trains in the current step
      
                for e in range(n_e):
                    batch_x_list.append(datat_list[e][0][offset:end,:])
                    batch_y_list.append(datat_list[e][1][offset:end,:])
                    loss_value, grads = grad_comb(model_list, batch_x_list[e], batch_y_list[e],e)
                    grads_list.append(grads)

                # update models 
                optimizer_list[countp].apply_gradients(zip(grads_list[countp], model_list[countp].trainable_variables))

                # computing training accuracy
                x_in = datat_list[n_e-1][0] 
                y_in = datat_list[n_e-1][1]
                y_ = tf.zeros_like(y_in, dtype=tf.float32)
                z_in = model_list[n_e-1](x_in)
                for e in range(n_e-1):
                    y_ = y_ + model_list[e](z_in)
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                acc_train = float(epoch_accuracy(y_in, y_))
                train_accuracy_results_0.append(acc_train)
                
                # terminate
                if(steps>=warm_start and acc_train<termination_acc):
                    flag = 'true' 
                    break      
                count = count +1
                steps = steps +1
                self.train_accuracy_results = train_accuracy_results_0
            if (flag == 'true'):
                break
                
        self.model_list = model_list 
        
        self.x_in      = x_in
        self.y_in      = y_in

        
        
        
    def evaluate(self, data_tuple_test):
        
        x_test = data_tuple_test[0][0]
        y_test = data_tuple_test[0][1]
        x_in   = self.x_in
        y_in   = self.y_in
        
        model_list = self.model_list
        n_e        = len(model_list)-1
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        

        # compute training accuracy
        ytr_ = tf.zeros_like(y_in, dtype=tf.float32)
        z_in = model_list[n_e](x_in)
        for e in range(n_e):
            ytr_ = ytr_ + model_list[e](z_in)
        train_acc =  float(train_accuracy(y_in, ytr_))

        # compute testing accuracy
        z_test = model_list[n_e](x_test)
        yts_ = tf.zeros_like(y_test, dtype=tf.float32)
        for e in range(n_e):
            yts_ = yts_ + model_list[e](z_test) 

        test_acc  =  float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc



class fixed_irm_game_model:
    def __init__(self, model_list, learning_rate, num_epochs, batch_size, termination_acc, warm_start):
        
        self.model_list        = model_list             # list of models for all the environments
        self.num_epochs        = num_epochs             # number of epochs 
        self.batch_size        = batch_size             # batch size for each gradient update
        self.termination_acc   = termination_acc        # threshold on accuracy below which we terminating 
        self.warm_start        = warm_start             # minimum number of steps we have to train before terminating due to accuracy falling below threshold
        self.learning_rate     = learning_rate          # learning rate in adam
    
    def fit(self, data_tuple_list):
        n_e  = len(data_tuple_list)                     # number of environments
        # combine the data from the different environments x_in: combined data from environments, y_in: combined labels from environments, e_in: combined environment indices from environments
        x_in = data_tuple_list[0][0]
        for i in range(1,n_e):
            x_c = data_tuple_list[i][0]
            x_in = np.concatenate((x_in, x_c), axis=0)
        y_in = data_tuple_list[0][1]
        for i in range(1,n_e):
            y_c = data_tuple_list[i][1]
            y_in = np.concatenate((y_in, y_c), axis=0)
        e_in = data_tuple_list[0][2]
        for i in range(1,n_e):
            e_c = data_tuple_list[i][2]
            e_in = np.concatenate((e_in, e_c), axis=0) 
            
        # cross entropy loss
        def loss_comb(model_list, x, y):
            loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            n_e = len(model_list)
            y_ = tf.zeros_like(y, dtype=tf.float32)
            # predict the model output from the ensemble
            for i in range(n_e):
                model_i = model_list[i]
                y_ = y_ + 0.5*model_i(x)

            return loss_object(y_true=y, y_pred=y_)
        # gradient of cross entropy loss for environment e
        def grad_comb(model_list, inputs, targets, e):
            with tf.GradientTape() as tape:
                loss_value = loss_comb(model_list, inputs, targets)
            return loss_value, tape.gradient(loss_value, model_list[e].trainable_variables)



    
        model_list = self.model_list
        learning_rate = self.learning_rate


        # initialize optimizer for all the environments and representation learner and store it in a list
        optimizer_list = []
        for e in range(n_e):
            optimizer_list.append(tf.keras.optimizers.Adam(learning_rate=learning_rate))

        ####### train

        train_accuracy_results_0 = []   # list to store training accuracy


        flag = 'false'
        num_epochs = self.num_epochs
        batch_size = self.batch_size
        num_examples= data_tuple_list[0][0].shape[0]
        period      = n_e               
        termination_acc = self.termination_acc
        warm_start      = self.warm_start
        steps           = 0
        for epoch in range(num_epochs):
            print ("Epoch: "  + str(epoch))
            datat_list = []
            for e in range(n_e):
                x_e = data_tuple_list[e][0]
                y_e = data_tuple_list[e][1]
                datat_list.append(shuffle(x_e,y_e)) 
            count = 0
            for offset in range(0,num_examples, batch_size):
                end = offset + batch_size
                batch_x_list = []  # list to store batches for each environment
                batch_y_list = []  # list to store batches of labels for each environment
                loss_value_list = []  # list to store loss values
                grads_list      = []  # list to store gradients
                countp = count % period # countp decides the index of the model which trains in the current step
                for e in range(n_e):
                    batch_x_list.append(datat_list[e][0][offset:end,:])
                    batch_y_list.append(datat_list[e][1][offset:end,:])
                    loss_value, grads = grad_comb(model_list, batch_x_list[e], batch_y_list[e],e)
                    grads_list.append(grads)
                # update the environment whose turn it is to learn
                optimizer_list[countp].apply_gradients(zip(grads_list[countp], model_list[countp].trainable_variables))

                # computing training accuracy
                y_ = tf.zeros_like(y_in, dtype=tf.float32)
                for e in range(n_e):
                    y_ = y_ + model_list[e](x_in)
                epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
                acc_train = float(epoch_accuracy(y_in, y_))
                train_accuracy_results_0.append(acc_train)
                
                if(steps>=warm_start and acc_train<termination_acc): ## Terminate after warm start and train acc touches threshold we dont want it to fall below 
                    flag = 'true' 
                    break      

                count = count +1
                steps = steps +1
                self.train_accuracy_results = train_accuracy_results_0
            if (flag == 'true'):
                break
        self.model_list = model_list 
        
        self.x_in      = x_in
        self.y_in      = y_in

        
        
        
    def evaluate(self, data_tuple_test):
        ##### evaluations jmtd
        x_test = data_tuple_test[0]
        y_test = data_tuple_test[1]
        x_in   = self.x_in
        y_in   = self.y_in
        
        model_list = self.model_list
        n_e        = len(model_list)
        train_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()
        test_accuracy= tf.keras.metrics.SparseCategoricalAccuracy()

        ytr_ = tf.zeros_like(y_in, dtype=tf.float32)
        for e in range(n_e):
            ytr_ = ytr_ + model_list[e](x_in)
        train_acc =  float(train_accuracy(y_in, ytr_))

        yts_ = tf.zeros_like(y_test, dtype=tf.float32)
        for e in range(n_e):
            yts_ = yts_ + model_list[e](x_test) 

        test_acc  =  float(test_accuracy(y_test, yts_))
        
        self.train_acc = train_acc
        self.test_acc  = test_acc