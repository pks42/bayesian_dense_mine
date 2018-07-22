from math import exp
import numpy as np
import random

def activation_fn(z):
    f = 1.0/(1.0 + exp(-z))  ##Sigmoid function
    return f

activation_fn = np.vectorize(activation_fn)

def d_sigmoid(f_z):
    derivative = f_z*(1.0 - f_z)
    return derivative

d_sigmoid = np.vectorize(d_sigmoid)

#CREATE NEURAL NET
L1 = np.zeros([9,1],dtype = int)  #INPUT LAYER
L1[8] = 1 # Bias term for input layer
L2 = np.zeros([4,1]) #HIDDEN LAYER
L2[3] = 1 # Bias term for hidden layer
L3 = np.zeros([8,1]) #OUTPUT LAYER

#DEFINE WEIGHTS
weights_L1 = np.random.rand(3,9) * 0.0001
weights_L2 = np.random.rand(8,4) * 0.0001

#HYPERPARAMETERS
no_epochs = 7000
learn_rate = 1.0
weight_decay = 0.0001

for i in range(no_epochs):
    d_weights_L1 = np.zeros([3,9])
    d_weights_L2 = np.zeros([8,4])
    sum_errors = 0
    for j in range(8):
        L1[j] = 1
        x = L1[:8,:]
        #Forward propagation
        z_L2 = np.matmul(weights_L1,L1)
        a_L2 = activation_fn(z_L2)
        L2[:3] = a_L2
        z_L3 = np.matmul(weights_L2,L2)
        a_L3 = activation_fn(z_L3)
        L3 = a_L3
        y = x
        #Back propagation 
        sum_errors += np.sum(a_L3 - y)
        d_L3 = (a_L3 - y)*d_sigmoid(z_L3)
        d_L2 = (np.matmul(weights_L2[:,:3].transpose(),d_L3)*d_sigmoid(z_L2))
        d_weights_L1[:,:8] = d_weights_L1[:,:8] + np.matmul(d_L2,x.transpose())
        d_weights_L1[:,8] = np.reshape((np.reshape(d_weights_L1[:,8],(3,1)) + d_L2),(3,))        
        d_weights_L2[:,:3] = d_weights_L2[:,:3] + np.matmul(d_L3,a_L2.transpose())
        d_weights_L2[:,3] = np.reshape((np.reshape(d_weights_L2[:,3],(8,1)) + d_L3),(8,))
        print(x)
        print(a_L3)
        L1[j] = 0
    print("Epoch: " + str(i) + "\t Error = " + str(sum_errors/8))
    weights_L1[:,:8] += - learn_rate*((0.125)*d_weights_L1[:,:8])+ (weight_decay*weights_L1[:,:8])
    weights_L1[:,8] += - learn_rate*(((0.125)*d_weights_L1[:,8]))
    weights_L2[:,:3] += - learn_rate*((0.125)*d_weights_L2[:,:3])+ (weight_decay*weights_L2[:,:3])
    weights_L2[:,3] += - learn_rate*(((0.125)*d_weights_L2[:,3]))