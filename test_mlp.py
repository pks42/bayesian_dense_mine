import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score

def unit_train_nn(x, y, num_epochs, use_cols):
    x = x[:,:, use_cols]
    x_dim = len(x[0,0])
    num_batches = len(x)

    # Preparing training data (inputs-outputs)  
    training_inputs = tf.placeholder(shape=[None, x_dim], dtype=tf.float32)  
    training_outputs = tf.placeholder(shape=[None, 1], dtype=tf.float32) #Desired outputs for each input
  
    """ 
    Hidden layer with x_dim neurons 
    """  
   
    # Preparing neural network parameters (weights and bias) using tf Variables  
    weights_1 = tf.Variable(tf.truncated_normal(shape=[x_dim, 4], dtype=tf.float32))  
    bias_1 = tf.Variable(tf.truncated_normal(shape=[1, 4], dtype=tf.float32))  

    # Preparing inputs of the activation function  
    af_input_1 = tf.matmul(training_inputs, weights_1) + bias_1  
  
    # Activation function of the output layer neuron  
    layer1_output = tf.nn.sigmoid(af_input_1)  

    # """ 
    # Hidden layer with 3 neurons 
    # """  
   
    # # Preparing neural network parameters (weights and bias) using tf Variables  
    # weights_2 = tf.Variable(tf.truncated_normal(shape=[4, 3], dtype=tf.float32))  
    # bias_2 = tf.Variable(tf.truncated_normal(shape=[1, 3], dtype=tf.float32))  

    # # Preparing inputs of the activation function  
    # af_input_2 = tf.matmul(layer1_output, weights_2) + bias_2  
  
    # # Activation function of the output layer neuron  
    # layer2_output = tf.nn.sigmoid(af_input_2)  
  
    """ 
    Output layer with one neuron 
    """  
  
    # Preparing neural network parameters (weights and bias) using tf Variables  
    weights_output = tf.Variable(tf.truncated_normal(shape=[4, 1], dtype=tf.float32))  
    bias_output = tf.Variable(tf.truncated_normal(shape=[1, 1], dtype=tf.float32))  
  
    # Preparing inputs of the activation function  
    af_output = tf.matmul(layer1_output, weights_output) + bias_output  
  
    # Activation function of the output layer neuron  
    predictions = tf.nn.sigmoid(af_output)  
  
    #-----------------------------------  
  
    # Measuring the prediction error of the network after being trained  
    loss = tf.losses.mean_squared_error(training_outputs, predictions)
  
    # Minimizing the prediction error using gradient descent optimizer  
    train_op = tf.train.AdamOptimizer(0.05).minimize(loss)  
  
    # Creating a tf Session  
    sess = tf.Session()  
  
    # Initializing the tf Variables (weights and bias)  
    sess.run(tf.global_variables_initializer())  
  
    # Training loop of the neural network  
    train_acc = 0
    for step in range(num_epochs):
        for j in range(num_batches):
            epoch_acc = 0
            op, err, p = sess.run(fetches=[train_op, loss, predictions],  
                              feed_dict={training_inputs: x[j],  
                                         training_outputs: y[j]})  
            p[p >= 0.5] = 1.0
            p[p < 0.5] = 0.0
            epoch_acc += accuracy_score(y[0], p)
        train_acc += (epoch_acc/num_batches)
        print(str(step), ": ", err)  
    train_acc = train_acc/num_epochs
    print("Training accuracy = " + str(train_acc))
    # Class scores of some testing data  
    print("Expected class scores : ", sess.run(predictions, feed_dict={training_inputs: x[0]}))  

    print('Actual scores:' +str(y[0]))

    # Printing hidden layer weights initially generated using tf.truncated_normal()  
    print("Hidden layer weights: ", sess.run(weights_1))  
  
    # Printing hidden layer bias initially generated using tf.truncated_normal()  
    print("Hidden layer biases : ", sess.run(bias_1))  
  
    # Printing output layer weights initially generated using tf.truncated_normal()  
    print("Output layer weights : ", sess.run(weights_output))  
  
    # Printing output layer bias initially generated using tf.truncated_normal()  
    print("Output layer biases : ", sess.run(bias_output))  
  
    # Closing the tf Session to free resources  
    sess.close()  


if __name__ == "__main__":
    x_test = np.array([[[5.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 3.47407e+03,
                            6.66700e+01, 4.50240e+04, 1.10730e+04, 1.00000e+00],
                        [3.90000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.57207e+03,
                            1.00000e+02, 5.92550e+04, 4.51000e+02, 1.00000e+00],
                        [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
                            7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
                        [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
                            8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00],
                        [5.60000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.62207e+03,
                            1.00000e+02, 5.99030e+04, 1.57360e+04, 1.00000e+00],
                        [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.02700e+03,
                            1.00000e+02, 3.92300e+04, 1.09300e+03, 1.00000e+00],
                        [5.20000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 4.22184e+03,
                            8.88900e+01, 5.47160e+04, 1.38500e+04, 1.00000e+00],
                        [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.55525e+03,
                            8.88900e+01, 3.31170e+04, 1.59800e+04, 1.00000e+00]],
                        [[6.40000e+01, 1.00000e+00, 1.00000e+00, 2.00000e+01, 3.82100e+03,
                            8.88900e+01, 4.95210e+04, 2.17300e+03, 1.00000e+00],
                        [6.50000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 4.14522e+03,
                            1.00000e+02, 5.37230e+04, 2.74600e+03, 1.00000e+00],
                        [6.40000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 2.98403e+03,
                            7.50000e+01, 3.86740e+04, 2.50500e+03, 0.00000e+00],
                        [6.20000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 2.89707e+03,
                            1.00000e+02, 3.75470e+04, 8.16000e+02, 1.00000e+00],
                        [6.00000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 7.96923e+03,
                            1.00000e+02, 1.03282e+05, 1.89300e+03, 1.00000e+00],
                        [6.50000e+01, 2.00000e+00, 2.00000e+00, 2.00000e+01, 8.39105e+03,
                            1.00000e+02, 1.08749e+05, 4.14400e+03, 1.00000e+00],
                        [5.40000e+01, 1.00000e+00, 3.00000e+00, 2.00000e+01, 6.26304e+03,
                            1.00000e+02, 8.11690e+04, 1.76110e+04, 1.00000e+00],
                        [4.70000e+01, 1.00000e+00, 2.00000e+00, 2.00000e+01, 3.75000e+03,
                            1.00000e+02, 4.86000e+04, 1.56850e+04, 1.00000e+00]]])
        
    y_test = np.array([[0., 0.,  1., 1., 0.,  1., 0.,  1.],
                        [1.,  1.,  1., 0., 0.,  1., 0., 0.]])
    print(y_test.shape)
    y_test = y_test.reshape((2,8,1))

    ## THIS HELPS CHOOSE WHICH COLUMNS ARE USED FROM X
    use_cols = [0,4,5]

    unit_train_nn(x_test, y_test, 100, use_cols)