# Exercise 1: Create a Basic RNN using Tensorflow. To simplify the problem, I will restrict some assumptions. Then we will
# overcome these problems step by step in next exercise

# To understand a theory of RNN, this is a perfect blog for you to know what is RNN:
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/


# However, in practice, there are some difficulty when try to create a RNN model
# 1/If you using only python as your tool:
#   - You have to know how to calculate gradients, and not a simple gradients like in basic NN, but in a very complex structure
#   - You have to create a structure of Input,hidden,output layer that works unifiedly

# 2/If you using a basic graph of Tensorflow and Theano (without API):
#   - Very hard to create a graph, because output of this state will have to be an input of the next state
# (You can try to create a simple one to see where is problem)

# To understand more clearly, I first determines some aspect:
# 1/ Input: Input will be a "sentence", contains some "words", each "words" is represent by some features
# 2/ Output: It depend on the problem you want the model to deal with. In this example is a mutilabel classifier

# To simplify the problem, I will assume:
#   +Every "sentence" will have a same length (Despite of it's not correct in real life)

# In this example, I will use MNIST data for recognize hand-written numbers. Although the problem not about "sentence",
# it can be transformed to a perfect example to see how RNN works in the simplest way.
# Data contains picture with 784 features (28 for rows and 28 columns). If I see in another way, each picture is a "sentence"
# contains 28 "words" having order, each "words" contains 28 features. So now it perfect for RNN

import pandas as pd

data=pd.read_csv("./data_2/train.csv")

data=data.values

x_train=data[:,1:]
y_train=data[:,0]

# To understand the idea more clearly how RNN works, it will worth to see what input, hidden, output layer containing
# 1/Input_layer: basic type will be like this (sequence_length,input_dim) where sequence_length is numer of words each sentence,
# input_dim is a features of each words. However, in NN, to run faster, usually have batch_size
# So the input_shape will be something like this (batch_size,sequence_length,input_dim)

# 2/Ouput_layer: Depend on your problem, Output_layer will have a different shape.
# This example is multilabel classifier, to determine which number from 0-9
# So the output_shape will be something like this (batch_size,10)

# 3/Hidden_layer: This parameter depends on user. Too high Hidden_layer may lead to overfit, too low hidden_layer may lead to underfit
# This is an art rather than science

# 4/Weights: In theory, or basic NN, usually you need at least 2 Weights_matrix, one for input -> hidden, one for hidden ->output
# However, when using API in RNN tensorflow, you only focus on Weight in hidden -> output

import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import numpy as np
batch_size=1
sequence_length=28
input_dim=28
output_dim=10

hidden_dim=20

x_=tf.placeholder(dtype=tf.float32,shape=[None,sequence_length,input_dim])
y_=tf.placeholder(dtype=tf.float32,shape=[None,output_dim])

weights=tf.Variable(tf.random_normal([hidden_dim,output_dim]))
biases=tf.Variable(tf.random_normal([output_dim]))

# Now come to a complex ones!!!!
# take a look first then I will try to explain why
def RNN(x,weights,biases):
    # x now has a shape like this (batch_size,sequence_length,input_dim)

    x=tf.transpose(x,[1,0,2])
    # x now has a shape like (sequence_length,batch_size,input_dim)

    x=tf.reshape(x,[-1,input_dim])
    # x now has a shape like (sequence_length*batch_size,input_dim)

    x=tf.split(0,sequence_length,x)
    # x now is a list of shape (sequence_length,input_dim), list contains num=batch elements


    lsmt_cell=rnn_cell.BasicLSTMCell(hidden_dim,forget_bias=1.0)

    outputs,states=rnn.rnn(lsmt_cell,x,dtype=tf.float32)

    return tf.matmul(outputs[-1],weights)
# First of all, I still don't know why to use rnn_cell. Maybe it create a structure for RNN
# However, I can only explain the "outputs" and "states" variable, "states" for the input in next state,
# "outputs" to calculate the prediction of each states

logits=RNN(x_,weights,biases)

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
# Optimizer will combine all tasks like calculate gradients and update all Variable(not placeholder) in the graph

init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    # To understand clearly about RNN flow, there are 3 case you have to solve (in my personal experience)

    # # Case 1: Try with 1 instance, loop about iteration (maybe 1000) to see if loss reduce
    # first_x=x_train[0]
    # first_y=y_train[0]
    #
    # # Transform data
    # first_x=np.reshape(first_x,(28,28))
    # tmp=np.zeros(10)
    # tmp[first_y]=1
    # first_y=tmp
    #
    # feed={x_:[first_x],y_:[first_y]}
    # iteration=1000
    # for i in range(iteration):
    #     loss=sess.run(cost,feed_dict=feed)
    #     print(loss)
    #     sess.run(optimizer,feed_dict=feed)


    # Case 2: Try with all instance in a data, but only one iteration:
    # To simplify, I will restrict data
    # restrict_x=x_train[:10000]
    # restrict_y=y_train[:10000]
    # total_loss=0
    # for x,y in zip(restrict_x,restrict_y):
    #     #Transform data
    #     tmp_x=np.reshape(x,(28,28))
    #     tmp_y=np.zeros(10)
    #     tmp_y[y]=1
    #     feed={x_:[tmp_x],y_:[tmp_y]}
    #     loss=sess.run(cost,feed_dict=feed)
    #     total_loss+=loss
    #     sess.run(optimizer,feed_dict=feed)
    # mean_loss=float(total_loss)/(len(restrict_x)//batch_size)
    # print(mean_loss)

    #Case 3: Combine both 1&2:
    restrict_x=x_train[:10000]
    restrict_y=y_train[:10000]
    iteration=50
    for i in range(iteration):
        total_loss=0
        for x, y in zip(restrict_x, restrict_y):
            # Transform data
            tmp_x = np.reshape(x, (28, 28))
            tmp_y = np.zeros(10)
            tmp_y[y] = 1
            feed = {x_: [tmp_x], y_: [tmp_y]}
            loss = sess.run(cost, feed_dict=feed)
            total_loss += loss
            sess.run(optimizer, feed_dict=feed)
        mean_loss=float(total_loss)/(len(restrict_x)//batch_size)
        print(mean_loss)



