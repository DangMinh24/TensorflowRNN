# Exercise 2: In this exercise, we try to solve a problem when using RNN for NLP in practice:
# In practice, each sentence have a different number of words. So how you gonna deal with this?

# In this link-
# http://www.wildml.com/2016/08/rnns-in-tensorflow-a-practical-guide-and-undocumented-features/
# People propose a solution when dealing with this problem:
# They will choose a sentence that have most words, and use as shape for sequence_length.
# However, if there is some special case, only one sentence have much much more words than other, and if we
# apply the length of this sentence to the other, the cost of memory and computation becomes very high


# However, I will try to apply the easiest one method, to see how it works
# In this example, we try to create a language model: You input an sentence, model will predict probability of each next word depend on previous word
# First, we need a practice data for this example. We would use a reddit-comments data, which only contains raw data
# So we have to preprocess data for working
from nltk import sent_tokenize,word_tokenize
import csv
from nltk import FreqDist
from itertools import chain
import numpy as np
START_SYMBOL="_START_"
END_SYMBOL="_END_"
UNKNOW_SYMBOL="_UNKNOWN_"
tokenized_sentences=[]
flag=0
with open("./data/reddit-comments-2015-08.csv","r") as f:
    reader=csv.reader(f)
    # Skip the header
    next(reader)

    for e in reader:
        # To restrict data and work faster
        if len(tokenized_sentences)>=2000:
            break
        sentences=sent_tokenize(e[0])
        for sent in sentences:
            tokens=word_tokenize(sent)
            tokens.insert(0,START_SYMBOL)
            tokens.append(END_SYMBOL)
            tokenized_sentences.append(tokens)

vocabulary_size=100

fd=FreqDist(chain(*tokenized_sentences))
vocabulary=[w for w,c in fd.most_common(vocabulary_size-1)]
vocabulary.append(UNKNOW_SYMBOL)
index2word=vocabulary
word2index=dict([(w,i) for i,w in enumerate(index2word)])

for i, sent in enumerate(tokenized_sentences):
    tokenized_sentences[i]=[w if w in vocabulary else UNKNOW_SYMBOL for w in sent]

x_train=[[word2index[w] for w in sent[:-1]] for sent in tokenized_sentences[:1000]]
y_train=[[word2index[w] for w in sent[1:]] for sent in tokenized_sentences[:1000]]

# # Now determine the sentence have maximum length:
# max_length=0
# for sent in x_train:
#     if len(sent)>max_length:
#         max_length=len(sent)
# print(max_length)
#
# # Then we have to padding with sentence that don't have enough words
# def padding(sent_index,max_length):
#     # You can create another token= __EMPTY__ or use the UNKNOWN ones
#     # In this example, I will use UNKNOWN ones
#     tmp_sent=sent_index.copy()
#     if len(sent_index)<max_length:
#         for i in range(max_length-len(sent_index)):
#             tmp_sent.append(word2index[UNKNOW_SYMBOL])
#     return tmp_sent
#
# def padding_data(x_,y_,max_length):
#     tmp_x=x_.copy()
#     tmp_y=y_.copy()
#
#     for iter,(x,y) in enumerate(zip(tmp_x,tmp_y)):
#         tmp_x[iter]=padding(x,max_length)
#         tmp_y[iter]=padding(y,max_length)
#
#     return tmp_x,tmp_y
#
# x_train,y_train=padding_data(x_train,y_train,max_length)
# # Now our data has a perfect same words each sentence. However, each word must have features
#
# def word_index2vec(word_index):
#     vec=np.zeros(vocabulary_size)
#     vec[word_index]=1
#     return vec
#
# for iter,sent in enumerate(x_train):
#     x_train[iter]=[word_index2vec(w) for w in sent]
#
# for iter,sent in enumerate(y_train):
#     y_train[iter]=[word_index2vec(w) for w in sent]
#
# print(x_train)

# Using max length cost a lot of memory!!!!
# So I try to reduce number of sentence
restrict_length=50
def padding(sentence,restrict):
    tmp_sent=sentence.copy()
    if len(sentence)<restrict:
        for i in range(restrict-len(sentence)):
            tmp_sent.append(word2index[UNKNOW_SYMBOL])
    elif len(sentence)>restrict:
        tmp_sent=tmp_sent[:restrict]
    return tmp_sent

def padding_data(x_,y_,restrict):
    for iter,(x,y) in enumerate(zip(x_,y_)):
        x_[iter]=padding(x,restrict)
        y_[iter]=padding(y,restrict)
    return x_,y_

x_train,y_train=padding_data(x_train,y_train,restrict_length)

def index2vec(index):
    vec=np.zeros(vocabulary_size)
    vec[index]=1
    return vec

for iter,(x,y) in enumerate(zip(x_train,y_train)):
    x_train[iter]=[index2vec(w) for w in x ]
    y_train[iter]=[index2vec(w) for w in y]


# Now we will see the result !!!
# Determine parameters:
import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
batch_size=1
sequence_length=50
input_dim=vocabulary_size
output_dim=vocabulary_size
hidden_dim=20
early_st=20

# In this example, output layer will be a size of vocabulary

x=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length,input_dim])
y=tf.placeholder(dtype=tf.float32,shape=[batch_size,sequence_length,output_dim])

W=tf.Variable(tf.random_normal(shape=[hidden_dim,vocabulary_size]))
bias=tf.Variable(tf.random_normal(shape=[1,vocabulary_size]))

early_stop=tf.placeholder(tf.int32)
def RNN(x,weights,biases):
    # firstly, x will have shape like this (batch,sequenceLenght,input_dim)
    x=tf.transpose(x,[1,0,2])

    x=tf.reshape(x,[-1,input_dim])
    # Now x have shape like this (sequence*batch,input_dim)

    x=tf.split(0,sequence_length,x)

    lmst_cell=rnn_cell.BasicLSTMCell(num_units=hidden_dim,state_is_tuple=True,forget_bias=1.0)

    outputs,states=rnn.rnn(lmst_cell,x,dtype=tf.float32,sequence_length=early_stop)

    # In this example, we will slightly change to loss function.
    # Instead of using only the last output in the RNN_ex, we use all the output to calculate loss
    # outputs now is a list "sequence_length" elements of tensorflow having shape (1,hidden_dim)
    # However, if we use concept list in python, we can not get a result we want to
    # So I try to represent list of tensors in tensorflow concept

    flat=tf.concat(0,outputs)


    # Now we have list of tensors, each elements for each predictions
    # We can compute pred=output*W +bias (tf.matmul(pred,W)+bias)
    # However, I using batch_matmul for each element in list
    return tf.batch_matmul(flat,weights)+bias


logits=RNN(x,W,bias)
y_flat=tf.reshape(y,[-1,output_dim])
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits,y_flat))

optimizer=tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)


with tf.Session() as sess:
    init=tf.initialize_all_variables()
    sess.run(init)

    # To understand more the flow of the RNN, try to run 3 cases (like in RNN_ex1):
    # Case 1:
    # feed={x:[x_train[0]],y:[y_train[0]]}
    # iteration=1000
    # for iter in range(iteration):
    #     loss=sess.run(cost,feed_dict=feed)
    #     print(loss)
    #     sess.run(optimizer,feed_dict=feed)

    # Case 2:
    # total_loss=0
    # for x_,y_ in zip(x_train,y_train):
    #     feed={x:[x_],y:[y_]}
    #     loss=sess.run(cost,feed_dict=feed)
    #     total_loss+=loss
    # mean_loss=total_loss/(len(x_train)//batch_size)
    # print(mean_loss)

    # Case 3:
    iteration=50
    for i in range(iteration):
        total_loss=0
        for x_,y_ in zip(x_train,y_train):
            feed={x:[x_],y:[y_],early_stop:early_st}
            loss=sess.run(cost,feed_dict=feed)
            total_loss+=loss
            sess.run(optimizer,feed_dict=feed)
        mean_loss=total_loss/(len(x_train)//batch_size)
        print(mean_loss)
