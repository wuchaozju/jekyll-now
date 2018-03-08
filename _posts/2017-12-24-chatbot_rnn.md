---
layout: post
title: 利用Tensorflow实现聊天机器人 1/3 - 循环神经网络 Recurrent Neural Network
published: false
---

在这三篇系列文章中，我将用Tensorflow来一步步实现一个聊天机器人（Chatbot）。

聊天机器人是自然语言处理（Natural Language Processing -NLP）的一个重要应用，[代表了未来人机交互的方向](https://www.inc.com/ben-parr/7-reasons-why-everyone-in-tech-is-obsessed-with-chatbots.html)——或许在不久的将来，我们无需安装和打开任何App，只需要跟某一个聊天机器人对话，就能实现所有的日常工作（订票、搜索、付款等）。

目前的技术，还不能做出一个完美的聊天机器人（实际上离完美差很远，TODO：ref），本文实现的也是一个较为粗糙的聊天机器人，但通过它，我们可以了解其基本概念，在最后一篇文章中，也会提出一些改进思路。

在这三篇文章中，我将会重点介绍RNN（本文）、Seq2Seq（下一篇）的基本概念，以及训练一个聊天机器人的细节（第三篇），本文需要神经网络和Tensorflow的基础，如果需要的话，建议首先阅读以下材料：
* 神经网络和深度学习的介绍：[Neural network and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html)
* Tensorflow的入门教程：[Getting started with Tensorflow](https://www.tensorflow.org/get_started/)

传统（深度）神经网络的一个问题是忽略了数据本身内在的时序性（或者换种说法——它忽略了数据在时序上的相关性）：例如对于一个视频来说，除了每一帧的内容，这些帧的前后顺序也很重要；而对于一段文字来说，除了一个个单词本身的意思，单词间的前后顺序也很重要。

Recurrent neural networks（RNN）就是处理这类时序数据的神经网络模型，其本质上，就是将之前读到的数据，编码为一种状态，而将此状态，也作为输入，与当前时刻的另一个数据点，作为神经网络的输入，如下图所示：

![RNN网络的输入和数据：注意中间这个神经网络节点，与传统神经网络不同，它的输入除了输入层之外，还有自身在上一个时刻的状态]({{"/assets/rnn.png"|xxx.xxx}})

RNN可以处理不同长度的数据输出（例如不同长度的句子），找到这些数据在时间上的关联性（一个句子开始时候的词和后面词的关系），然后利用从这些信息，去完成各种工作（如翻译到成一种语言、回答问题等等），这种RNN网络的能力是很强的（与图灵机等价[Siegelmann and Sontag 1991]）。
differentiable end to end. The derivative of the loss function can be calculated with respect to each of the parameters (weights) in the model

RNN的原理就简单到这里，如果想详细了解，我建议去读者看看[这篇论文](ref: a critical review of recurrent neural networks for sequence learning)。

基本RNN

from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)

batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])



W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)

# Unpack columns
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)

# Forward passes
cell = tf.contrib.rnn.BasicRNNCell(state_size)
states_series, current_state = tf.contrib.rnn.static_rnn(cell, inputs_series, init_state)

logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)

def plot(loss_list, predictions_series, batchX, batchY):
    plt.subplot(2, 3, 1)
    plt.cla()
    plt.plot(loss_list)

    for batch_series_idx in range(5):
        one_hot_output_series = np.array(predictions_series)[:, batch_series_idx, :]
        single_output_series = np.array([(1 if out[0] < 0.5 else 0) for out in one_hot_output_series])

        plt.subplot(2, 3, batch_series_idx + 2)
        plt.cla()
        plt.axis([0, truncated_backprop_length, 0, 2])
        left_offset = range(truncated_backprop_length)
        plt.bar(left_offset, batchX[batch_series_idx, :], width=1, color="blue")
        plt.bar(left_offset, batchY[batch_series_idx, :] * 0.5, width=1, color="red")
        plt.bar(left_offset, single_output_series * 0.3, width=1, color="green")

    plt.draw()
    plt.pause(0.0001)


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    plt.ion()
    plt.figure()
    plt.show()
    loss_list = []

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size))

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * truncated_backprop_length
            end_idx = start_idx + truncated_backprop_length

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })≠≠≠≠≠≠≠

            loss_list.append(_total_loss)

            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
                plot(loss_list, _predictions_series, batchX, batchY)

plt.ioff()
plt.show()

LSTM



