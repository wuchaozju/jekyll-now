---
layout: post
title: 利用Tensorflow实现聊天机器人 1/3 - 循环神经网络 Recurrent Neural Network
published: true
---

在这三篇系列文章中，我将用Tensorflow来一步步实现一个聊天机器人（Chatbot）。

聊天机器人是自然语言处理（Natural Language Processing -NLP）的一个重要应用，[代表了未来人机交互的方向](https://www.inc.com/ben-parr/7-reasons-why-everyone-in-tech-is-obsessed-with-chatbots.html)——或许在不久的将来，我们无需安装和打开任何App，只需要跟某一个聊天机器人对话，就能实现所有的日常工作（订票、搜索、付款等）。

目前的技术，还不能做出一个完美的聊天机器人（实际上[离完美差很远](https://medium.com/swlh/why-chatbots-suck-f9dad7a54d5c)），本文实现的也是一个较为粗糙的聊天机器人.但通过它，我们可以了解其基本概念，而在最后一篇文章中，也会提出一些改进思路。

在这三篇文章中，我将会重点介绍RNN（本文）、Seq2Seq（下一篇）的基本概念，以及训练一个聊天机器人的细节（第三篇），本文需要神经网络和Tensorflow的基础，建议首先阅读以下材料：
* 神经网络和深度学习的介绍：[Neural network and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html)
* Tensorflow的入门教程：[Getting started with Tensorflow](https://www.tensorflow.org/get_started/)

传统（深度）神经网络的一个问题是忽略了数据在时序上的相关性：例如对于一个视频来说，除了每一帧的内容，这些帧的前后顺序也很重要；而对于一段文字来说，除了一个个单词本身的意思，单词间的前后顺序也很重要。

Recurrent neural networks（RNN）就是处理这类时序数据的神经网络模型（它也是一个端到端的网络，即对Loss函数的求导可以作用到网络连接的每个权重参数上，如果不明白端到端的含义，可以参考[这篇知乎的答疑](https://www.zhihu.com/question/51435499)），其本质上，就是将之前读到的数据，编码（Encode）为一种状态（Hidden state），而将此状态，也作为输入，与当前时刻的数据点，作为神经网络的输入，如下图所示：

![RNN网络的输入和数据：注意中间这个神经网络节点，与传统神经网络不同，它的输入除了输入层之外，还有自身在上一个时刻的状态]({{"/assets/rnn.png"|RNN网络}})

所以一个RNN网络，它的输入有两个：输入数据和自身上一个时刻的状态；输出也有两个：数据输出和这一时刻网络的状态。模型运行时，网络不停循环地获取当前输入，加上前一个时刻的状态，进行计算以获得新的状态，并将此状态，再放入到下一个时刻的网络中作为输入。

可以将这种循环过程展开，想象成多个网络的连接：
![RNN网络展开：将RNN按时序展开成多个网络的连接]({{"/assets/rnn_time.png"|RNN网络展开}})

RNN可以处理不同长度的数据输出（例如不同长度的句子），找到这些数据在时间上的关联性（句子开头的词和后面词的关系），然后利用从这些信息，去完成各种工作（如翻译为另一种语言、回答问题等等），这种RNN网络的能力是很强的（[与图灵机等价](http://people.cs.georgetown.edu/~cnewport/teaching/cosc844-spring17/pubs/nn-tm.pdf)）。

RNN的原理就简单到这里，如果想详细了解，我建议去读者看看[这篇论文](https://arxiv.org/abs/1506.00019)。

下面就来用Tensorflow写一个基本RNN的例子。在这个例子中，我们试图用RNN网络来预测随机数：对于一个由0和1组成序列（如“01001”），我们希望RNN能根据当前输入来预测下一个输入（如当RNN看到“0100”之后，能成功预测下一位是“1”）。

代码在[这篇文章](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)的基础上做了修改。

首先将import必要的包：

```python
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
```

接下去定义一些参数：
```python
num_epochs = 100 #训练时epoch的数量
total_series_length = 50000 #输入序列的最大长度
truncated_backprop_length = 15 #每次训练RNN时，读取时间序列的长度
state_size = 4 #状态矢量的size
num_classes = 2 #输出类别数量，对于0和1来说就是2
echo_step = 3 #偏移量
batch_size = 5 #batch的大小
num_batches = total_series_length//batch_size//truncated_backprop_length #batch的数量
```

对其中一些参数做解释：
* echo_step: 在这个例子中，RNN的输入x是一个序列（如“0100100”），其输出y也是一个序列。y相比于x是滞后的，因为RNN要看到x的一部分数据才能正确预测出y，此参数就是设置y比x滞后多少；
* epoch: 每个epoch都是一个完整的训练周期（one full training cycle，可参考[这篇问答](https://stackoverflow.com/questions/31155388/meaning-of-an-epoch-in-neural-networks-training)）；
* batch: 一个神经网络每次训练（获取梯度更新）时，不是将所有的输入数据都放进去，而是每次放一点，这一点就是一个batch；
* truncated_backprop_length: 对每个batch，读取时间序列的长度。

下面是生成训练数据的代码：
```python
def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))
    y = y.reshape((batch_size, -1))

    return (x, y)
```
其中x是输入数据，y是输出数据，它们被一起用来训练模型，即一个RNN网络。x是一个01组成的序列，y将x延后三步（将[x1,x2,...]变成[0,0,0,x1,x2,...]）。然后将x和y变成每行长度为batch_size=5的矩阵:

![Batch，[图片来源](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)]({{"/assets/batch_size.png"|batch_size}})

接下来定义placeholder，在Tensorflow中placeholder代表后续在运行中会提供的数据，这里包括输入batchX_placeholder，输出batchY_placeholder，和初始状态矢量init_state。注意，因为训练是按batch进行的，所以这些placeholder都增加了batch_size的维度。
```python
batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])
```

然后定义Variable，在Tensorflow中，Variable就是训练时更新的参数。W和b用于更新状态，W2和b2用于产生输出，参数的具体作用后面会看到。
```python
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)
```

将batch的数据拆分成columns：
```python
inputs_series = tf.split(batchX_placeholder, truncated_backprop_length, 1)
labels_series = tf.unstack(batchY_placeholder, axis=1)
```

接下去就是构建网络的过程：
```python
current_state = init_state # 目前的状态矢量
states_series = [] # 所有状态矢量放到一个list中
for current_input in inputs_series: # 遍历输入数据
    current_input = tf.reshape(current_input, [batch_size, 1]) # 获取当前输入
    input_and_state_concatenated = tf.concat([current_input, current_state], 1)  # 将输入和状态合，成为新的输入

    next_state = tf.tanh(tf.matmul(input_and_state_concatenated, W) + b)  #用此输入生成新的状态
    states_series.append(next_state)
    current_state = next_state
```
其过程可由下图表示：

![RNN网络构建，[图片来源](https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767)]({{"/assets/rnn_network.png"|rnn_network}})


将状态经过一层神经网络，预测输出：
```python
logits_series = [tf.matmul(state, W2) + b2 for state in states_series] 
predictions_series = [tf.nn.softmax(logits) for logits in logits_series] #利用softmax预测输出
```

计算loss，并以此作为训练目标，训练用[Adagrad Optimizer](http://cs.stanford.edu/~ppasupat/a9online/uploads/proximal_notes.pdf)实现。
```python
losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)
```

下面是具体的训练过程：
```python
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())  # 变量初始化

    for epoch_idx in range(num_epochs):
        x,y = generateData()
        _current_state = np.zeros((batch_size, state_size)) # 初始状态设置为0

        print("New data, epoch", epoch_idx)

        for batch_idx in range(num_batches): # 按batch训练
            start_idx = batch_idx * truncated_backprop_length # 当前输入数据的起点
            end_idx = start_idx + truncated_backprop_length # 当前输入数据的重点

            batchX = x[:,start_idx:end_idx]
            batchY = y[:,start_idx:end_idx]

            # 输入数据，开始训练
            _total_loss, _train_step, _current_state, _predictions_series = sess.run(
                [total_loss, train_step, current_state, predictions_series],
                feed_dict={
                    batchX_placeholder:batchX,
                    batchY_placeholder:batchY,
                    init_state:_current_state
                })

            # 观察loss的变化
            if batch_idx%100 == 0:
                print("Step",batch_idx, "Loss", _total_loss)
```
将上述代码保持成RNNTutorial.py，然后运行它，可以看到loss不停缩小：

![RNN训练]({{"/assets/rnn_training_output.png"|rnn_training_output}})

由此，我们就实现了一个非常简单的RNN网络，在实际应用中，我们很少用这种很简单的RNN网络（主要是因为[Vanishing gradient问题](http://neuralnetworksanddeeplearning.com/chap5.html)）,而采用LSTM等网络结构，关于LSTM，本篇不做介绍，请读者阅读[这篇文章](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)。

在下一篇文章中，我们将来看看如何在RNN基础上构建Seq2Seq网络。



