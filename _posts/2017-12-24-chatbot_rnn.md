---
layout: post
title: 利用Tensorflow实现聊天机器人 1/3 - 循环神经网络 Recurrent Neural Network 
published: false
---

在这三篇系列文章中，我将用Tensorflow来一步步实现一个聊天机器人（Chatbot）。

聊天机器人是自然语言处理（Natural Language Processing -NLP）的一个重要应用，[代表了未来人机交互的方向](https://www.inc.com/ben-parr/7-reasons-why-everyone-in-tech-is-obsessed-with-chatbots.html)——或许在不久的将来，我们无需安装和打开任何App，只需要跟某一个聊天机器人对话，就能实现所有的日常工作（订票、搜索、付款等）。

目前的技术，还不能做出一个完美的聊天机器人（TODO：ref），本文实现的也是一个较为粗糙的聊天机器人，但通过它，我们可以了解其基本概念，在最后一篇文章中，也会提出一些改进思路。

在这三篇文章中，我将会重点介绍RNN（本文）、Seq2Seq（下一篇）的基本概念，以及训练一个聊天机器人的细节（第三篇），本文需要神经网络和Tensorflow的基础，如果需要的话，建议首先阅读以下材料：
* 神经网络的介绍：[Neural network and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html)，[深度神经网络]
* Tensorflow的入门教程：[Getting started with Tensorflow](https://www.tensorflow.org/get_started/)

传统（深度）神经网络的一个问题是忽略了数据本身内在的时序性（或者换一种说法——时序上的相关性）：例如对于一个视频来说，除了每一帧的内容之外，这些帧的前后顺序也很重要；而对于一段文字来说，除了单词之外，单词的前后顺序也很重要。Recurrent neural networks（RNN）就是一种这类时序数据的神经网络模型，我在这里简单介绍它的概念，如果想详细了解，我建议去读者看看[这篇论文](ref: a critical review of recurrent neural networks for sequence learning)。

本质上，将前面读到的数据，转化为一种状态，而将此状态，也作为一种输入，与当前时刻的另一个数据点，作为神经网络的输入。

RNN可以处理不同长度的数据输出，
时间上的关联性

这种网络的能力是很强的，与图灵机等价[Siegelmann and Sontag 1991]
differentiable end to end. The derivative of the loss function can be calculated with respect to each of the parameters (weights) in the model




