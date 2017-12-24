---
layout: post
title: 利用Tensorflow实现聊天机器人 1/3 - 循环神经网络 Recurrent Neural Network 
published: false
---

我计划用三篇文章，实现一个聊天机器人（Chatbot）。

聊天机器人是自然语言处理（Natural Language Processing -NLP）的一个重要应用，[代表了未来人机交互的方向](https://www.inc.com/ben-parr/7-reasons-why-everyone-in-tech-is-obsessed-with-chatbots.html)——或许在不久的将来，我们无需安装和打开任何App，只需要跟某一个聊天机器人对话，就能实现所有的日常工作（订票、搜索、付款等）。

在这三篇文章中，我将会重点介绍RNN（本文）、Seq2Seq（下一篇）的基本概念，以及训练一个聊天机器人的细节（第三篇），本文需要神经网络和Tensorflow的基础，如果需要的话，建议首先阅读以下材料：
* 神经网络的介绍：[Neural network and deep learning](http://neuralnetworksanddeeplearning.com/chap1.html)
* Tensorflow的入门教程：[Getting started with Tensorflow](https://www.tensorflow.org/get_started/)

