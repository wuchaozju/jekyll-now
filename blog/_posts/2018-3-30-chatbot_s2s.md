---
layout: post
title: 利用Tensorflow实现聊天机器人 2/3 - Seq2Seq网络
published: false
---

在[上一篇文章](./chatbot_rnn/??)中，我们实现了一个基本的RNN网络。这篇文章将在RNN的基础上，实现一个Seq2Seq（Sequence-to-Sequence，序列到序列）网络。

Seq2Seq网络的基本思想为：通过两个相连的RNN网络，实现从原序列（如一段文本）到目标序列（如另一段文本）的转换。它采用了Encoder-Decoder的结构：

![图片来源：https://nlp.stanford.edu/projects/nmt/Luong-Cho-Manning-NMT-ACL2016-v4.pdf]({{"/assets/CovUd.png"|Encoder-Decoder}})

Encoder即编码器，所谓的编码，就是对数据找到另一种表示方法，这种表示方法往往是更简洁的（或者说更抽象的），

其中Encoder和Decoder都是一个RNN

