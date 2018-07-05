# coding: utf-8

import tensorflow as tf
import numpy as np
import pdb


def add_layer(input,in_size, out_size,fun_on=None):
    """
    隐藏层
    :param input:
    :param in_size:
    :param out_size:
    :param fun_on:
    :return:
    """
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    pdb.set_trace()
    print('input:', type(input))
    print('Weights:', type(Weights))
    wx_plus_b = tf.matmul(input, Weights) + biases
    if fun_on is None:
        outputs = wx_plus_b
    else:
        outputs = fun_on(wx_plus_b)
    return outputs

# 26 建造神经网络
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]  # 只用līnspace的话只有一个维度（即一行）
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(x_data, 1, 10, fun_on=tf.nn.relu)
pre = add_layer(l1, 10, 1, fun_on=None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - pre), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))