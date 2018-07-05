# coding: utf-8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# lesson 21 矩阵计算
max1 = tf.constant([[3, 3]])
max2 = tf.constant([[3], [3]])

product = tf.matmul(max1, max2)
with tf.Session() as se:
    res = se.run(product)
    print(res)

# 22 变量计算
state = tf.Variable(0, name='counter')
one = tf.constant(1)

new_state = tf.add(state, one)
update = tf.assign(state, new_state)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))

# 23 placeholder
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.2], input2: [2.0]}))


# 24 激励函数；为了对函数进行变形方便分析，如sigmal。可以着重对某些特征进行辨认
# 有线性和非线性的

# 25 增加隐藏层
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
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    wx_plus_b = tf.matmul(input,Weights) + biases
    if fun_on is None:
        outputs = wx_plus_b
    else:
        outputs = fun_on(wx_plus_b)
    return outputs

# 26 建造神经网络
x_data = np.linspace(-1,1,300)[:,np.newaxis]  # 只用līnspace的话只有一个维度（即一行）
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32,[None, 1])
ys = tf.placeholder(tf.float32,[None, 1])

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

# 27 可视化

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x_data, y_data)
plt.show()

# 可视化拟合
plt.ion()
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={xs: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)



