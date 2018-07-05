# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import pdb
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])  # 这里的None表示此张量的第一个维度可以是任何长度的
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
pdb.set_trace()
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 这个就可以看成我们的模型，表示x乘以W
# 交叉熵是用来衡量我们的预测用于描述真相的低效性
y_ = tf.placeholder("float", [None, 10])
# 计算交叉熵:
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # reduce_sum 计算张量的所有元素的总和


# 用梯度下降算法以0.01的学习速率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

# 开始训练模型
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# 用 tf.equal 来检测我们的预测是否真实标签匹配(索引位置一样表示匹配)
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))