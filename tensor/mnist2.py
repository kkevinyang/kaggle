# coding: utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()  # 交互式环境

# 构建Softmax 回归模型

x = tf.placeholder("float", shape=[None, 784])  # 占位符
y_ = tf.placeholder("float", shape=[None, 10])  # shape参数能够自动捕捉因数据维度不一致导致的错误

W = tf.Variable(tf.zeros([784, 10]))  # 权重
b = tf.Variable(tf.zeros([10]))  # 偏置

sess.run(tf.initialize_all_variables())  # 变量需要通过seesion初始化后，才能在session中使用

y = tf.nn.softmax(tf.matmul(x, W) + b)  # 把向量化后的图片x和权重矩阵W相乘，加上偏置b

cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 损失函数是目标类别和预测类别之间的交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 训练模型：用最速下降法让交叉熵下降

for i in range(1000):
    batch = mnist.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # tf.argmax，给出某个tensor对象在某一维上的其数据最大值所在的索引值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))  # 将布尔值转换为浮点数，计算出平均值

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))  # 准确率

"""
构建一个多层卷积网络
"""

# 权重初始化


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])  # 对于每一个输出通道都有一个对应的偏置量

# 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数
x_image = tf.reshape(x, [-1, 28, 28, 1])  # 把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数


def conv2d(x, W):
    """
    卷积
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """
    池化
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')


h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 应用ReLU激活函数
h_pool1 = max_pool_2x2(h_conv1)  # 进行max pooling

"""
第二层卷积
"""

W_conv2 = weight_variable([5, 5, 32, 64])  # 每个5x5的patch会得到64个特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)


"""
密集连接层
"""

W_fc1 = weight_variable([7 * 7 * 64, 1024])  # 加入一个有1024个神经元的全连接层，用于处理整个图片
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# 加入dropout, 减少过拟合
keep_prob = tf.placeholder("float")  # 代表一个神经元的输出在dropout中保持不变的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 可以屏蔽神经元的输出, 还会自动处理神经元输出值的scale

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

"""
训练和评估模型
"""

cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # 用ADAM优化器来做梯度最速下降
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 在feed_dict中加入额外的参数keep_prob来控制dropout比例
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

"""
[out]:
......
step 19700, training accuracy 0.98
step 19800, training accuracy 1
step 19900, training accuracy 1
test accuracy 0.993
"""

