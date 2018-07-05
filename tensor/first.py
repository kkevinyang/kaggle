# coding: utf-8

import tensorflow as tf
import numpy as np

# 用 NumPy 随机生成 100 个数据
x_data = np.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.initialize_all_variables()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(W), sess.run(b))

# 输出结果为：
# 0 [[-0.14751725  0.75113136]] [ 0.2857058]
# 20 [[ 0.06342752  0.32736415]] [ 0.24482927]
# 40 [[ 0.10146417  0.23744738]] [ 0.27712563]
# 60 [[ 0.10354312  0.21220125]] [ 0.290878]
# 80 [[ 0.10193551  0.20427427]] [ 0.2964265]
# 100 [[ 0.10085492  0.201565  ]] [ 0.298612]
# 120 [[ 0.10035028  0.20058727]] [ 0.29946309]
# 140 [[ 0.10013894  0.20022322]] [ 0.29979277]
# 160 [[ 0.1000543   0.20008542]] [ 0.29992008]
# 180 [[ 0.10002106  0.20003279]] [ 0.29996923]
# 200 [[ 0.10000814  0.20001261]] [ 0.29998815]

