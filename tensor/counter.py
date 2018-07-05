# coding: utf-8
"""
最简单的计数器，仅仅为了展示基本方式
"""
import tensorflow as tf

# 创建一个变量，  初始化为标量 0
state = tf.Variable(0, name="counter")

# 创建一个operation, 其作用是使state 增加 1
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)  # 这样才能重复执行+1的操作,实际上就代表：state=new_value

# 启动图后, 变量必须先经过`初始化` (init) op 初始化,
# 首先必须增加一个`初始化` op 到图中.
init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)  # 运行 init_op

    print(sess.run(state))  # 打印出事状态

    for _ in range(10):
        sess.run(new_value)
        print(sess.run(new_value))
