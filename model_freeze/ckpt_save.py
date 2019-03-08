# coding=utf-8

import tensorflow as tf
import os.path

MODEL_DIR = "model/ckpt"
MODEL_NAME = "model.ckpt"

if not tf.gfile.Exists(MODEL_DIR):
    tf.gfile.MakeDirs(MODEL_DIR)

input_holder = tf.placeholder(tf.float32, shape=[1], name="input_holder")

w1 = tf.Variable(tf.constant(5.0, shape=[1]), name="w1")
b1 = tf.Variable(tf.constant(1.0, shape=[1]), name="b1")
_y = (input_holder * w1) + b1

predictions = tf.greater(_y, 50, name="predictions")

# 所有变量初始化
init = tf.global_variables_initializer()
saver = tf.train.Saver()    # 声明saver，用于保存模型

with tf.Session() as sess:
    sess.run(init)
    print("predictions: ", sess.run(predictions, feed_dict={input_holder: [10.0]}))
    # 保存模型
    saver.save(sess, os.path.join(MODEL_DIR, MODEL_NAME))

    graph_def = tf.get_default_graph().as_graph_def()
    print("%d ops in the final graph." % len(graph_def.node))  # 包含了前向和反向传播的节点

for op in tf.get_default_graph().get_operations():
    print(op.name, op.values())

