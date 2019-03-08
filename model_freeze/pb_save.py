import tensorflow as tf
from tensorflow.python.framework import graph_util


output_graph_path = "model/protocol_buffer/model_pb.pb"

input_holder = tf.placeholder(tf.float32, shape=[1], name="w1")

w1 = tf.Variable(tf.constant(5.0, shape=[1]), name="w1")
b1 = tf.Variable(tf.constant(1.0, shape=[1]), name="b1")

_y = (input_holder * w1) + b1

predictions = tf.greater(_y, 10, name='predictions')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    predictions = sess.run(predictions, feed_dict={input_holder: [20.]})
    print(predictions)

    # 保存模型
    graph_def = tf.get_default_graph().as_graph_def()  # 获取到当前默认图，并得到其序列化后的"GraphDef"
    # 将变量转换为常量
    output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ["predictions"]  # 需要保存哪些节点
    )
    with tf.gfile.GFile(output_graph_path, "wb") as f:
        f.write(output_graph_def.SerializeToString())  # 写入文件

    print("%d ops in the final graph." % len(output_graph_def.node))    # 保存了前向传播的节点


