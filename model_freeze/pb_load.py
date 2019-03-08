# coding=utf-8

import tensorflow as tf

def load_graph(model_dir):
    with tf.gfile.GFile(model_dir, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph: # 创建默认图
            tf.import_graph_def(graph_def, name="")  # 将graphdef导入默认图
            return graph

if __name__ == "__main__":
    graph = load_graph("model/protocol_buffer/model.pb")

    for op in graph.get_operations():
        print(op.name, op.values)

    x = graph.get_tensor_by_name('import/input_holder:0')
    y = graph.get_tensor_by_name('import/predictions:0')

    with tf.Session() as sess:
        y_out = sess.run(y, feed_dict={x: [10]})
        print(y_out)
