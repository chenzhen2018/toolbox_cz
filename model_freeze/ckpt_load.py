import tensorflow as tf


model_path = "model/ckpt/"

checkpoint = tf.train.get_checkpoint_state(model_path)  # 检查状态是否可用
input_checkpoint = checkpoint.model_checkpoint_path  # 获得ckpt文件路径

saver = tf.train.import_meta_graph(input_checkpoint + ".meta")  # 得到图

with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)  # 恢复图并得到数据
    print("predictions: ", sess.run("predictions:0", feed_dict={"input_holder:0": [10.0]}))