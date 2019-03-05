### 用于将图片数据生成TFRecord格式
源码来源于：https://github.com/hzy46/Deep-Learning-21-Examples/tree/master/chapter_3/data_prepare
源码在Ubuntu上测试，python2

该代码在Win10+python3.6进行更改测试；

代码解析位于：https://chenzhen.online/2019/02/27/2019-02-27-TensorFlow%E4%B8%ADTFRecords%E6%95%B0%E6%8D%AE%E7%9A%84%E5%AD%98%E5%82%A8%E4%B8%8E%E8%AF%BB%E5%8F%96/


### 使用条件

目录结构：
```python
data_prepare/
  data/
    train/
      class1/
      class2/
      ...
    validation/
      class1/
      class2/
  data_prepare.py
  tfrecord.py
