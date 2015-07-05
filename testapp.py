# Test Application for Spark-Caffe integration
from pyspark import SparkContext, SparkConf
import numpy as np
from sparkcaffe import CaffeNetWithSGD

conf = SparkConf().setAppName("SparkCaffe Test")
conf.set("spark.executor.memory", "1g")
sc = SparkContext(conf=conf)


def random_minibatch(x):
    data = np.random.randint(0, 256, size=(32, 1, 16, 16)).astype('float32')
    label = np.random.randint(0, 10, size=(32, 1, 1, 1)).astype('float32')
    return (data, label)


def random_data(x):
    data = np.random.randint(0, 256, size=(1, 16, 16)).astype('float32')
    label = np.random.randint(0, 10, size=(1,)).astype('float32')
    return (data, label)


def dummy_load_mnist():
    return sc.parallelize(xrange(32*100)).map(random_data)

data_rdd = dummy_load_mnist()
barista = CaffeNetWithSGD("models/solver.prototxt",
                          "models/train_val.prototxt")

minibatch_rdd = barista.group_by_minibatch(data_rdd)
barista.train(minibatch_rdd, iterations=3)

sc.stop()
