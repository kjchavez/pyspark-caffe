# Can I perist a Caffe network object?
import copy
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles
from pyspark import StorageLevel

conf = SparkConf().setAppName("SparkCaffe Test")
conf.set("spark.executor.memory", "1g")
sc = SparkContext(conf=conf)

sc.addFile("models/solver.prototxt")
sc.addFile("models/train_val.prototxt")

solver = SparkFiles.get("solver.prototxt")
architecture = SparkFiles.get("train_val.prototxt")


def create_net(solver_filename):
    from caffe import SGDSolver

    net = SGDSolver(str(solver_filename)).net
    return net

netRDD = sc.parallelize([solver]*2, 2) \
             .map(create_net)

netRDD.persist(StorageLevel.MEMORY_ONLY)


def extract_unique_val(net):
    return net.params['conv1'][0].data[0, 0, 0, 0]

print netRDD.map(extract_unique_val).collect()

sc.stop()
