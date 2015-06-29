import sys
import numpy as np
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles


def get_param_name(param, idx):
    if idx == 0:
        return param+"_W"
    elif idx == 1:
        return param+"_b"
    else:
        return param+"_%d" % idx


def compute_gradient(iterator):
    print sys.path
    from caffe import SGDSolver
    solver_file = str(SparkFiles.get("solver.prototxt"))
    solver = SGDSolver(solver_file)
    net = solver.net
    for data, label in iterator:
        net.set_input_arrays(data, label, 0)
        net.forward()
        net.backward()

        # Extract and return gradients as dict
        for param in net.params:
            for idx, blob in enumerate(net.params[param]):
                param_name = get_param_name(param, idx)
                yield (param_name, blob.diff)


def reduce_gradient(lhs, rhs):
    return lhs + rhs

conf = SparkConf().setAppName("SparkCaffe Test")
conf.set("spark.executor.memory", "1g")
sc = SparkContext(conf=conf)

# Add necessary files
sc.addFile("models/train_val.prototxt")
sc.addFile("models/solver.prototxt")


def random_minibatch(x):
    data = np.random.randint(0, 256, size=(32, 1, 16, 16)).astype('float32')
    label = np.random.randint(0, 10, size=(32, 1, 1, 1)).astype('float32')
    return (data, label)


data_rdd = sc.parallelize(xrange(100)).map(random_minibatch)
gradients = data_rdd.mapPartitions(compute_gradient)
global_grad = gradients.reduceByKey(reduce_gradient).collect()

for param, grad in global_grad:
    print "="*80
    print param
    print "="*80
    print grad

sc.stop()
