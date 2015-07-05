import sys
import numpy as np
import time
import itertools
from random import randint
from pyspark import SparkContext, SparkConf
from pyspark import SparkFiles
from pyspark import RDD

from caffe import SGDSolver
from netutils import *


def OUTPUT(output):
    with open("output.txt", 'a') as fp:
        print >> fp, output


def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)


def compute_gradient(iterator, solver_filename, params, data_index):
    tic = time.time()
    solver = SGDSolver(str(solver_filename))
    net = solver.net
    toc = time.time()
    with open("timing.txt", 'a') as fp:
        print >> fp, "Caffe setup overhead = %0.2f ms" % (1000*(toc - tic))

    data, label = nth(iterator, data_index, default=(None, None))
    if data is None:
        yield (None, None)

    else:
        tic = time.time()
        set_net_params(net, params)
        net.set_input_arrays(data, label, 0)
        net.forward()
        net.backward()
        toc = time.time()
        with open("timing.txt", 'a') as fp:
            print >> fp, "Caffe compute time = %0.2f ms" % (1000*(toc - tic))

        # Extract and return gradients as dict
        for param in net.params:
            for idx, blob in enumerate(net.params[param]):
                param_name = get_param_name(param, idx)
                yield (param_name, blob.diff)


def reduce_gradient(lhs, rhs):
    return lhs + rhs


def group_partition_by_minibatch(X, batch_shape):
    batch_size = batch_shape[0]
    minibatch_data = np.empty(batch_shape, dtype='float32')
    minibatch_label = np.empty((batch_size, 1, 1, 1), dtype='float32')
    idx = 0
    for data, label in X:
        minibatch_data[idx] = data
        minibatch_label[idx] = label
        idx += 1
        if idx == batch_size:
            yield (minibatch_data, minibatch_label)
            idx = 0


class CaffeNetWithSGD:
    def __init__(self, solver_prototxt, architecture_prototxt,
                 input_blob="data-label", score_blob="score",
                 update='sgd',
                 learning_rate=1e-4):

        net = SGDSolver(solver_prototxt).net
        self.batch_shape = net.blobs.values()[0].data.shape
        self.batch_size = self.batch_shape[0]

        self.layer_names = [name for name in net._layer_names]

        try:
            self.input_index = self.layer_names.index(input_blob)
        except ValueError:
            print "Error: input_blob not found in net"
            sys.exit(1)

        assert(score_blob in self.layer_names)

        self._solver_filename = solver_prototxt
        self._architecture_filename = architecture_prototxt

        # Get pre-initialized weights from Caffe
        self.weights = extract_net_params(net)
        OUTPUT(pretty_format(self.weights))

        # Store training settings
        if update == 'sgd':
            self.update_fn = self.sgd_step
        elif update == 'momentum':
            self.update_fn = self.momemuntum_sgd_step
        elif update == 'rmsprop':
            self.update_fn = self.rmsprop_step
        else:
            raise ValueError("Update algorithm %s not supported." % update)

        self.learning_rate = learning_rate

    def group_by_minibatch(self, dataRDD):
        return dataRDD.mapPartitions(
                   lambda x: group_partition_by_minibatch(x, self.batch_shape))

    def predict(self, X):
        """ Assumes X is an RDD or a list of (data, label) minibatch tuples."""

        if isinstance(X, RDD):
            # Distribute files
            X.context.addFile(self._solver_filename)
            X.context.addFile(self._architecture_filename)
            X.mapPartitions(self.predict)

        solver_filename = \
            SparkFiles.get(self._solver_filename.rsplit('/', 1)[-1])
        architecture_filename = \
            SparkFiles.get(self._architecture_filename.rsplit('/', 1)[-1])

        # Might need to modify path to architecture file inside solver file.
        # Maybe we should do this before shipping the file since all Spark
        # tmp directories will be identically named.

        net = SGDSolver(solver_filename).net

        for minibatch_data, minibatch_label in X:
            # TODO: update function call for latest Caffe
            net.set_input_arrays(minibatch_data,
                                 minibatch_label,
                                 self.input_index)
            output = net.forward(end=self.score_blob)
            scores = output[self.score_blob]
            pred = np.argmax(scores, axis=1).squeeze()
            yield pred

    def ship_prototxt_to_data(self, rdd):
        rdd.context.addFile(self._solver_filename)
        rdd.context.addFile(self._architecture_filename)
        solver_filename = \
            SparkFiles.get(self._solver_filename.rsplit('/', 1)[-1])
        architecture_filename = \
            SparkFiles.get(self._architecture_filename.rsplit('/', 1)[-1])

        return solver_filename, architecture_filename

    def sgd_step(self, grads):
        """Update weights on driver, possibly multiple times using vanilla SGD.

        It's important to note that if multiple updates are done, they will use
        'stale' gradients.

        Args:
            grads: dict with key -> ((str) param_name, (int) update_index),
                   and value -> np.ndarray

        Returns:
            None
        """
        for (param_name, update_index), gradient in grads.items():
            self.weights[param_name] -= self.learning_rate * gradient

    def momemuntum_sgd_step(self, grads):
        raise NotImplementedError()

    def rms_prop_step(self, grads):
        raise NotImplementedError()

    def train(self, minibatchRDD, iterations=100,
              initialWeights=None, staleTol=0):
        # Step 1: Broadcast model parameters
        # Step 2: Copy parameters to Caffe
        # Step 3: Run Caffe on an minibatch for each partition
        # Step 4: Collect gradients and update model
        if initialWeights is not None:
            raise NotImplementedError(
                    "Explicit weight initialization not yet supported")

        weights = minibatchRDD.context.broadcast(self.weights)

        minibatchRDD.cache()
        solver, architecture = self.ship_prototxt_to_data(minibatchRDD)
        for i in xrange(iterations):
            gradRDD = minibatchRDD.mapPartitions(
                          lambda x: compute_gradient(x, solver, weights.value, i))
            global_grads = gradRDD \
                .map(lambda x: ((x[0], randint(0, staleTol)), x[1])) \
                .reduceByKey(reduce_gradient) \
                .collectAsMap()

            # Global grads are now on driver as dictionary
            OUTPUT(global_grads)
            self.update_fn(global_grads)

        return


conf = SparkConf().setAppName("SparkCaffe Test")
conf.set("spark.executor.memory", "1g")
sc = SparkContext(conf=conf)

# Add necessary files
# sc.addFile("models/train_val.prototxt")
# sc.addFile("models/solver.prototxt")


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

# gradients = data_rdd.mapPartitions(compute_gradient)
# global_grad = gradients.reduceByKey(reduce_gradient).collect()

# for param, grad in global_grad:
#     print "="*80
#     print param
#     print "="*80
#     print grad

sc.stop()
