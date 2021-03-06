import sys
import os
import numpy as np
import time
import itertools
from random import randint
from pyspark import SparkFiles
from pyspark import RDD
from pyspark import StorageLevel

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

    tic = time.time()
    data, label = nth(iterator, data_index, default=(None, None))
    toc = time.time()
    with open("timing.txt", 'a') as fp:
        print >> fp, "Advancing iterator = %0.2f ms" % (1000*(toc - tic))

    if data is None:
        yield (None, None)

    else:
        tic = time.time()
        set_net_params(net, params)
        net.set_input_arrays(data, label, 0)
        outputs = net.forward()
        net.backward()
        toc = time.time()
        with open("timing.txt", 'a') as fp:
            print >> fp, "Caffe compute time = %0.2f ms" % (1000*(toc - tic))

        # Extract and return gradients as dict
        tic = time.time()
        for param in net.params:
            for idx, blob in enumerate(net.params[param]):
                param_name = get_param_name(param, idx)
                yield (param_name, blob.diff)
        toc = time.time()
        with open("timing.txt", 'a') as fp:
            print >> fp, "Caffe yield param time = %0.2f ms" % (1000*(toc - tic))

        # Also yield the outputs of the network
        tic = time.time()
        for item in outputs.items():
            yield item
        toc = time.time()
        with open("timing.txt", 'a') as fp:
            print >> fp, "Caffe yield output = %0.2f ms" % (1000*(toc - tic))


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
                 learning_rate=1e-4,
                 log_dir="log-sparknet"):

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

        self.log_dir = log_dir
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)


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

    def track_progress(self, grads, outputs):
        """Compute informative statistics about training progress. """
        # By default, we simply track variance of parameters and loss
        param_variance = ProgressStats.compute_param_variance(self.weights)
        with open(os.path.join(self.log_dir, 'outputs.txt'), 'a') as fp:
            for name in outputs:
                print >> fp, name + ":", outputs[name]

        with open(os.path.join(self.log_dir, 'param-var.txt'), 'a') as fp:
            for name in param_variance:
                print >> fp, name + ":", param_variance[name]

        grad_variance = ProgressStats.compute_grad_variance(grads)
        with open(os.path.join(self.log_dir, 'grad-var.txt'), 'a') as fp:
            for name in grad_variance:
                print >> fp, name + ":", grad_variance[name]

    def train(self, minibatchRDD, iterations=100,
              initialWeights=None, staleTol=0):
        if initialWeights is not None:
            raise NotImplementedError(
                    "Explicit weight initialization not yet supported")

        minibatchRDD.persist(storageLevel=StorageLevel.MEMORY_ONLY)
        solver, architecture = self.ship_prototxt_to_data(minibatchRDD)
        for i in xrange(iterations):
            loop_tic = time.time()
            # Step 1: Broadcast model parameters
            weights = minibatchRDD.context.broadcast(self.weights)
            toc = time.time()
            OUTPUT("Broadcast time: %0.2f ms" % (1000*(toc-loop_tic)))

            # Step 2: Run Caffe on an minibatch for each partition
            grads_output_rdd = minibatchRDD.mapPartitions(
                        lambda x: compute_gradient(x, solver, weights.value, i))
            grads_output_rdd.persist(storageLevel=StorageLevel.MEMORY_ONLY)

            # Step 3: Reduce gradients to driver
            # NOTE: I think the problem is reduceByKey creates intermediate
            # objects to hold results of addition, i.e., too many memory allocs
            net_params = self.weights.keys()
            global_grads = grads_output_rdd \
                .filter(lambda x: x[0] in net_params) \
                .map(lambda x: ((x[0], randint(0, staleTol)), (x[1], 1)),
                     preservesPartitioning=True) \
                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
                .collectAsMap()
            toc = time.time()
            OUTPUT("Time to global_grads: %0.2f ms" % (1000*(toc - loop_tic)))

            tic = time.time()
            for name, (grad_sum, count) in global_grads.items():
                global_grads[name] = grad_sum
                global_grads[name] /= count
            toc = time.time()
            OUTPUT("Normalize grads: %0.2f ms" % (1000*(toc-tic)))

            tic = time.time()
            global_outputs = grads_output_rdd \
                .filter(lambda x: x[0] not in net_params) \
                .mapValues(lambda x: (x, 1)) \
                .reduceByKey(lambda x, y: (x[0] + y[0], x[1] + y[1])) \
                .collectAsMap()
            toc = time.time()
            OUTPUT("Collect loss: %0.2f ms" % (1000*(toc-tic)))

            for name, (output_sum, count) in global_outputs.items():
                global_outputs[name] = np.squeeze(output_sum)
                global_outputs[name] /= count

            # Step 4: Update model on driver
            tic_ = time.time()
            self.update_fn(global_grads)
            toc_ = time.time()
            OUTPUT("Driver-side update time: %0.2f ms" % (1000*(toc_-tic_)))

            # Step 5: (Optionally) Record some progress metrics
            self.track_progress(global_grads, global_outputs)
            loop_toc = time.time()
            OUTPUT("Iteration time: %0.2f ms" % (1000*(loop_toc-loop_tic)))

        return
