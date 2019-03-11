#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#  
#      http://www.apache.org/licenses/LICENSE-2.0
#  
#  or in the "license" file accompanying this file. This file is distributed 
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either 
#  express or implied. See the License for the specific language governing 
#  permissions and limitations under the License.

from __future__ import print_function

import argparse
import os

import numpy as np
import chainer
import chainermn
import chainer.functions as F
import chainer.links as L
from chainer import training, serializers
from chainer.training import extensions
from chainer.datasets import tuple_dataset

import sys
from chainer.training import extension
from chainer.training.extensions import log_report as log_report_module
from chainer.training.extensions import util


# Define the network to train MNIST
class MLP(chainer.Chain):
    def __init__(self, n_units, n_out):
        super(MLP, self).__init__()
        with self.init_scope():
            # the size of the inputs to each layer will be inferred
            self.l1 = L.Linear(None, n_units)  # n_in -> n_units
            self.l2 = L.Linear(None, n_units)  # n_units -> n_units
            self.l3 = L.Linear(None, n_out)  # n_units -> n_out

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)


class CustomizedPrintReport(extension.Extension):

    """Trainer extension to print the accumulated results.
    This extension uses the log accumulated by a :class:`LogReport` extension
    to print specified entries of the log in a human-readable format.
    Args:
        entries (list of str): List of keys of observations to print.
        log_report (str or LogReport): Log report to accumulate the
            observations. This is either the name of a LogReport extensions
            registered to the trainer, or a LogReport instance to use
            internally.
        out: Stream to print the bar. Standard output is used by default.
    """

    def __init__(self, entries, log_report='LogReport', out=sys.stdout):
        self._entries = entries
        self._log_report = log_report
        self._out = out

        self._log_len = 0  # number of observations already printed

        # format information
        entry_widths = [max(10, len(s)) for s in entries]

        header = '  '.join(('{:%d}' % w for w in entry_widths)).format(
            *entries) + '\n'
        self._header = header  # printed at the first call

        templates = []
        for entry, w in zip(entries, entry_widths):
            templates.append((entry, '{:<%dg}  ' % w, ' ' * (w + 2)))
        self._templates = templates

    def __call__(self, trainer):
        out = self._out

        if self._header:
            out.write(self._header)
            self._header = None

        log_report = self._log_report
        if isinstance(log_report, str):
            log_report = trainer.get_extension(log_report)
        elif isinstance(log_report, log_report_module.LogReport):
            log_report(trainer)  # update the log report
        else:
            raise TypeError('log report has a wrong type %s' %
                            type(log_report))

        log = log_report.log
        log_len = self._log_len
        while len(log) > log_len:
            # delete the printed contents from the current cursor
            if os.name == 'nt':
                util.erase_console(0, 0)
            else:
                out.write('\033[J')
            self._print(log[log_len])
            log_len += 1
        self._log_len = log_len

    def serialize(self, serializer):
        log_report = self._log_report
        if isinstance(log_report, log_report_module.LogReport):
            log_report.serialize(serializer['_log_report'])

    def _print(self, observation):
        out = self._out
        for entry, template, empty in self._templates:
            if entry in observation:
                out.write(entry + '=' + template.format(observation[entry]))
            else:
                out.write(empty)
        out.write('\n')
    
if __name__=='__main__':
    
    num_gpus = int(os.environ['SM_NUM_GPUS'])

    parser = argparse.ArgumentParser()
    
    # retrieve the hyperparameters we set from the client (with some defaults)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--communicator', type=str, default='pure_nccl' if num_gpus > 0 else 'naive')
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--alpha', type=float, default=0.001)
    
    # Data, model, and output directories. These are required.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    args, _ = parser.parse_known_args()
    
    train_data = np.load(os.path.join(args.train, 'train.npz'))['images']
    train_labels = np.load(os.path.join(args.train, 'train.npz'))['labels']

    test_data = np.load(os.path.join(args.test, 'test.npz'))['images']
    test_labels = np.load(os.path.join(args.test, 'test.npz'))['labels']

    train = chainer.datasets.TupleDataset(train_data, train_labels)
    test = chainer.datasets.TupleDataset(test_data, test_labels)

    # Create the network
    model = L.Classifier(MLP(1000, 10))

    comm = chainermn.create_communicator(args.communicator)

    # comm.inter_rank gives the rank of the node. This should only print on one node.
    if comm.inter_rank == 0:
        print('# Minibatch-size: {}'.format(args.batch_size))
        print('# epoch: {}'.format(args.epochs))
        print('# communicator: {}'.format(args.communicator))    
 
    # comm.intra_rank gives the rank of the process on a given node.
    device = comm.intra_rank if num_gpus > 0 else -1
    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()

    # Setup an optimizer
    if args.optimizer == 'Adam':
        optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.Adam(args.alpha), comm)
    else:        
        optimizer = chainermn.create_multi_node_optimizer(chainer.optimizers.SGD(args.learning_rate), comm)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(5e-4))

    # Load the MNIST dataset
    train_iter = chainer.iterators.SerialIterator(train, args.batch_size)
    test_iter = chainer.iterators.SerialIterator(test, args.batch_size,
                                                 repeat=False, shuffle=False)


    # Write output files to output_data_dir. These are zipped and uploaded to S3 output path as output.tar.gz.
    updater = training.StandardUpdater(train_iter, optimizer, device=device)
    trainer = training.Trainer(updater, (args.epochs, 'epoch'), out=args.output_data_dir)

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model, device=device)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot for each specified epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epochs, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport(log_name=None))

    # Save two plot images to the result dir
    if extensions.PlotReport.available():
        trainer.extend(
            extensions.PlotReport(['main/loss', 'validation/main/loss'],
                                  'epoch', file_name='loss.png'))
        trainer.extend(
            extensions.PlotReport(
                ['main/accuracy', 'validation/main/accuracy'],
                'epoch', file_name='accuracy.png'))

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(CustomizedPrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))

    # Run the training
    trainer.run()
    
    # Save the model (only on one host).
    if comm.rank == 0:
        serializers.save_npz(os.path.join(args.model_dir, 'model.npz'), model)


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    This function loads models written during training into `model_dir`.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model
    
    For more on `model_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk
    
    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    model = L.Classifier(MLP(1000, 10))
    serializers.load_npz(os.path.join(model_dir, 'model.npz'), model)
    return model.predictor
