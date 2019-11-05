# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
# PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import argparse

import numpy as np

import tensorflow as tf


def main(args):
    train_dir = args.train
    
    x_train = np.load(os.path.join(train_dir, 'train.npz'))['image']
    y_train = np.load(os.path.join(train_dir, 'train.npz'))['label']
    x_test = np.load(os.path.join(train_dir, 'test.npz'))['image']
    y_test = np.load(os.path.join(train_dir, 'test.npz'))['label']
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])    

    callbacks = []
    
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(
        args.model_dir + '/checkpoint-{epoch}.h5')
                    )

    model.fit(x_train, y_train,
              epochs=args.epochs,
              batch_size=args.batch_size,
              callbacks=callbacks)
    
    model.evaluate(x_test, y_test)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--batch_size',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        type=int,
                        default=12)
    
    parser.add_argument('--model-dir',
                        type=str,
                        default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train',
                        type=str,
                        default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()
    main(args)