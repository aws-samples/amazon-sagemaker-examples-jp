import os

import chainer
import numpy as np
from PIL import Image
from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from six import BytesIO
import logging
import json


def model_fn(model_dir):
    """
    This function is called by the Chainer container during hosting when running on SageMaker with
    values populated by the hosting environment.
    
    Here, we load the pre-trained model's weights. `voc_bbox_label_names` contains
    label names, and `SSD300` defines the network architecture. We pass in the
    number of labels and the path to the model for `SSD300` to load.

    Args:
        model_dir (str): path to the directory containing the saved model artifacts

    Returns:
        a loaded Chainer model

    For more on `model_fn` and `save`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    # Loads a pretrained SSD model.
    chainer.config.train = False
    path = os.path.join(model_dir, 'ssd_model.npz')
    model = SSD300(n_fg_class=len(voc_bbox_label_names), pretrained_model=path)
    return model
    
def input_fn(input_bytes, content_type):
    """This function is called on the byte stream sent by the client, and is used to deserialize the
    bytes into a Python object suitable for inference by predict_fn.
    
    Args:
        input_bytes (numpy array): a numpy array containing the data serialized by the Chainer predictor
        content_type: the MIME type of the data in input_bytes
    Returns:
        a NumPy array represented by input_bytes.
    """
    if content_type == 'application/x-npy':
        stream = BytesIO(input_bytes)
        return np.load(stream)
    elif content_type == 'image/jpeg':
        stream = Image.open(BytesIO(input_bytes))
        return np.asarray(stream).transpose(2,0,1)
    else:
        raise ValueError('Content type must be application/x-npy or image/jpeg')

def predict_fn(input_data, model):
    """
    This function receives a NumPy array and makes a prediction on it using the model returned
    by `model_fn`.
   
    
    The Chainer container provides an overridable post-processing function `output_fn`
    that accepts this function's return value and serializes it back into `npy` format, which
    the Chainer predictor can deserialize back into a NumPy array on the client.

    Args:
        input_data (bytes): a NumPy array containing the data serialized by the Chainer predictor
        model: the return value of `model_fn`
    Returns:
        a NumPy array containing predictions which will be returned to the client

    For more on `input_fn`, `predict_fn` and `output_fn`, please visit the sagemaker-python-sdk repository:
    https://github.com/aws/sagemaker-python-sdk

    For more on the Chainer container, please visit the sagemaker-chainer-containers repository:
    https://github.com/aws/sagemaker-chainer-containers
    """
    with chainer.using_config('train', False), chainer.no_backprop_mode():
        bboxes, labels, scores = model.predict([input_data])
        result = {
            'bbox': bboxes[0].tolist(),
            'label':  labels[0].tolist(),
            'score': scores[0].tolist()
        }
        return result
