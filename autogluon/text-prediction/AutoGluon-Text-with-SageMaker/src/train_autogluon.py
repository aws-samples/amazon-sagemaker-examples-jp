import argparse
import sys
import os
import json

import logging
logging.basicConfig(level=logging.INFO)

import cloudpickle
import pickle

from autogluon import TextClassification as task


if __name__ == '__main__':


    # Receive hyperparameters passed via create-training-job API
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    parser.add_argument('--log-interval', type=float, default=4)

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    args = parser.parse_args()

    # Set hyperparameters after parsing the arguments
    num_epochs = args.epochs
    model_dir = args.model_dir
    training_dir = args.train

    #autogluon
    dataset = task.Dataset(filepath=os.path.join(training_dir, 'train.csv'), usecols=['text', 'target'])
    predictor = task.fit(dataset, epochs=num_epochs,pretrained_dataset='wiki_multilingual_uncased')
    cloudpickle.dump(predictor, open('%s/model'% model_dir, 'wb'))


def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.
    :param: model_dir The directory where model files are stored.
    :return: a model
    """

    try:
        net = pickle.load(open(os.path.join(model_dir, "model"), 'rb'))
        return net

    except Exception as e:
        return json.dumps("model_fn_error: " + str(e))


def transform_fn(net, data, input_content_type, output_content_type):
    """
    Transform a request using the Gluon model. Called once per request.
    :param net: The Gluon model.
    :param data: The request payload.
    :param input_content_type: The request content type.
    :param output_content_type: The (desired) response content type.
    :return: response payload and content type.
    """
    # we can use content types to vary input/output handling, but
    # here we just assume json for both

    try:
        sentence = json.loads(data)
        logging.info("Received_data: {}".format(sentence))
        output = net.predict_proba(sentence)
        response_body = json.dumps(output.asnumpy().tolist())
        return response_body, output_content_type

    except Exception as e:
        return json.dumps(str(e)), output_content_type
