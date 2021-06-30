from time import sleep
import datetime,json
import awsiot.greengrasscoreipc
from awsiot.greengrasscoreipc.model import (
    PublishToTopicRequest,
    PublishMessage,
    JsonMessage
)
import tensorflow as tf
from PIL import Image
import os, sys
import numpy as np
from logging import getLogger
logger = getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

logger.info(f'argv:{sys.argv}')

model_path = os.path.join(sys.argv[1],'1.h5')

logger.info('start publisher...')

TIMEOUT = 10
interval = 60
logger.info('start to load model')

model = tf.keras.models.load_model(model_path)

logger.info('start to iot client')

ipc_client = awsiot.greengrasscoreipc.connect()

topic = "my/topic"

logger.info('start loop')

while True:
    noise = np.random.uniform(-1, 1, (1,7,7,1))
    img_array = ((model.predict(noise)*127.5)+127.5).astype(np.uint8)
    img = Image.fromarray(img_array[0,:,:,0])
    file_name = '/tmp/' + datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).strftime('%Y%m%d%H%M%S') + '.png'
    img.save(file_name)

    message = {"file_name": file_name }
    message_json = json.dumps(message).encode('utf-8')

    request = PublishToTopicRequest()
    request.topic = topic
    publish_message = PublishMessage()
    publish_message.json_message = JsonMessage()
    publish_message.json_message.message = message
    request.publish_message = publish_message
    operation = ipc_client.new_publish_to_topic()
    operation.activate(request)
    future = operation.get_response()
    future.result(TIMEOUT)

    logger.info(f'publish message: {message_json}')
    sleep(interval)
