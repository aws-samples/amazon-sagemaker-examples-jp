from time import sleep
import datetime,json
from awsiot.greengrasscoreipc import connect
from awsiot.greengrasscoreipc.model import (
    QOS,
    PublishToIoTCoreRequest
)
import tensorflow as tf
from PIL import Image
import os, sys
import numpy as np
import signal
from logging import getLogger
logger = getLogger(__name__)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

TIMEOUT = 10

ipc_client = connect()

INTERVAL = 60

logger.info('start to load model')
gan_model = tf.keras.models.load_model('/app/gan.h5')
classifier_model = tf.keras.models.load_model('/app/classifier.h5')

topic = "inference/result"

def signal_handler(signal, frame):
    logger.info(f"Received {signal}, exiting")
    sys.exit(0)

# Register SIGTERM for shutdown of container
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

cnt = 0
while True:
    # 撮影(GANで画像生成)
    noise = np.random.uniform(-1, 1, (1,7,7,1))
    img_array = ((gan_model.predict(noise)*127.5)+127.5).astype(np.uint8)
    img = Image.fromarray(img_array[0,:,:,0])
    
    # 撮影した画像をnumpy配列へ変換
    img_array = (np.array(img).reshape(1,28,28,1)-127.5)/127.5
    pred_y = np.argmax(classifier_model.predict(img_array))
    
    result = 'anomaly' if pred_y % 2 == 0 else 'normal'

    cnt = cnt + 1
    message = {
        "timestamp": str(datetime.datetime.now()), 
        "message": result,
        "counter": str(cnt),
        "component_version" : "1.0.0"
    }

    request = PublishToIoTCoreRequest(topic_name=topic, qos=QOS.AT_LEAST_ONCE, payload=bytes(json.dumps(message), "utf-8"))
    operation = ipc_client.new_publish_to_iot_core()
    operation.activate(request)
    future = operation.get_response()
    future.result(TIMEOUT)
    
    logger.info("publish")
    sleep(INTERVAL)
