import argparse
import json
import mxnet as mx
import numpy as np
from mxnet import gluon
import os
from pathlib import Path
from datetime import datetime

import pandas as pd
import gluonts
from gluonts.model.n_beats import NBEATSEnsembleEstimator, NBEATSEstimator
from gluonts.dataset.multivariate_grouper import MultivariateGrouper
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor

if __name__ == '__main__':
    
    num_cpus = int(os.environ['SM_NUM_CPUS'])
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    ctx = mx.gpu() if num_gpus > 0 else mx.cpu()
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--columns', type=str, default='pm2.5')
    parser.add_argument('--prediction-length', type=int, default=12)
    parser.add_argument('--context-length', type=int, default=168)
    parser.add_argument('--frequency', type=str, default="1H")

    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    args = parser.parse_args()
    
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.learning_rate
    columns = args.columns.split(",")
    prediction_length = args.prediction_length
    context_length = args.context_length
    freq = args.frequency
    
    
    model_dir = args.model_dir
    train_path = args.train
    hosts = args.hosts
    
    ## Data Preparation
    # Read dataframe from csv
    data = pd.read_csv(os.path.join(train_path, "train.csv"),
                   sep=',', 
                   index_col=0,
                   parse_dates=True)
    
    # Set dataset for gluon-ts
    # Extract column specified by "columns"
    training_data = ListDataset(
        [{"start": data.index[0], "target": data[c]} for c in columns],
        freq = freq
    )
    
    ## Training configuration    
    # Setting for distributed training
    #     if len(hosts) == 1:
    #         kvstore = 'device' if num_gpus > 0 else 'local'
    #     else:
    #         kvstore = 'dist_device_sync' if num_gpus > 0 else 'dist_sync'
    #

    estimator = NBEATSEstimator(
            freq=freq, 
            prediction_length=prediction_length,
            context_length = context_length,
            trainer=Trainer(batch_size=batch_size, 
                            ctx=ctx, 
                            epochs=epochs, 
                            hybridize=True, 
                            init="xavier", 
                            learning_rate=lr)
    )
    
    predictor = estimator.train(training_data)
    
    # Save model 
    # GluonTS gives "serialize" that supports saving a gluonts model conveniently.
    predictor.serialize(Path(model_dir))
    
def model_fn(model_dir):
    
    return Predictor.deserialize(Path(model_dir))

def transform_fn(net, data, input_content_type, output_content_type):
    
    try:
        data = json.loads(data) 
        
        #How many time-series are included?
        N = len(data["value"])
        
        #Create dataset
        test_data = ListDataset(
            [{"start": datetime.strptime(data["index"], "%Y-%m-%d %H:%M:%S"), 
              "target": np.array(data["value"][n])} for n in range(N)
            ],
            freq = data["freq"]
        )
        
        # prediction
        forecast_it = net.predict(test_data)
        forecasts = list(forecast_it)
        
        result = []
        for n in range(N):
            result.append(forecasts[n].samples.tolist())
        response_body = json.dumps(result)
        return response_body, output_content_type
    
    except Exception as e:
        print(e)
        return json.dumps(str(e)), output_content_type
    
