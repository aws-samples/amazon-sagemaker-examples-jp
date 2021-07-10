import argparse
from collections import OrderedDict
import json
import numpy as np
import sys
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

import transformers
from transformers import BertJapaneseTokenizer
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup

from EarlyStopping import EarlyStopping

import logging
logging.basicConfig(level=logging.INFO)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQUENCE_LENGTH=128

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.device_count() > 1:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# Converting the lines to BERT format
# Thanks to https://www.kaggle.com/httpwwwfszyc/bert-in-keras-taming
def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0
    for text in tqdm(example):
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens_a+["[SEP]"])+[0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)
    return np.array(all_tokens)


def loss_fn(preds, labels):
    preds = preds.view(-1)
    labels = labels.view(-1)
    assert(preds.shape == labels.shape)
    
    loss = nn.BCEWithLogitsLoss()(preds, labels)
    
    return loss


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size


def train(train_loader, model, optimizer, is_distributed):
    model.train()
    
    avg_loss = 0.
    avg_accuracy = 0.
    tk0 = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)
        
    optimizer.zero_grad()

    for i, (x_batch, y_batch) in tk0:

            y_pred = model(x_batch.to(DEVICE),
                           attention_mask=(x_batch > 0).to(DEVICE),
                           labels=None)
            loss = loss_fn(y_pred[0], y_batch.to(DEVICE))
            loss.backward()
            
            if is_distributed and not use_cuda:
                # average gradients manually for multi-machine cpu case only
                _average_gradients(model)
            
            optimizer.step()
            optimizer.zero_grad()
        
            avg_loss += loss.item() / len(train_loader)
            avg_accuracy += torch.mean(
            ((torch.sigmoid(y_pred[0]) >= 0.5) == (y_batch >= 0.5).to(DEVICE)).to(torch.float)).item() / len(train_loader)

            tk0.set_postfix(loss=loss.item(), avg_loss=avg_loss)
    
    log = OrderedDict([('avg_loss', avg_loss), ('avg_acc', avg_accuracy)])
    tk0.close()
    
    return log


# Run validation
def evaluate(valid_loader, model):
    model.eval()
    
    avg_loss = 0.
    valid_preds = []
    valid_trues = []
    
    with torch.no_grad():
        tk0 = tqdm(valid_loader)
        for i, (x_batch, y_batch) in enumerate(tk0):
            y_pred = model(x_batch.to(DEVICE),
                         attention_mask=(x_batch > 0).to(DEVICE), 
                         labels=None)
            
            loss = loss_fn(y_pred[0], y_batch.to(DEVICE))
            avg_loss += loss.item() / len(valid_loader)
            outputs_np = torch.sigmoid(y_pred[0]).cpu().detach().numpy()
            targets_np = y_batch.unsqueeze(1).numpy()
            valid_preds.append(outputs_np)
            valid_trues.append(targets_np)
    
    valid_preds = np.vstack(valid_preds)
    valid_trues = np.vstack(valid_trues)
    acc = accuracy_score((valid_trues >= 0.5), (valid_preds >= 0.5))
    val_log = OrderedDict([('val_loss', avg_loss), ('val_acc', acc)])
    tk0.close()
    
    return val_log

    
if __name__ == '__main__':
    
    # Receive hyperparameters passed via create-training-job API
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=5e-6)
    
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    parser.add_argument('--backend', type=str, default=None, 
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--val', type=str, default=os.environ.get('SM_CHANNEL_VAL'))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))

    args = parser.parse_args()
    
    # Set hyperparameters after parsing the arguments
    batch_size = args.batch_size
    lr = args.learning_rate
    num_epochs = args.epochs
    current_host = args.current_host
    hosts = args.hosts
    model_dir = args.model_dir
    training_dir = args.train
    val_dir = args.val
    
    #is_distributed = len(args.hosts) > 1 and args.backend is not None
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    
    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ['RANK'] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info('Initialized the distributed environment: \'{}\' backend on {} nodes. '.format(
            args.backend, dist.get_world_size()) + 'Current host rank is {}. Number of gpus: {}'.format(
            dist.get_rank(), args.num_gpus))
    
    # fix seed
    seed_torch()
    
    # Data loading
    train_df = pd.read_csv(os.path.join(training_dir, 'train.tsv'), sep ='\t')
    valid_df = pd.read_csv(os.path.join(val_dir, 'valid.tsv'), sep ='\t')
    
    # convert BERT dataset
    tr_sequences = convert_lines(train_df["review_body"].fillna("DUMMY_VALUE"),
                                 MAX_SEQUENCE_LENGTH, tokenizer)
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(tr_sequences, dtype=torch.long), 
                                                   torch.tensor(train_df['star_rating'].values, dtype=torch.float))
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
    
    val_sequences = convert_lines(valid_df["review_body"].fillna("DUMMY_VALUE"),
                                  MAX_SEQUENCE_LENGTH, 
                                  tokenizer)
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(val_sequences, dtype=torch.long),
                                                   torch.tensor(valid_df['star_rating'].values, dtype=torch.float))
    valid_loader = torch.utils.data.DataLoader(valid_dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=False)
    
    # Load pre-trained bert model 
    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', 
                                                          num_labels=1)
    
    model.zero_grad()
    model = model.to(DEVICE)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, 
                      lr=lr, eps=1e-8)
    
    
    if is_distributed and DEVICE != 'cpu':
        # multi-machine multi-gpu case
        model = nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = nn.DataParallel(model)

    
    es = EarlyStopping(patience=5, mode="max")
    path = os.path.join(args.model_dir, 'model.pth')

    for epoch in range(num_epochs):
    
        log = train(train_loader, model, optimizer, is_distributed)
        val_log = evaluate(valid_loader, model)
        
        es(val_log["val_acc"], model, model_path=path)
        
        if es.early_stop:
            logger.info("Early stopping")
            break
        

def model_fn(model_dir):
    """
    Load the gluon model. Called once when hosting service starts.

    :param: model_dir The directory where model files are stored.
    :return: a model (in this case a Gluon network)
    """

    model = BertForSequenceClassification.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking', 
                                                          num_labels=1)
    model = torch.nn.DataParallel(model)
    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    return {"net": model, "tokenizer": tokenizer}


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
    model = net["net"]
    tokenizer = net["tokenizer"]
    model.to(DEVICE)
    
    # Assume one line of text
    parsed = json.loads(data)
    logging.info("Received_data: {}".format(parsed))
    parsed = tokenizer.tokenize(parsed)
    
    #added by manome
    if len(parsed) > MAX_SEQUENCE_LENGTH:
        parsed = parsed[:MAX_SEQUENCE_LENGTH-2]
    
    logging.info("Tokens: {}".format(parsed))
    #x_batch = tokenizer.convert_tokens_to_ids(["[CLS]"]+parsed+["[SEP]"])+[0] * (MAX_SEQUENCE_LENGTH - len(parsed) - 2)
    x_batch = tokenizer.convert_tokens_to_ids(["[CLS]"]+parsed+["[SEP]"]) # do not zero padding
    x_batch = torch.LongTensor(x_batch).unsqueeze(0)    
    
    model.eval()
    with torch.no_grad():
        output = model(x_batch.to(DEVICE), 
                     attention_mask=(x_batch>0).to(DEVICE), 
                     labels=None)
    response_body = json.dumps(torch.sigmoid(output[0]).cpu().detach().numpy().tolist()[0])
    return response_body, output_content_type
