
        
#####################################
## 
##             Imports
## 
#####################################


import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

import time
import datetime

from torch.utils.data import Dataset, TensorDataset, DataLoader

from cantokenizer import CanTokenizer



import warnings

warnings.filterwarnings("ignore")

def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

#####################################
## 
##            Data Utils
## 
#####################################


import numpy as np
import os
class textDataset(Dataset):
    def __init__(self, 
                 data_path, 
                 seq_length,
                 batch_size, 
                 eval=False, 
                 eval_num_samples=0, 
                 cooked=True):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.eval_num_samples = (eval_num_samples // batch_size)
        self.eval = eval

        assert cooked
        

        self.length = os.stat(data_path+"ids").st_size//(seq_length*2) // batch_size
        path = data_path+"data_original_cooked"
        with open(path, 'rb') as f:
            pass

            
        self.cooked = cooked

        self.ids_bin_buffer = None
        self.dataset = None


    def __len__(self):
        return self.length
    

    def __getitem__(self, i):
        if self.dataset is None:
            data_path = self.data_path
            prefix = 'devm_' if self.eval else 'train_'
            path = data_path+prefix+"ids" 
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap

            path = data_path+prefix+"mask"
            self.masked_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.masked_bin_buffer = self.masked_bin_buffer_mmap 

            path = data_path+prefix+"labels"
            self.labels_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.labels_bin_buffer = self.labels_bin_buffer_mmap
        

        return self.dataset.__getbatch__(i*self.batch_size)

                
    def __getbatch__(self, i, size):
        assert self.cooked
        if self.ids_bin_buffer is None:
            data_path = self.data_path
            prefix = 'devm_' if self.eval else 'train_'

            path = data_path+prefix+"ids" 
            self.ids_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.ids_bin_buffer = self.ids_bin_buffer_mmap

            path = data_path+prefix+"mask"
            self.attention_mask_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.attention_mask_bin_buffer = self.masked_bin_buffer_mmap 

            path = data_path+prefix+"labels"
            self.labels_bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self.labels_bin_buffer = self.labels_bin_buffer_mmap
        
        if self.cooked:
            start = seq_length*i*2
            shape = (size,self.seq_length)
            
            ids_buffer = np.frombuffer(self.ids_bin_buffer, dtype=np.int16, count=seq_length*size, offset=start).reshape(shape)
            labels_buffer = np.frombuffer(self.labels_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)
            attention_mask_buffer = np.frombuffer(self.attention_mask_bin_buffer, dtype=np.int8, count=seq_length*size, offset=start // 2).reshape(shape)

            return (
                torch.LongTensor(ids_buffer),
                torch.LongTensor(attention_mask_buffer), 
                torch.LongTensor(labels_buffer), 
            )

#####################################
## 
##             Modelling
## 
#####################################





from transformers import get_linear_schedule_with_warmup


tokenizer = None


from transformers.modeling_bert import BertLayerNorm


class NLIPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = BertLayerNorm(3)
        self.dense = nn.Linear(config.hidden_size, 3)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = gelu_fast(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states
        
        
class ElectraNLI(nn.Module):
    def __init__(self, args, dis_config):
        super().__init__()
        self.args = args
        import importlib
        
        module = importlib.import_module(args.base_model)
        
        self.discriminator = module.ElectraForPreTraining(dis_config)
        self.nli_predictions = NLIPredictions(dis_config)	
        self.loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        
    def forward(self, ids, mask, labels=None):
                
        device = ids.device
        logits = self.discriminator(ids, attention_mask=mask)[0]
        pred = nli_predictions(logits)
        outputs = (pred,)
        if labels != None:
            loss = self.loss_fct(pred, labels)
            outputs += (loss,)
        return outputs


from transformers import ElectraConfig
    
    
def get_model(args):
    if args.model_size == 'debug':
        num_hidden_layers = 1
        embedding_size = 8
        hidden_size = 16
        intermediate_size = 32
        num_attention_heads = 2

    elif args.model_size == 'tiny':
        num_hidden_layers = 4
        embedding_size = 128
        hidden_size = 336
        intermediate_size = 1344
        num_attention_heads = 12
    elif args.model_size == 'small':
        num_hidden_layers = 12
        embedding_size = 128
        hidden_size = 256
        intermediate_size = 1024
        num_attention_heads = 4
    elif args.model_size == 'base':
        num_hidden_layers = 12
        embedding_size = 768
        hidden_size = 768
        intermediate_size = 3072
        num_attention_heads = 12

    else:
        raise Exception('Which model? small, base, large')
    

    discriminator_config = ElectraConfig(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )

    model = ElectraNLI(
        args, 
        dis_config = discriminator_config
    )
    return model

def get_loss(model, sample, args, device, gpu=0, report=False):

    ids, mask, labels = sample

    if gpu:
        ids = ids.to(device)
        mask = mask.to(device)
        labels = labels.to(device)

    pred, loss = model(ids, mask, labels)


    log = None
    if report:
        log = OrderedDict()
        log['acc'] = (pred.argmax(-1) == labels).sum()
        log['acc_tot'] = labels.shape[0]

    return loss, log


def evaluate(model, sample, args, device, record, gpu=0, report=False):
    ids, mask, labels = sample

    if gpu:
        ids = ids.to(device)
        mask = mask.to(device)
        
    pred = model(ids, mask, labels)[0]

    if 'correct_tot' not in record:
        record['correct_tot'] = 0
    if 'correct' not in record:
        record['correct'] = 0

    record['correct'] += (pred.argmax(-1) == labels).sum()
    record['correct_tot'] += labels.shape[0]

def post_evaluate(record, args):
    record['accuracy'] = record['correct'] / record['correct_tot']


def log_formatter(log):
    log['acc'] = log['acc'] / log['acc_tot']


def get_tokenizer(args):
    return CanTokenizer(vocab_file = args.vocab_file)

def set_parser(parser):
    parser.add_argument('--base-model', help='model file to import')
    parser.add_argument('--model-size', default='small',
                        help='model size, '
                                'e.g., "small", "base", "large" (default: small)')

    parser.add_argument('--cooked', action='store_true', help='whether data are cooked')


def get_dataset(args):
    return textDataset(
        args.data, 
        args.seq_length, 
        args.batch_size,
        eval_num_samples=0,
        cooked=args.cooked
    )

def get_eval_dataset(args):
    return textDataset(
        args.data, 
        args.seq_length, 
        args.batch_size,
        eval_num_samples=0,
        cooked=args.cooked,
        eval=True
    )
    pass



get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset