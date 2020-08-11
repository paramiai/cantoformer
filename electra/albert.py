
        
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

import math
gelu_new_K = math.sqrt(2 / math.pi)

def gelu_fast(x):
    return 0.5 * x * (1 + torch.tanh(gelu_new_K * (x + 0.044715 * torch.pow(x, 3))))
    


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
                 eval_num_samples=100000, 
                 cooked=None):
        self.seq_length = seq_length
        self.batch_size = batch_size
        self.data_path = data_path
        
        self.eval_num_samples = (eval_num_samples // batch_size)
        self.eval = eval
        
        try:
            if cooked == False:
                raise

            self.length = os.stat(data_path+"data_original_cooked").st_size//(seq_length*2) // batch_size
            path = data_path+"data_original_cooked"
            with open(path, 'rb') as f:
                pass
            cooked = True
        except:
            if cooked == True:
                raise Exception ('Not yet cooked')
            self.length = os.stat(data_path+"data_original").st_size//(seq_length*2) // batch_size
            cooked = False


        self.actual_length = self.length


        if self.length < eval_num_samples:
            self.eval_num_samples = eval_num_samples = ((self.length // 10) // batch_size)

        if eval:
            self.length = eval_num_samples
        
            
        self.cooked = cooked

        self.ids_bin_buffer = None
        self.dataset = None


    def __len__(self):
        return self.eval_num_samples if self.eval else self.length - self.eval_num_samples
    

    def __getitem__(self, i):
        if self.dataset is None:
            from data_utils import textDataset as _textDataset
            self.dataset = _textDataset(
                data_path=self.data_path, 
                seq_length=self.seq_length, 
                batch_size=self.batch_size, 
                eval=self.eval, 
                eval_num_samples=self.eval_num_samples, 
                cooked=self.cooked
                )

        return self.dataset.__getbatch__(i*self.batch_size, self.batch_size)

    def __getbatch__(self, i):
        return self.dataset(i, self.batch_size)


#####################################
## 
##             Modelling
## 
#####################################





from transformers import get_linear_schedule_with_warmup
from transformers import AlbertForMaskedLM
from transformers.activations import get_activation

from transformers import AlbertConfig

tokenizer = None

    
def get_model(args):
    if args.model_size == 'debug':
        num_hidden_layers = 1
        embedding_size = 8
        hidden_size = 16
        intermediate_size = 32
        num_attention_heads = 2
        args.gen_ratio = 2

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
    

    config = AlbertConfig(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        hidden_size = hidden_size // args.gen_ratio,
        intermediate_size = intermediate_size // args.gen_ratio,
        num_attention_heads=num_attention_heads // args.gen_ratio,
    )

    model = AlbertForMaskedLM(
        config
    )
    return model

def get_loss(model, sample, args, device, gpu=0, report=False):

    generator_input, generator_labels, generator_mask, generator_original = sample

    if gpu:
        generator_input = generator_input.to(device)
        generator_labels = generator_labels.to(device)
        generator_mask = generator_mask.to(device)


    loss, scores = model(generator_input, 
                         attention_mask=generator_mask, 
                         labels=generator_labels)

    total_loss = loss

    log = None
    if report:
        log = OrderedDict()
        log['loss'] = total_loss


    return total_loss, log


def log_formatter(log):
    pass
def get_tokenizer(args):
    return CanTokenizer(vocab_file = args.vocab_file)

def set_parser(parser):
    parser.add_argument('--model-size', default='small',
                        help='model size, '
                                'e.g., "small", "base", "large" (default: small)')
    parser.add_argument("--gen-ratio", default=4, type=int,
                        help="discriminator to generator ratio")
    parser.add_argument("--disc-weight", default=50, type=int,
                        help="discriminator loss scalar")

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
    pass

def evaluate(model, sample, args, device, record, gpu=0, report=False):
    pass

def post_evaluate(record, args):
    pass



get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset