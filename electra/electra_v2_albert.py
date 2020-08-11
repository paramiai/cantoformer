
#####################################
## 
##            Data Utils
## 
#####################################


from torch.utils.data import Dataset
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
"""
    Copyright 2019 Tae Hwan Jung
    ALBERT Implementation with forking
    Clean Pytorch Code from https://github.com/dhlee347/pytorchic-bert
"""

""" Transformer Model Classes & Config Class """

import math
import json
from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def _init_weights( module):
    """ Initialize the weights """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()


class Config(NamedTuple):
    "Configuration for BERT model"
    vocab_size: int = None # Size of Vocabulary
    hidden_size: int = 768 # Dimension of Hidden Layer in Transformer Encoder
    intermediate_size: int = 768*4 # Dimension of Intermediate Layers in Positionwise Feedforward Net
    embedding_size: int = 128 # Factorized embedding parameterization
    classifier_dropout_prob: float = 0.1 # finetune dropout
    hidden_dropout_prob: float = 0.1 # finetune dropout
    num_hidden_layers: int = 12 # Numher of Hidden Layers
    num_attention_heads: int = 768//64 # Numher of Heads in Multi-Headed Attention Layers
    #activ_fn: str = "gelu" # Non-linear Activation Function Type in Hidden Layers
    max_position_embeddings: int = 512 # Maximum Length for Positional Embeddings
    n_segments: int = 2 # Number of Sentence Segments

    rank: int = 0

    @classmethod
    def from_json(cls, file):
        return cls(**json.load(open(file, "r")))

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
        embedding_size = 128
        hidden_size = 768
        intermediate_size = 3072
        num_attention_heads = 12

    else:
        raise Exception('Which model? small, base, large')
    

    generator_config = Config(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        hidden_size = hidden_size // args.gen_ratio,
        intermediate_size = intermediate_size // args.gen_ratio,
        num_attention_heads=num_attention_heads // args.gen_ratio,

        rank=args.rank
    )

    discriminator_config = Config(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,

        rank=args.rank
    )

    model = Electra(
        args, 
        gen_config = generator_config, 
        dis_config = discriminator_config
    )
    return model

def merge_last(x, n_dims):
    "merge the last n_dims to a dimension"
    s = x.size()
    assert n_dims > 1 and n_dims < len(s)
    return x.view(*s[:-n_dims], -1)

def split_last(x, shape):
    "split the last dimension to given shape"
    shape = list(shape)
    assert shape.count(-1) <= 1
    if -1 in shape:
        shape[shape.index(-1)] = int(x.size(-1) / -np.prod(shape))
    return x.view(*x.size()[:-1], *shape)


class Embeddings(nn.Module):
    "The embedding module from word, position and token_type embeddings."
    def __init__(self, cfg):
        super().__init__()
        # Original BERT Embedding
        # self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.hidden) # token embedding

        # factorized embedding
        self.tok_embed1 = nn.Embedding(cfg.vocab_size, cfg.embedding_size)
        self.tok_embed2 = nn.Linear(cfg.embedding_size, cfg.hidden_size)

        self.pos_embed = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size) # position embedding
        # self.seg_embed = nn.Embedding(cfg.n_segments, cfg.hidden) # segment(token type) embedding

        self.norm = LayerNorm(cfg.hidden_size)
        # self.drop = nn.Dropout(cfg.classifier_dropout_prob)
        
        self.pos = None

    def forward(self, x):
        seq_len = x.size(1)
        if self.pos is None or self.pos.device != x.device or self.pos.shape[0] != x.shape[1]:
            self.pos = pos = torch.arange(seq_len, dtype=torch.long, device=x.device)
        else:
            pos = self.pos

        # factorized embedding
        e = self.tok_embed1(x)
        e = self.tok_embed2(e)
        e = e + self.pos_embed(pos)
        #return self.drop(self.norm(e))
        return self.norm(e)

class MultiHeadedSelfAttention(nn.Module):
    """ Multi-Headed Dot Product Attention """
    def __init__(self, cfg):
        super().__init__()
        self.proj_q = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.proj_k = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.proj_v = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        # self.drop = nn.Dropout(cfg.p_drop_attn)
        self.scores = None # for visualization
        self.num_attention_heads = cfg.num_attention_heads

    def forward(self, x, mask):
        """
        x, q(query), k(key), v(value) : (B(batch_size), S(seq_len), D(dim))
        mask : (B(batch_size) x S(seq_len))
        * split D(dim) into (H(num_attention_heads), W(width of head)) ; D = H * W
        """
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q, k, v = self.proj_q(x), self.proj_k(x), self.proj_v(x)
        q, k, v = (split_last(x, (self.num_attention_heads, -1)).transpose(1, 2)
                   for x in [q, k, v])
        # (B, H, S, W) @ (B, H, W, S) -> (B, H, S, S) -softmax-> (B, H, S, S)
        scores = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        if mask is not None:
            mask = mask[:, None, None, :].float()
            scores -= 10000.0 * (1.0 - mask)
        #scores = self.drop(F.softmax(scores, dim=-1))
        scores = F.softmax(scores, dim=-1)
        # (B, H, S, S) @ (B, H, S, W) -> (B, H, S, W) -trans-> (B, S, H, W)
        h = (scores @ v).transpose(1, 2).contiguous()
        # -merge-> (B, S, D)
        h = merge_last(h, 2)
        self.scores = scores
        return h


class PositionWiseFeedForward(nn.Module):
    """ FeedForward Neural Networks for each position """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size)
        #self.activ = lambda x: activ_fn(cfg.activ_fn, x)

    def forward(self, x):
        # (B, S, D) -> (B, S, D_ff) -> (B, S, D)
        return self.fc2(gelu_fast(self.fc1(x)))


class Block(nn.Module):
    """ Transformer Block """
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadedSelfAttention(cfg)
        self.proj = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.norm1 = LayerNorm(cfg.hidden_size)
        self.pwff = PositionWiseFeedForward(cfg)
        self.norm2 = LayerNorm(cfg.hidden_size)
        if not hasattr(cfg,'hidden_dropout_prob'):
            setattr(cfg,'hidden_dropout_prob', cfg.classifier_dropout_prob)
        self.drop = nn.Dropout(cfg.hidden_dropout_prob)

        self.config = cfg

    def forward(self, x, mask):
        h = self.attn(x, mask)
        h = self.norm1(x + self.drop(self.proj(h)))
        h = self.norm2(h + self.drop(self.pwff(h)))
        return h


class Transformer(nn.Module):
    """ Transformer with Self-Attentive Blocks"""
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embed = Embeddings(cfg)
        # Original BERT not used parameter-sharing strategies
        # self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_hidden_layers)])

        # To used parameter-sharing strategies
        self.num_hidden_layers = cfg.num_hidden_layers
        self.block = Block(cfg)

    def forward(self, x, mask):
        h = self.embed(x)
        for _ in range(self.num_hidden_layers):
            h = self.block(h, mask)

        return h

class ElectraForMaskedLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = Transformer(config)
        self.dense = nn.Linear(config.hidden_size, config.vocab_size)
        self.norm = LayerNorm(config.vocab_size)
        
        # self.loss_fct = nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token
        self.loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        
        self.apply(_init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        generator_sequence_output = self.transformer(
            input_ids,
            attention_mask,
        )

        hidden_states = self.dense(generator_sequence_output)
        hidden_states = gelu_fast(hidden_states)
        prediction_scores = self.norm(hidden_states)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            #per_example_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)).view(labels.shape[0], labels.shape[1])
            #loss = ((per_example_loss * attention_mask).sum(1) / (1e-6 + attention_mask.sum(1))).sum()
            
            loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,)
        return ((loss,) + output) if loss is not None else output



class ElectraForPreTraining(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.transformer = Transformer(config)

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense_prediction = nn.Linear(config.hidden_size, 1)

        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')
        
        self.apply(_init_weights)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
    ):
        discriminator_sequence_output = self.transformer(
            input_ids,
            attention_mask,
        )

        hidden_states = self.dense(discriminator_sequence_output)
        hidden_states = gelu_fast(hidden_states)
        logits = self.dense_prediction(hidden_states)

        output = (logits,)

        if labels is not None:
            losses = ((self.loss_fct(logits.squeeze(-1), labels.float()) * attention_mask).sum(1) / (1e-6 + attention_mask.sum(1))).sum()
            #loss_fct = nn.CrossEntropyLoss()
            #losses = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))
            output = (losses,) + output

        return output  # (loss), scores, (hidden_states), (attentions)


class Electra(nn.Module):
    def __init__(self, args, gen_config, dis_config):
        super().__init__()
        self.args = args
        self.generator = ElectraForMaskedLM(gen_config)
        self.discriminator = ElectraForPreTraining(dis_config)
        self.sigmoid = nn.Sigmoid()
        self.noise = torch.rand(((args.seq_length, args.vocab_size)))

        self.discriminator_labels = torch.zeros(args.batch_size, args.seq_length, gen_config.hidden_size, dtype=torch.long)
        
        #self.discriminator.transformer.embed.tok_embed1 = self.generator.transformer.embed.tok_embed1
        
    def forward(self, generator_input, generator_labels = None, 
                generator_mask = None, generator_original = None, 
                return_acc=False):
        device = generator_input.device
        if generator_labels != None:
            generator_loss, generator_scores = self.generator(generator_input, 
                                                              attention_mask=generator_mask, 
                                                              labels=generator_labels)

            with torch.no_grad():

                eps = 1e-7 if generator_scores.dtype == torch.float16 else 1e-9
                """
                
                noise = -torch.log(-torch.log(torch.rand((generator_scores.shape[1],generator_scores.shape[-1]), device=device) + 1e-9) + 1e-9)
                discriminator_input = torch.where(generator_labels>=0, 
                                                  torch.argmax((generator_scores + noise  # gumbel noise
                                                                ),dim=2),
                                                  generator_original)
                discriminator_labels = torch.where(discriminator_input == generator_original, 
                                                   torch.zeros_like(generator_original, device=device), 
                                                   torch.ones_like(generator_original, device=device))

                """

                if self.noise.device != device:
                    self.noise = self.noise.to(device)
                noise = self.noise.uniform_().add_(eps).log_().neg_().add_(eps).log_().neg_()

                if generator_original.shape != self.discriminator_labels.shape or device != self.discriminator_labels.device:
                    self.discriminator_labels = torch.zeros(generator_original.shape, device=device)
                else:
                    self.discriminator_labels.zero_()
                
                pred = torch.argmax(generator_scores + noise,dim=2)
                discriminator_input_mask = (generator_labels>=0).long()

                pred *= discriminator_input_mask
                generator_original *= (1-discriminator_input_mask)

                pred += generator_original
                discriminator_input = pred

                self.discriminator_labels.add_((discriminator_input != generator_original).long())
                discriminator_labels = self.discriminator_labels



                
            discriminator_loss, discriminator_scores = self.discriminator(discriminator_input, 
                                                        attention_mask=generator_mask, labels = discriminator_labels)

            # mask  [1,1,1,1,0,0]
            # arg   [1,0,0,1,0,0]
            # label [1,1,1,0,0,0]
            
            acc_true = 1
            acc_true_tot = 1
            acc_false = 1
            acc_false_tot = 1
            gen_corr = 1
            gen_tot = 1
            
            if return_acc:
                with torch.no_grad():
                    gen_label_mask = (generator_labels >= 0).long()
                    gen_corrects = (torch.argmax(generator_scores, dim=-1) == generator_labels) 
                    gen_corr = gen_corrects.sum()
                    gen_tot = gen_label_mask.sum()
                    
                    arg = (self.sigmoid(discriminator_scores).squeeze(-1) > 0.5).long()
                    acc_true = (arg * discriminator_labels).sum() 
                    acc_true_tot = discriminator_labels.sum()

                    neg_arg = (1 - arg)  * generator_mask
                    neg_discriminator_labels = (1 - discriminator_labels) * generator_mask

                    acc_false = (neg_arg * neg_discriminator_labels).sum() 
                    acc_false_tot = neg_discriminator_labels.sum()
                    
                
            return (generator_loss, discriminator_loss, generator_scores, discriminator_scores, 
            (acc_true, acc_true_tot, acc_false, acc_false_tot, gen_corr, gen_tot), discriminator_input, discriminator_labels)
    
        else:
            generator_scores = self.generator(generator_input, attention_mask=generator_mask)
            discriminator_input = torch.argmax(generator_scores[0],dim=2)
            discriminator_scores = self.discriminator(discriminator_input, attention_mask=generator_mask)
            return generator_scores, discriminator_scores



from cantokenizer import CanTokenizer

from collections import OrderedDict

def get_loss(model, sample, args, device, gpu=0, report=False):

    generator_input, generator_labels, generator_mask, generator_original = sample

    if gpu:
        generator_input = generator_input.to(device)
        generator_labels = generator_labels.to(device)
        generator_mask = generator_mask.to(device)
        generator_original = generator_original.to(device)


    (   generator_loss, 
        discriminator_loss, 
        gen_o, dis_o, 
        (acc_true, acc_true_tot, acc_false, acc_false_tot, gen_corr, gen_tot), 
        discriminator_input, 
        discriminator_labels) = model(generator_input, generator_labels, 
                                    generator_mask, generator_original, return_acc=report)

    total_loss = (generator_loss + discriminator_loss * args.disc_weight)

    log = None
    if report:
        log = OrderedDict()
        log['loss'] = total_loss
        log['gloss'] = generator_loss
        log['dloss'] = discriminator_loss
        log['acc_true'] = acc_true
        log['acc_true_tot'] = acc_true_tot
        log['acc_false'] = acc_false 
        log['acc_false_tot'] = acc_false_tot
        log['gen_corr'] = gen_corr
        log['gen_tot'] = gen_tot


    return total_loss, log


def log_formatter(log):
    log['tacc'] = log['acc_true'] / log['acc_true_tot']
    log['facc'] = log['acc_false'] / log['acc_false_tot']
    log['gacc'] = log['gen_corr'] / log['gen_tot']
    del log['acc_true']
    del log['acc_true_tot']
    del log['acc_false']
    del log['acc_false_tot']
    del log['gen_corr']
    del log['gen_tot']

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