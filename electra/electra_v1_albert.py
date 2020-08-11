
        
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
from transformers import AlbertPreTrainedModel, AlbertModel
from transformers.activations import get_activation

from transformers import AlbertConfig

tokenizer = None


from transformers.modeling_bert import BertLayerNorm


class ElectraGeneratorPredictions(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.LayerNorm = BertLayerNorm(config.embedding_size)
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)

    def forward(self, generator_hidden_states):
        hidden_states = self.dense(generator_hidden_states)
        hidden_states = gelu_fast(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)

        return hidden_states


class ElectraForMaskedLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.electra = AlbertModel(config)	
        self.generator_predictions = ElectraGeneratorPredictions(config)
        
        self.loss_fct = nn.CrossEntropyLoss(reduction='none')  # -100 index = padding token
        
        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)
        self.init_weights()

    def get_input_embeddings(self):
        return self.electra.get_input_embeddings()

    def get_output_embeddings(self):
        return self.generator_lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        **kwargs
    ):
        if "masked_lm_labels" in kwargs:
            warnings.warn(
                "The `masked_lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("masked_lm_labels")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."

        generator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )
        generator_sequence_output = generator_hidden_states[0]	
        prediction_scores = self.generator_predictions(generator_sequence_output)
        prediction_scores = self.generator_lm_head(prediction_scores)

        loss = None
        # Masked language modeling softmax layer
        if labels is not None:
            '''
            if prediction_scores.device != torch.device('cpu'):
                labels[labels<0] = 0
                
            loss = cross_entropy(
                prediction_scores.view(-1, prediction_scores.size(-1)),
                labels.view(-1),
                reduction='sum',
            )'''
            per_example_loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)).view(labels.shape[0], labels.shape[1])
            loss = ((per_example_loss * attention_mask).sum(1) / (1e-6 + attention_mask.sum(1))).sum()
            
            #loss = self.loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        output = (prediction_scores,)
        return ((loss,) + output) if loss is not None else output

	
class ElectraDiscriminatorPredictions(nn.Module):	
    """Prediction module for the discriminator, made up of two dense layers."""	
    def __init__(self, config):	
        super().__init__()	
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)	
        self.dense_prediction = nn.Linear(config.hidden_size, 1)	
        self.config = config	
    def forward(self, discriminator_hidden_states):	
        hidden_states = self.dense(discriminator_hidden_states)	
        hidden_states = get_activation(self.config.hidden_act)(hidden_states)	
        logits = self.dense_prediction(hidden_states)	
        return logits	

class ElectraForPreTraining(AlbertPreTrainedModel):	
    def __init__(self, config):	
        super().__init__(config)	
        self.electra = AlbertModel(config)	
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)	
        self.init_weights()	
        self.loss_fct = nn.BCEWithLogitsLoss(reduction='none')	


    def get_input_embeddings(self):
        return self.electra.get_input_embeddings()

    def get_output_embeddings(self):
        return self.discriminator_predictions


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask,
            token_type_ids,
            position_ids,
            head_mask,
            inputs_embeds,
            output_attentions,
            output_hidden_states,
        )
        discriminator_sequence_output = discriminator_hidden_states[0]
	
        logits = self.discriminator_predictions(discriminator_sequence_output)	

        output = (logits,)

        if labels is not None:
            losses = ((self.loss_fct(logits.squeeze(-1), labels.float()) * attention_mask).sum(1) / (1e-6 + attention_mask.sum(1))).sum()
            output = (losses,) + output
            

        return output  # (loss), scores, (hidden_states), (attentions)


gg = 0
class Electra(nn.Module):
    def __init__(self, args, gen_config, dis_config):
        super().__init__()
        self.args = args
        self.generator = ElectraForMaskedLM(gen_config)
        self.discriminator = ElectraForPreTraining(dis_config)
        self.sigmoid = nn.Sigmoid()
        self.noise = torch.rand(((args.seq_length, args.vocab_size)))

        self.discriminator_labels = torch.zeros(args.batch_size, args.seq_length, gen_config.hidden_size)

        self.discriminator.electra.embeddings = self.generator.electra.embeddings
        
        
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
                if self.noise.shape[1] != generator_input.shape[1]:
                    self.noise = torch.rand(((generator_input.shape[1], self.args.vocab_size)), device=device)
                elif self.noise.device != device:
                    self.noise = self.noise.to(device)
                noise = self.noise
                noise.uniform_()
                noise.add_(eps).log_().neg_()
                noise.add_(eps).log_().neg_()

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


    
def get_model(args):
    if args.model_size == 'debug':
        num_hidden_layers = 1
        embedding_size = 8
        hidden_size = 16
        num_hidden_groups = 1
        intermediate_size = 32
        num_attention_heads = 2
        args.gen_ratio = 2

    elif args.model_size == 'small':
        num_hidden_layers = 12
        embedding_size = 128
        hidden_size = 256
        num_hidden_groups = 1
        intermediate_size = 1024
        num_attention_heads = 4
    elif args.model_size == 'base':
        num_hidden_layers = 12
        embedding_size = 128
        hidden_size = 768
        num_hidden_groups = 1
        intermediate_size = 3072
        num_attention_heads = 12

    else:
        raise Exception('Which model? small, base, large')
    

    generator_config = AlbertConfig(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        num_hidden_groups=num_hidden_groups,
        hidden_size = hidden_size // args.gen_ratio,
        intermediate_size = intermediate_size // args.gen_ratio,
        num_attention_heads=num_attention_heads // args.gen_ratio,
    )

    discriminator_config = AlbertConfig(
        max_position_embeddings=args.seq_length,
        vocab_size=args.vocab_size,

        num_hidden_layers=num_hidden_layers,
        embedding_size=embedding_size,

        num_hidden_groups=num_hidden_groups,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_attention_heads=num_attention_heads,
    )

    model = Electra(
        args, 
        gen_config = generator_config, 
        dis_config = discriminator_config
    )
    return model

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