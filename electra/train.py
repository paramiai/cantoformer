


##############################################################################
##############################################################################
####
####   Added util to make decayed learning rates in layers of transformer
####
##############################################################################
##############################################################################



import re
def model_get_parameters(model, 
                         lr, 
                         lw_lr_decay=1, #0.908517, 
                         weight_decay=None):
    named_parameters = list(model.named_parameters())
    lr_factors = []
    no_decay = [
        "bias",
        "LayerNorm",
        "layer_norm"
    ]
    layers = set()
    for k, v in named_parameters:
        r = re.search(r'.layers.(\d+)',k)
        if r:
            layer = int(r.group(1))
            layers.add(layer)

    num_layers = len(layers)

    for k, v in named_parameters:
        if not v.requires_grad:
          continue
        param = {
            'params': v,
        }
        factor = 1
        if lw_lr_decay and lw_lr_decay != 1:
            r = re.search(r'.layers.(\d+)',k)
            if r:
                layer = int(r.group(1))
                factor = lw_lr_decay**(num_layers-layer)

            elif 'embed_tokens.weight' in k or 'embed_positions' in k:
                layer = 0
                factor = lw_lr_decay**(num_layers-layer)
 
        param['lr'] = lr * factor

        if weight_decay and weight_decay != 0:
            wd = 0 if any(e in k for e in no_decay) else weight_decay
            param['weight_decay'] = wd
        
        lr_factors.append(param)
    return lr_factors
      
      



##############################################################################
##############################################################################
####
####   Training
####
##############################################################################
##############################################################################




from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
try:
    import torch_xla
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.data_parallel as dp
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.utils.utils as xu
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp
    import torch_xla.test.test_utils as test_utils
    tpu_enabled = True
except:
    tpu_enabled = False
from transformers.optimization import get_linear_schedule_with_warmup
from collections import OrderedDict

from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
try:
    torch.backends.cudnn.benchmark = True
except:
    pass
from torch.nn.parallel import DistributedDataParallel as DDP
try:
    from apex.parallel import DistributedDataParallel as DDP

    from apex import amp
    import apex
    apex_enabled = True
except:
    apex_enabled = False

def format_log(log, formatter):
    args = []
    for k, v in log.items():
        try: log[k] = v.item() 
        except: pass
    formatter(log)
    for k, v in log.items():
        try: v = v.item()
        except: pass
        if isinstance(v, float):
            if v < 0.001:
                v = ('%.2E' %v).replace('E','e').replace('-0', '-')
            elif v < 0.1:
                v = '%.4f'%v
            elif v < 100:
                v = '%.2f'%v
            else:
                v = '%.f'%v
        args.append((k, v))

    return ' | '.join(k+':'+str(v) for k, v in args)

def _train_update(log, formatter):
    xm.master_print(format_log(log, formatter))

    

def args_float(x):
    try:
        x = float(eval(x))
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))
    return x


def _catalog_shared_params(module, memo=None, prefix=''):
    if memo is None:
        first_call = True
        memo = {}
    else:
        first_call = False
    for name, param in module._parameters.items():
        param_prefix = prefix + ('.' if prefix else '') + name
        if param not in memo:
            memo[param] = []
        memo[param].append(param_prefix)
    for name, m in module._modules.items():
        if m is None:
            continue
        submodule_prefix = prefix + ('.' if prefix else '') + name
        _catalog_shared_params(m, memo, submodule_prefix)
    if first_call:
        return [x for x in memo.values() if len(x) > 1]


def do_share_parameters_again(model, shared_parameters, log=False):
    if shared_parameters:
        if log:
            print('Sharing parameters for TPU', flush=True)

        for k, v in shared_parameters.items():
            if k.endswith('.weight'):
                k = k[:-7]
            for e in v:
                if e.endswith('.weight'):
                    e = e[:-7]

                statement = 'model.'+e+' = ' + 'model.'+k
                if log:
                    print(statement, flush=True)
                exec(statement)



def train(rank, args):
    print('enter train @ %s'%(rank), flush=True)
    args.rank = rank
    torch.manual_seed(42)

    tokenizer = get_tokenizer(args)
    args.vocab_size = tokenizer._tokenizer.get_vocab_size()
    
    train_dataset = get_dataset(args)

    if args.total_num_updates < 100:
        args.total_num_updates = len(train_dataset) * args.total_num_updates

    if args.warmup_updates < 1:
        args.warmup_updates = int(args.total_num_updates * args.warmup_updates)
    else:
        args.warmup_updates = int(args.warmup_updates)
    
    train_sampler = None
    if args.gpus:
        dist.init_process_group(
            'nccl', 
            rank=rank, 
            world_size=args.world_size
        )
        if args.gpus > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=args.gpus,
                rank=rank,
                shuffle=False)

    else:
        rank = xm.get_ordinal()
        if xm.xrt_world_size() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=rank,
                shuffle=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size if not hasattr(train_dataset, '__getbatch__') else None,
        sampler=train_sampler,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers)
        

    eval_loader = None
    if args.eval_dir:
    
        eval_sampler = None
        if args.gpus:
            dist.init_process_group(
                'nccl', 
                rank=rank, 
                world_size=args.world_size
            )
            if args.gpus > 1:
                traieval_samplern_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=args.gpus,
                    rank=rank,
                    shuffle=False)

        else:
            rank = xm.get_ordinal()
            if xm.xrt_world_size() > 1:
                eval_sampler = torch.utils.data.distributed.DistributedSampler(
                    train_dataset,
                    num_replicas=xm.xrt_world_size(),
                    rank=rank,
                    shuffle=False)
                
        eval_dataset = get_eval_dataset(args)
        eval_loader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=args.batch_size if not hasattr(train_dataset, '__getbatch__') else None,
            sampler=eval_sampler,
            pin_memory=True,
            shuffle=False,
            num_workers=args.num_workers)

    if args.gpus:
        assert apex_enabled
        torch.cuda.set_device(rank)


        ##########################
        ##
        ##  Model Creation
        ##
        ##########################
        model = get_model(args)

        model.cuda(rank)

        device = torch.device('cuda:'+str(rank))

        ##########################
        ##
        ##  Init Optimizer
        ##
        ##########################

        optimizer = apex.optimizers.FusedAdam(
            model_get_parameters(model,
                                 lr=args.lr,
                                 lw_lr_decay=args.lw_lr_decay,
                                 weight_decay=args.weight_decay
                                 ),  

                                 # use this function to set extra optimizer arguments, 
                                 # see model_get_parameters
            betas=(0.9, 0.999), 
            eps=1e-6,
            lr=args.lr, 
            weight_decay=args.weight_decay
        )




        model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
        model = DDP(model)
        batches = train_loader

    else:
        assert tpu_enabled
        device = xm.xla_device()


        ##########################
        ##
        ##  Model Creation
        ##
        ##########################
        
        model = get_model(args)


        ##########################
        ##
        ##  For shared parameters, TPU requires modules to be tied after .to(device)
        ##  So we first find the shared parameters first
        ##
        ##########################

        shared_parameters = {e[0]: e[1:] for e in _catalog_shared_params(model)}

        model.to(device)
        
        do_share_parameters_again(model, shared_parameters, log = rank == 0)



        ##########################
        ##
        ##  Init Optimizer
        ##
        ##########################

        optimizer = optim.Adam(
            model_get_parameters(model,
                                 lr=args.lr,
                                 lw_lr_decay=args.lw_lr_decay,
                                 weight_decay=args.weight_decay
                                 ),  

                                 # use this function to set extra optimizer arguments, 
                                 # see model_get_parameters
            lr=args.lr,
            weight_decay=args.weight_decay
        )


        writer = None
        if xm.is_master_ordinal():
            writer = test_utils.get_summary_writer(args.save_dir)
                
        xm.rendezvous("load_checkpoint")  # wait for all workers
        xm.mark_step()

        # tracker = xm.RateTracker()
    if args.restore_file:
        states = torch.load(args.restore_file, map_location=device)
        for k, v in list(states.items()):
            if k.startswith('module.'):
                del states[k]
                k = k[7:]
                states[k] = v
            if k.endswith('position_ids'):
                del states[k]
                states[k[:-12] + 'position_embeddings'] = v
        try:
            model.load_state_dict(states)
        except Exception as err:
            import traceback
            traceback.print_exc()
            model.load_state_dict(states, strict=False)
            
        
    model.train()

    if args.anomaly_detection and rank == 0:
        torch.set_anomaly_enabled(True)

    ##########################
    ##
    ##  Init LR Scheduler
    ##
    ##########################

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=args.warmup_updates, 
        num_training_steps=args.total_num_updates, 
    )

    step_i = 0

    err = None
    try:
        if rank == 0:
            pbar = tqdm(total=args.total_num_updates)
        while step_i < args.total_num_updates:
            if not args.gpus:
                batches = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
            for sample in batches:
                step_i += 1
                if step_i > args.total_num_updates:
                    break

                report_step = step_i % args.log_interval == 0

                while True: # the loop only for apex Gradient Overflow
                    optimizer.zero_grad()

                    total_loss, log = get_loss(
                        model, 
                        sample, 
                        args=args, 
                        device=device, 
                        gpu=args.gpus, 
                        report=report_step
                    )

                    if args.gpus:
                        default_optimizer_step = optimizer.step

                        with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                            scaled_loss.backward()

                        # If Amp detects an overflow, it patches optimizer.step.  In other words, if optimizer.step
                        # was left unpatched, there was no overflow, and we don't need to replay.
                        if optimizer.step is default_optimizer_step:
                            optimizer.step()
                            break

                        optimizer.step() # If an overflow was detected, "optimizer.step" is the patched call, which does 
                                         # nothing but restore optimizer.step to default_optimizer_step.
                        if rank == 0:
                            print("Overflowed, reducing loss scale and replaying batch.", flush=True)
                        
                    else:
                        total_loss.backward()
                        xm.optimizer_step(optimizer)
                        xm.mark_step()

                        break



                scheduler.step()

                if report_step:
                    if 'loss' not in log:
                        log['loss'] = total_loss

                    if args.gpus:
                        if rank == 0:
                            pbar.set_description(format_log(log, log_formatter))
                    else:
                        xm.add_step_closure(_train_update, args=(log, log_formatter))

                    if args.report_metrics:
                        xm.master_print(met.metrics_report())

                
                if rank == 0:
                    pbar.update(1)

        if eval_loader is not None:
            model.eval()
            if not args.gpus:
                batches = pl.ParallelLoader(eval_loader, [device]).per_device_loader(device)
            with torch.no_grad():
                record = OrderedDict()

                for sample in batches:
                    evaluate(
                        model, 
                        sample, 
                        args=args, 
                        device=device, 
                        record=record,
                        gpu=args.gpus, 
                        report=report_step
                    )



                post_evaluate(record, args=args)

            import json
            print('',flush=True)
            print(json.dumps(record),flush=True)
            print('',flush=True)
        

    except Exception as _err:
        err = _err
    finally:
        save_fn = os.path.join(args.save_dir, 'checkpoint_final.pt')
        folder = os.path.split(os.path.abspath(save_fn))[0]
        os.makedirs(folder, exist_ok=True)
        if rank == 0 and args.gpus:
            torch.save(model.state_dict(), save_fn )
            if err:
                raise err
        else:
            xm.save(model.state_dict(), save_fn )
            if err:
                raise err

import importlib


    


parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to data')
                    
parser.add_argument('--model', help='model file to import')

args, unknown = parser.parse_known_args()

module = importlib.import_module(args.model)
get_model = module.get_model
get_loss = module.get_loss
log_formatter = module.log_formatter
get_tokenizer = module.get_tokenizer
evaluate = module.evaluate
post_evaluate = module.post_evaluate
set_parser = module.set_parser
get_dataset = module.get_dataset
get_eval_dataset = module.get_eval_dataset


get_model
get_loss
log_formatter 
get_tokenizer
evaluate
post_evaluate
set_parser
get_dataset




set_parser(parser)

parser.add_argument('--restore-file', default='',
                    help='pt file to be restored')

parser.add_argument('--eval-dir', default='',
                    help='path to eval data')

parser.add_argument('--vocab-file', default='',
                    help='vocab file of tokenizer')
parser.add_argument( '--batch-size', type=int, 
                    help='maximum number of sentences in a batch')
                    
parser.add_argument( '--seq-length', type=int, 
                    help='maximum number of tokens in a sentence')
parser.add_argument('--warmup-updates', default=10000, type=args_float, 
                    help='warmup the learning rate linearly for the first N updates')

parser.add_argument('--total-num-updates', default=1000000, type=int)
parser.add_argument('--lr', default=0.0005, type=args_float)
parser.add_argument('--weight-decay', default=0.01, type=args_float)
parser.add_argument('--lw-lr-decay', default=1, type=args_float, 
                    help='layer-wise learning rate decay')

parser.add_argument('--tpu', action='store_true', help='use TPU instead of CUDA')
parser.add_argument('--report-metrics', action='store_true', help='use TPU instead of CUDA')

parser.add_argument('--log-interval', type=int, default=100,
                    help='log progress every N batches (when progress bar is disabled)')

parser.add_argument('--save-dir', metavar='DIR', default='checkpoints',
                    help='path to save checkpoints')

parser.add_argument('--gpus', default=torch.cuda.device_count(), type=int,
                    help='number of gpus per node')
parser.add_argument( '--num-cores', default=8, type=int, 
                    help='num of tpu cores')
parser.add_argument( '--num-workers', default=1, type=int, 
                    help='num of preprocessing workers')
parser.add_argument( '--anomaly-detection', action='store_true', help='enable pytorch anomaly detection')

args = parser.parse_args()

if __name__ == '__main__':


    if args.gpus:
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        args.world_size = args.gpus # * (args.num_workers or 1)
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    else:
        xmp.spawn(train, args=(args,), nprocs=args.num_cores)




