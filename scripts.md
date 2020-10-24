
## **Scripts used**


```bash

###############################
###############################
####                       ####
####  Pretraining scripts  ####
####                       ####
###############################
###############################


#####################################
##
##  Electra Small
##
##  TPU v3-8 ( ~ 2.4 days )  
##           ( ~ 19 steps / s )
##
#####################################

python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_small \
  --hparams '{"model_size":"small","num_train_steps":4000000,"use_tpu":true,"num_tpu_cores":8,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-1","max_seq_length":384,"learning_rate":0.0005,"train_batch_size":128,"save_checkpoints_steps":50000,"iterations_per_loop":1000}'


#####################################
##
##  Electra Base
##
##  TPU v3-8 ( ~ 8 days ) 
##           ( ~ 1.754 steps / s )
##
#####################################

python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_base \
  --hparams '{"model_size":"base","num_train_steps":1200000,"use_tpu":true,"num_tpu_cores":8,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-2","max_seq_length":384,"learning_rate":0.0002,"train_batch_size":256,"save_checkpoints_steps":50000,"iterations_per_loop":1000}'


##########################################################
##
##  Electra x Albert (L-12, H-2024)
##
##  TPU Pod v3-32 ( ~ 3.3 days )  
##                ( ~ 2.776 steps / s )
##                ( stopped at 700000 steps 
##                  because I got 7 day access but 2 days 
##                  spent on  large model and found that 
##                  v3-32 is not enough to train it, due 
##                  to smaller batch size )
##
##########################################################

python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_albert_yue_384_12_2048 \
  --hparams '{"model_size":"large","num_train_steps":800000,"use_tpu":true,"num_tpu_cores":32,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-3","max_seq_length":384,"model_hparam_overrides":{"num_hidden_layers":12,"hidden_size":2048,"embedding_size":128},"generator_hidden_size":0.25,"learning_rate":0.0002,"train_batch_size":256,"save_checkpoints_steps":50000,"iterations_per_loop":1000,"mask_prob": 0.25}'


##############################
##############################
####                      ####
####  Finetuning scripts  ####
####                      ####
##############################
##############################


#####################
##
##  Electra Small
##
#####################

FINETUNE_DATA_DIR=$BUCKET/finetuning_data

python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_small \
  --hparams '{"model_size": "small","task_names": ["mnli"],"vocab_size":32056,"max_seq_length":128,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":3,"weight_decay_rate":0,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_small \
  --hparams '{"model_size": "small","task_names": ["squad"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":2,"weight_decay_rate":0,"layerwise_lr_decay":0.8,"raw_data_dir":"'$FINETUNE_DATA_DIR'","preprocessed_data_dir":"'$FINETUNE_TFRD_DIR'","answerable_uses_start_logits":false,"joint_prediction":false}'
  

python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_small \
  --hparams '{"model_size": "small","task_names": ["drcd"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":2,"weight_decay_rate":0,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'","answerable_uses_start_logits":false,"joint_prediction":false}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_small \
  --hparams '{"model_size": "small","task_names": ["drcd"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":2,"weight_decay_rate":0,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'","answerable_uses_start_logits":false,"joint_prediction":false,"init_checkpoint":""}'


#####################
##
##  Electra Base
##
#####################


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_base \
  --hparams '{"model_size": "base","task_names": ["mnli"],"vocab_size":32056,"max_seq_length":128,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-2","train_batch_size":32,"learning_rate":1e-4,"num_train_epochs":3,"weight_decay_rate":0,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_base \
  --hparams '{"model_size": "base","task_names": ["squad"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-2","train_batch_size":48,"learning_rate":1e-4,"num_train_epochs":2,"weight_decay_rate":0.01,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'","preprocessed_data_dir":"'$FINETUNE_TFRD_DIR'","answerable_uses_start_logits":false,"joint_prediction":false}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_base \
  --hparams '{"model_size": "base","task_names": ["drcd"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-2","train_batch_size":32,"learning_rate":1.4e-4,"num_train_epochs":2,"weight_decay_rate":0.01,"layerwise_lr_decay":0.8,"raw_data_dir":"'$FINETUNE_DATA_DIR'","preprocessed_data_dir":"'$FINETUNE_TFRD_DIR'","answerable_uses_start_logits":false,"joint_prediction":false}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_384_base \
  --hparams '{"model_size": "base","task_names": ["drcd"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-1","train_batch_size":32,"learning_rate":1.4e-4,"num_train_epochs":2,"weight_decay_rate":0.01,"layerwise_lr_decay":0.8,"raw_data_dir":"'$FINETUNE_DATA_DIR'","answerable_uses_start_logits":false,"joint_prediction":false,"init_checkpoint":""}'


#######################################
##
##  Electra x Albert (L-12, H-2024)
##
#######################################


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_albert_yue_384_12_2048 \
  --hparams '{"model_size": "large","task_names": ["mnli"],"vocab_size":32056,"max_seq_length":128,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-3","train_batch_size":128,"learning_rate":5e-5,"num_train_epochs":2,"model_hparam_overrides":{"num_hidden_layers":12,"hidden_size":2048,"embedding_size":128},"weight_decay_rate":0,"layerwise_lr_decay":0,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_albert_yue_384_12_2048 \
  --hparams '{"model_size": "large","task_names": ["squad"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-3","train_batch_size":48,"learning_rate":3e-5,"num_train_epochs":2,"model_hparam_overrides":{"num_hidden_layers":12,"hidden_size":2048,"embedding_size":128},"weight_decay_rate":0,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'


python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_albert_yue_384_12_2048_1.2M \
  --hparams '{"model_size": "large","task_names": ["drcd"],"vocab_size":32056,"max_seq_length":384,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-18","train_batch_size":32,"learning_rate":5e-5,"num_train_epochs":2,"model_hparam_overrides":{"num_hidden_layers":12,"hidden_size":2048,"embedding_size":128},"weight_decay_rate":0,"layerwise_lr_decay":0.85,"raw_data_dir":"'$FINETUNE_DATA_DIR'","write_test_outputs":true,"answerable_uses_start_logits":false,"joint_prediction":false,"init_checkpoint":"$DATA_DIR/corpus_tf_384_9/models/electra_albert_yue_384_12_2048/finetuning_models/squad_model_1"}'

```
