<p align="center">
    <br>
    <img src="imgs/粵.png" width="100"/>
    <br>
</p>

<h3 align="center">
<p><b>Cantoformer <br/>廣東話嘅語言 AI</b> </p>
</h3>

Recent advances in AI enable smarter applications based on texts. 
It's good but they are mostly in English due to its abundance of texts available from the Internet.

This repository explores LM in **Cantonese (Yue Chinese, 廣東話)**, a langauge predominantly spoken in Guangzhou, Hong Kong and Macau, and containing very challenging lingual properties for AI to learn.

最近嘅 AI 講到好巴閉咁，又話咩咩 chatbot 可以好似真人咁同人對答，但其實喺呢個「語言處理」嘅領域入面，好多嘅資源都只係得英文，所以要落手做廣東話 NLP，其實唔容易。

所以諗住喺呢度開個 Repo ，鼓勵更多人開發廣東話 AI。

## Challenges

- Mixed Languages (English, Chinese, Yue)
- Complex Syntax
- Scarce Resource
- Many Homonyms & Homophones

## Framework to be used

- `Tensorflow`
- `Pytorch`
- `🤗Transformers`

## Libraries to be used

- `🤗Tokenizers`
- `Jieba`

## Data to be used

- English
  - [Wiki](https://dumps.wikimedia.org/)
  - [BookCorpus](https://github.com/soskek/bookcorpus)
  - News Articles
  - Forum Data

- Chinese/Cantonese
  - [Wiki](https://dumps.wikimedia.org/)
  - News Articles
  - Forum Data



## Something to explore

1. **Sentence Order Prediction (SOP)**

   SOP is a pretraining objective that is used in Albert. StructBERT also introduces Sentence Structural Objective, but since the code for electra reads the data sequentially, this repo explores SOP first

2. **Cluster Objective**
   
   [DocProduct](https://github.com/re-search/DocProduct) is a cool project training a BERT model to cluster similar Q&A -- if a text A answers the question Q, then Q and A will be close in vector representation. 

   This means the model must predict the possible contexts (before and after) in order to embed a vector that can minimize the cost function

   Details refer to the [DocProduct](https://github.com/re-search/DocProduct) repo.




## To Do List

- [x] Normalize Chinese characters
- [ ] ELECTRA-small **(Working)**
- [ ] ELECTRA-small-sop **(Working)**
- [ ] ELECTRA-small-cluster **(Working)**
- [ ] ELECTRA-small-sop-cluster **(Working)**
- [ ] ELECTRA-albert-small **(Working)**
- [ ] ELECTRA-albert-small-sop **(Working)**
- [ ] ELECTRA-base
- [ ] ELECTRA-large
      

## Model Comparison

|     Model     |   params #    |  bs  |  lr  |    L/H    |  MNLI-en  |
| ------------- | -------------:|:----:|-----:|:---------:|:---------:|
|    BERT (b)   |      108M     |      |      |  12/256   |   84.4    |
|    BERT (l)   |      334M     |      |      |  12/256   |   87.1    |
|  alBERT (b)   |      12M      |      |      |  12/768   |   84.6    |
|  alBERT (l)   |      18M      |      |      |  24/1024  |   86.5    |
|  alBERT (xl)  |      60M      |      |      |  24/2048  |   87.9    |
|  alBERT (xxl) |      235M     |      |      |  12/4096  |   90.6    |
|  ELECTRA (s)  |      14M      | 128  | 5e-4 |  12/256   |   81.6    |
|  ELECTRA (b)  |      110M     | 256  | 2e-4 |  12/768   |   88.5    |
|  ELECTRA (l)  |      335M     | 2048 | 2e-4 |  24/1024  |   90.7    |
|   XLM-R (b)   |      270M     |      |      |  12/768   |           |
|   XLM-R (l)   |      550M     |      |      |  24/1024  |   89.0    |





## Experiment 1 (electra in en+zh :D)



|     Model           |   params #    |    L/H    |  MNLI-en<br/>(Acc)  |  DRCD-dev<br/>(EM/F1)  |
| ------------------- | -------------:|:---------:|:---------:|:-----------:|
|  bert (b)           |     110M      |  12/256   |           |  85.0/91.2  |
|  alBERT (b)         |      12M      |  12/768   |   [84.6](https://github.com/google-research/albert)    |             |
|  ELECTRA (s)        |      14M      |  12/256   |   [81.6](https://openreview.net/pdf?id=r1xMH1BtvB)    |  [84.0/89.5](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd)  |
|  Ours               |      12M      |  12/768   |   80.5    |  81.8/88.0  |

> So close... Performance in bilingual model does drop. For such a small model, this drop is acceptable (I think :D).

> Models will be uploaded later!


#### Small model scripts


```bash
DATA_DIR=$BUCKET/corpus_tf_256_3
python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_256_small \
  --hparams '{"model_size":"small","num_train_steps":4000000,"use_tpu":true,"num_tpu_cores":8,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-1","max_seq_length":256,"learning_rate":0.0005,"train_batch_size":128,"save_checkpoints_steps":50000,"iterations_per_loop":1000}'

DATA_DIR=$BUCKET/corpus_tf_256_3
python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_256_base \
  --hparams '{"model_size":"base","num_train_steps":4000000,"use_tpu":true,"num_tpu_cores":8,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-1","max_seq_length":256,"learning_rate":0.0002,"train_batch_size":256,"save_checkpoints_steps":50000,"iterations_per_loop":1000}'

##############################
##
##  Finetuning scripts
##
##############################

DATA_DIR=$BUCKET/corpus_tf_256_3
FINETUNE_DATA_DIR=$BUCKET/finetuning_data
GCS_STAT_CACHE_MAX_AGE=0 GCS_READ_CACHE_DISABLED=1 \
python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_256_small \
  --hparams '{"model_size": "small","task_names": ["mnli"],"vocab_size":32056,"max_seq_length":256,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-a1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":3,"weight_decay_rate":0,"layerwise_lr_decay":0.8,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'

1.15M
mnli: accuracy: 77.98 - loss: 0.78
1.70M
mnli: accuracy: 78.94 - loss: 0.81
3.00M
mnli: accuracy: 79.34 - loss: 0.84
4.0M
mnli: accuracy: 80.46 - loss: 0.83
drcd: exact_match: 81.81 - f1: 87.97


DATA_DIR=$BUCKET/corpus_tf_256_3
FINETUNE_DATA_DIR=$BUCKET/finetuning_data
GCS_STAT_CACHE_MAX_AGE=0 GCS_READ_CACHE_DISABLED=1 \
python3 run_finetuning.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_256_small \
  --hparams '{"model_size": "small","task_names": ["mnli","drcd"],"vocab_size":32056,"max_seq_length":256,"vocab_file":"cantokenizer-vocab.txt","use_tpu":true,"num_tpu_cores":8,"tpu_name":"node-a1","train_batch_size":32,"learning_rate":3e-4,"num_train_epochs":3,"weight_decay_rate":0,"layerwise_lr_decay":0.8,"raw_data_dir":"'$FINETUNE_DATA_DIR'"}'

4.0M
mnli: accuracy: 80.04 - loss: 0.78
drcd: exact_match: 80.45 - f1: 86.87

```



#### Base model scripts

```bash
DATA_DIR=$BUCKET/corpus_tf_256_3
python3 run_pretraining.py \
  --data-dir $DATA_DIR \
  --model-name electra_yue_256_base \
  --hparams '{"model_size":"base","num_train_steps":4000000,"use_tpu":true,"num_tpu_cores":8,"vocab_size":32056,"vocab_file":"cantokenizer-vocab.txt","tpu_name":"node-1","max_seq_length":256,"learning_rate":0.0002,"train_batch_size":256,"save_checkpoints_steps":50000,"iterations_per_loop":1000}'

##############################
##
##  Finetuning scripts
##
##############################

```

## Benchmarks

### **ELECTRA-base / EN-MNLI**
|   ( Acc. )    |      Dev  |
| ------------- |:------------:|
| Ours          |     ?    |
| [Google](https://github.com/google-research/electra)  | 88.8 |


### **ELECTRA-base / DRCD**
|   ( EM / F1 )    |      Dev      |  Test  |
| ------------- |:-------------:|:------:|
| Ours          |        ?      |     ?  |
| [HFL](https://github.com/ymcui/Chinese-ELECTRA)  | 87.5 / 92.5        |   86.9 / 91.8 |


## References

### Expected Losses / Training Curves during Pre-Training.

https://github.com/google-research/electra/issues/3