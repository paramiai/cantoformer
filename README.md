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