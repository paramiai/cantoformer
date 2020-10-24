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

AI 喺呢幾年發展得好快，好多嘢都話用 AI 處理會醒好多，但其實喺「語言處理」嘅領域入面，好多嘅資源都只係得英文，所以要落手做廣東話嘅 NLP，其實唔容易。

所以諗住喺呢度開個 Repo ，鼓勵更多人開發廣東話 AI。


## **Challenges**

- Mixed Languages (English, Chinese, Yue) <br/>
  夾雜多種語言
- Complex Syntax <br/>
  語法複雜
- Scarce Resource <br/>
  資源稀少
- Many Homonyms & Homophones in online texts <br/>
  網上嘅字通常有好多一語多義／同音異字


## **Remediation**

We adopt the following preprocessing to the model:<br/>
用呢個 model 前我哋會對文字做一啲嘅處理：

- WordPiece Tokenizer from [**forked 🤗Tokenizers**](https://github.com/ecchochan/tokenizers/tree/zh-norm-4) which,

  - strips accents like the original BERT<br/>
    除去[組合附加符號](https://zh.wikipedia.org/zh-hk/%E7%B5%84%E5%90%88%E9%99%84%E5%8A%A0%E7%AC%A6%E8%99%9F) (e.g. `à` → `a`)

  - uses lower casing<br/>
    使用細階英文

  - treats symbols/numers as a separate token<br/>
    符號／數字全部當係一個 token

  - Simplified Chinese → Traditional Chinese (Since most of our corpus are in Trad. Chinese)<br/>
    簡轉繁（因為文本大部分都係繁體字）

    **Using OpenCC v1.1.1 from [here](https://github.com/BYVoid/OpenCC.git)**

  - normalizes Unicode Characters (Some are hand-crafted) by<br/>
    統一中文字符（其中一啲係人手分類）
    - Symbols of the same functionality 相同功能嘅符號 (e.g. `【` → `[` )
    - Variant Chinese characters 異體字 (e.g. `俢` → `修` )
    - Deomposing rare characters 將罕見字拆開 (e.g. `偆` → `亻春` )

    **(Mapping [here](./zh_char2str_mapping.txt))**

- Newlines are regarded as a token, i.e. `<nl>`



## **Framework to be used**

- `Tensorflow`
- `Pytorch`

## **Libraries to be used**

- `OpenCC` (Simpl-to-Trad, 簡轉繁) @ v1.1.1
- `🤗Tokenizers` ([forked version](https://github.com/ecchochan/tokenizers/tree/zh-norm-4) is used for normalization)

```bash
# Installing OpenCC v1.1.1 by
sudo bash ./install_opencc.sh

# Installing by forked 🤗 Tokenizers by 
pip3 install 'git+https://github.com/ecchochan/tokenizers.git@zh-norm-4#egg=version_subpkg&subdirectory=bindings/python'
# This takes some time!

# This is forked from tokenizers@v0.8.1
# with python package renamed to tokenizers_zh

```

## **Corpus**

|    zh    |    en    |
|:--------:|:--------:|
| ~ 80 GB<br/>(incl. ~ 20 GB Cantonese)  | ~ 100 GB |


## **Evaluation**

Since we have NO datasets in Cantonese, we evaluate the models on **both English and Chinese** datasets:

- [**MNLI**](https://cims.nyu.edu/~sbowman/multinli/) (Entailment Prediction)
- [**DRCD**](https://github.com/DRCSolutionService/DRCD) (Reading Comprehension)
- [**SQuAD-v2**](https://rajpurkar.github.io/SQuAD-explorer) (Reading Comprehension)
- [**CMRC2018**](https://github.com/ymcui/cmrc2018) (Reading Comprehension)

## **Something to explore**

1. **Sentence Order Prediction (SOP)**

   SOP is a pretraining objective that is used in Albert. StructBERT also introduces Sentence Structural Objective, but since the code for electra reads the data sequentially, this repo explores SOP first

2. **Cluster Objective**
   
   [DocProduct](https://github.com/re-search/DocProduct) is a cool project training a BERT model to cluster similar Q&A -- if a text A answers the question Q, then Q and A will be close in vector representation. 

   This means the model must predict the possible contexts (before and after) in order to embed a vector that can minimize the cost function

   Details refer to the [DocProduct](https://github.com/re-search/DocProduct) repo.

## **To Do List**

- [x] Normalize Chinese characters
- [x] ELECTRA-small
- [x] ELECTRA-base
- [x] ELECTRA-base-sop
- [x] ELECTRA-albert-base
- [x] ELECTRA-albert-xlarge
- [ ] ELECTRA-base-cluster
- [ ] ELECTRA-large
- [ ] Evaluation in Cantonese dataset
- [ ] Upload to 🤗Huggingface
      

## **Model Comparisons**

|   |     Model     |   params #    |    L/H    |  MNLI-en  |  DRCD-dev<br/>(EM/F1)  | SQuADv2-dev<br/>(EM/F1)  | CMRC2018-dev<br/>(EM/F1)  |
|:-:| ------------- | -------------:|:---------:|:---------:|:-----------:|:-----------:|:-----------:|
|🐤|    BERT (s)   |      12M      |  12/256   |   [77.6](https://github.com/google-research/bert)    |       |   [60.5/64.2](https://huggingface.co/mrm8488/bert-small-finetuned-squadv2)🤗
|🐦|    BERT (b)   |      110M     |  12/768   |   [84.3](https://github.com/google-research/bert)    |  [85.0/91.2](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd)  | [72.4/75.8](https://huggingface.co/twmkn9/bert-base-uncased-squad2)🤗
|🦅|    BERT (l)   |      334M     |  12/1024  |   [87.1](https://github.com/google-research/bert)    |  |  [92.8/86.7](https://github.com/google-research/bert)
|   |
|🐦|  roBERTa (b)  |      110M     |  12/768   |   [87.6](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#results)    | [86.6/92.5](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) | [78.5/81.7](https://huggingface.co/deepset/roberta-base-squad2)🤗
|🦅|  roBERTa (l)  |      335M     |  24/1024  |   [90.2](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#results)    |  |[88.9/94.6](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#results)
|   |
|🐤|  alBERT (b)   |      12M      |  12/768   |   [84.6](https://github.com/google-research/albert)    |    |   [79.3/82.1](https://github.com/google-research/albert)
|🐤|  alBERT (l)   |      18M      |  24/1024  |   [86.5](https://github.com/google-research/albert)    |    |   [81.8/84.9](https://github.com/google-research/albert)
|🐦|  alBERT (xl)  |      60M      |  24/2048  |   [87.9](https://github.com/google-research/albert)    |    |   [84.1/87.9](https://github.com/google-research/albert)
|🦅|  alBERT (xxl) |      235M     |  12/4096  |   [90.6](https://github.com/google-research/albert)    |    |   [86.9/89.8](https://github.com/google-research/albert)
|   |
|🐤|  ELECTRA (s)  |      14M       |  12/256   |   [81.6](https://openreview.net/pdf?id=r1xMH1BtvB)    |  [83.5/89.2](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) |  [69.7/73.4](https://huggingface.co/mrm8488/electra-small-finetuned-squadv2)🤗
|🐦|  ELECTRA (b)  |      110M      |  12/768   |   [88.5](https://openreview.net/pdf?id=r1xMH1BtvB)    |  [89.6/94.2](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) | [80.5/83.3](https://openreview.net/pdf?id=r1xMH1BtvB) | [69.3/87.0](https://github.com/ymcui/Chinese-ELECTRA#%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3cmrc-2018)
|🦅|  ELECTRA (l)  |      335M      |  24/1024  |   [90.7](https://openreview.net/pdf?id=r1xMH1BtvB)    | [88.8/93.3](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) | [88.0/90.6](https://openreview.net/pdf?id=r1xMH1BtvB)
|  |
|🐦|   XLM-R (b)   |      270M     |  12/768   |           |
|🦅|   XLM-R (l)   |      550M     |  24/1024  |   [89.0](https://arxiv.org/abs/1911.02116)    |
|   |
|   |   **Ours (1.2M)**    |
|🐤|  [ELECTRA (s)](https://drive.google.com/drive/folders/1LdDE6s7bKl_0qVxk5zuOBh8Z0Vg_J2RP)  |      14M       |  12/256   | **80.7**  |  **82.1/88.0**  | **69.4/72.1**
|🐦|  [ELECTRA (b)](https://drive.google.com/drive/folders/1YH4ORT6dnSsZGSdd3WFB_VDF4iBtycTb)  |      110M      |  12/768   | **86.3**  |  **88.2/92.5** |  **80.4/83.3**
|🐦|  [albert (xl)](https://drive.google.com/drive/folders/1ASk9uk25XVyiaHmb_epqGiWjzE1JtMFd)  |      60M      |  12/2048   | **87.7**  |  **89.9/94.7** |  **82.9/85.9**
|   |
|   |   **Ours (1.5M)**    |
|🐦|  [ELECTRA (b)](https://drive.google.com/drive/folders/1AnjhDoVxk8Wu6qmeV92Vh5pAbmo-NtoJ)  |      110M      |  12/768   | **86.8**  |  **88.5/93.3** |  **80.8/83.7** |  **67.4/86.7**
|   |  + *finetuned after SQuAD*  |            |     |          | **89.5/94.1** |     |  **70.2/88.5**
|||

---

## **Individual Comparisions**

### **Small Models 🐤**

|   |     Model     |   params #    |    L/H    |  MNLI-en  |  DRCD-dev<br/>(EM/F1)  | SQuADv2-dev<br/>(EM/F1)  |
|:-:| ------------- | -------------:|:---------:|:---------:|:-----------:|:-----------:|
|🐤|    BERT (s)   |      12M      |  12/256   |   [77.6](https://github.com/google-research/bert)    |       |   [60.5/64.2](https://huggingface.co/mrm8488/bert-small-finetuned-squadv2)🤗
|   |
|🐤|  alBERT (b)   |      12M      |  12/768   |   [84.6](https://github.com/google-research/albert)    |    |   [79.3/82.1](https://github.com/google-research/albert)
|🐤|  alBERT (l)   |      18M      |  24/1024  |   [86.5](https://github.com/google-research/albert)    |    |   [81.8/84.9](https://github.com/google-research/albert)
|   |
|🐤|  ELECTRA (s)  |      14M       |  12/256   |   [81.6](https://openreview.net/pdf?id=r1xMH1BtvB)    |  [79.8/86.7](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) |  [69.7/73.4](https://huggingface.co/mrm8488/electra-small-finetuned-squadv2)🤗
|   |
|   |   **Ours**    |
|🐤|  ELECTRA (s)  |      14M       |  12/256   | **80.7**  |  **82.1/88.0**  | **69.4/72.1**


---

### **Base Models 🐦**

|   |     Model     |   params #    |    L/H    |  MNLI-en  |  DRCD-dev<br/>(EM/F1)  | SQuADv2-dev<br/>(EM/F1)  | CMRC2018-dev<br/>(EM/F1)  |
|:-:| ------------- | -------------:|:---------:|:---------:|:-----------:|:-----------:|:-----------:|
|🐦|    BERT (b)   |      110M     |  12/768   |   [84.3](https://github.com/google-research/bert)    |  [85.0/91.2](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd)  | [72.4/75.8](https://huggingface.co/twmkn9/bert-base-uncased-squad2)🤗
|   |
|🐦|  roBERTa (b)  |      110M     |  12/768   |   [87.6](https://github.com/pytorch/fairseq/blob/master/examples/roberta/README.md#results)    | [86.6/92.5](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) | [78.5/81.7](https://huggingface.co/deepset/roberta-base-squad2)🤗 | [67.4/87.2](https://github.com/ymcui/Chinese-BERT-wwm#%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3cmrc-2018)
|   |
|🐦|  ELECTRA (b)  |      110M      |  12/768   |   [88.5](https://openreview.net/pdf?id=r1xMH1BtvB)    |  [89.6/94.2](https://github.com/ymcui/Chinese-ELECTRA#%E7%B9%81%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3drcd) | [80.5/83.3](https://openreview.net/pdf?id=r1xMH1BtvB) | [69.3/87.0](https://github.com/ymcui/Chinese-ELECTRA#%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3cmrc-2018)
|   |
|   |   **Ours**    |
|🐦|  ELECTRA (b)  |      110M      |  12/768   | **86.3**  |  **88.2/92.5** |  **80.4/83.3**
|   |   **Ours (1.5M)**    |
|🐦|  ELECTRA (b)  |      110M      |  12/768   | **86.8**  |  **88.5/93.3** |  **80.8/83.7**
|   |  + *finetuned after SQuAD*  |            |     |          | **89.5/94.1** |     


---

## **Downloads 🐤🐦**

Electra checkpoints are put [here in Google Drive](https://drive.google.com/drive/folders/1FGu_2C5nQ2HVk6wn33w7PUp6JmlVR1cC).

Electra-albert checkpoints are [here in Google Drive](https://drive.google.com/drive/folders/1_UgF8LlmO9GSdkk_R9d29_qebx1sDoHi)

---

## **Explorations**

|   |     Model     |   params #    |    L/H    |  MNLI-en  |  DRCD-dev<br/>(EM/F1)  | SQuADv2-dev<br/>(EM/F1)  |
|:-:| ------------- | -------------:|:---------:|:---------:|:-----------:|:-----------:|
|   |   **Ours (1.5M)**    |
|🐦|  ELECTRA (b)  |      110M      |  12/768   | **86.8**  |  **88.5/93.3** |  **80.8/83.7** | 
|   |  + *finetuned after SQuAD*  |            |     |          | **89.5/94.1** |     |  
|   |
|   |   **Ours (1.5M) + SOP**    |
|🐦|  ELECTRA (b)  |      110M      |  12/768   | **87.1**  |  **88.6/93.6** |  **80.4/83.2** |
|   |  + *finetuned after SQuAD*  |            |     |          | **89.7/94.1** |     |

---

## **References**

### Expected Losses / Training Curves during Pre-Training.

https://github.com/google-research/electra/issues/3

---

## **Credits**

Special thanks to **Google's TensorFlow Research Cloud (TFRC)** for providing TPU-v3 for all the training in this repo!
