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


## To Do List

- [x] Apply normalization on Chinese characters
- [ ] ELECTRA-base + Whole Word Masking Preprocessing **(Working)**
- [ ] ELECTRA-base + Whole Word Masking Pretraining
- [ ] Finetuning on EN-MNLI
- [ ] Compare results to existing benchmarks
- [ ] Release `canto-mnli` dataset for evaluation
      


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

### **ELECTRA-small / EN-MNLI**
|   ( Acc. )    |      Dev  |
| ------------- |:------------:|
| Ours (1M steps)          |     75.9    |
| Ours (4M steps)          |     77.1    |
| Google (1M steps)          |     79.7    |
| Google (4M steps)          |     81.6    |

## References

### Expected Losses / Training Curves during Pre-Training.

https://github.com/google-research/electra/issues/3
