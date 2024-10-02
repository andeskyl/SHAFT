# Private Inference Cost of BERT-base and BERT-large
This directory evaluates the private inference costs of BERT-base and BERT-large.
## Preparation
Install dependencies:
```bash
pip install -r requirements.txt
```
## Running Experiments
Computation cost of private BERT-base inference for a length-128 input:
```bash
bash test_bert_base_128_comp.sh
```
Communication cost of private BERT-base inference for a length-128 input:
```bash
bash test_bert_base_128_comm.sh
```
Computation cost of private BERT-large inference for a length-128 input:
```bash
bash test_bert_large_128_comp.sh
```
Communication cost of private BERT-large inference for a length-128 input:
```bash
bash test_bert_large_128_comm.sh
```
Private inference accuracy of BERT-base:
```bash
bash test_bert_base_acc.sh
```
(Optional) Plaintext inference accuracy of BERT-base: 
```bash
bash test_bert_base_plain.sh
```
