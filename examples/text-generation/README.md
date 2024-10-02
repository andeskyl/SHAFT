# Private Inference Cost of GPT-2
This directory evaluates the cost of generating one token with GPT-2 in private.
## Preparation
Install dependencies:
```bash
pip install -r requirements.txt
```
## Running Experiments
Computation cost of private GPT-2 inference for a length-64 input:
```bash
bash test_gpt2_64_comp.sh
```
Communication cost of private GPT-2 inference for a length-64 input:
```bash
bash test_gpt2_64_comm.sh
```
Computation cost of private GPT-2 inference for a length-128 input:
```bash
bash test_gpt2_128_comp.sh
```
Communication cost of private GPT-2 inference for a length-128 input:
```bash
bash test_gpt2_128_comm.sh
```
