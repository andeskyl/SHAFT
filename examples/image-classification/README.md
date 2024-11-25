# Private Inference Cost of ViT-base
This directory evaluates the private inference cost of ViT-base.
## Preparation
Install dependencies:
```bash
pip install -r requirements.txt
```
## Running Experiments
Computation cost of private ViT-base inference for a 224×224 RGB image:
```bash
bash test_vit_base_224_comp.sh
```
Communication cost of private ViT-base inference for a 224×224 RGB image:
```bash
bash test_vit_base_224_comm.sh
```