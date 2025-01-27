# SHAFT: Secure, Handy, Accurate, and Fast Transformer Inference
This repository implements secure, handy, accurate, and fast transformer inference based on [CrypTen](https://github.com/facebookresearch/CrypTen).

## Installing SHAFT
The following commands run successfully on Ubuntu 22.04 with Python 3.10.12.
### 0. Set up Virtual Environment (Recommended)
```bash
python3 -m venv ~/env/shaft
source ~/env/shaft/bin/activate
```
### 1. Install Dependencies
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install wheel==0.40.0
```
### 2. Install SHAFT
```bash
git clone https://github.com/andeskyl/SHAFT
cd SHAFT
pip install .
```

### 3. Install Transformers (for Hugging Face Integration)
```bash
git clone -b 'v4.45.0' --depth 1 https://github.com/huggingface/transformers
pip install ./transformers
```

## Running Experiments
We have a set of sub-directories in the `examples` directory for reproducible experimental results. Additional dependencies for the experiments are included in the `requirements.txt` file in each subdirectory under the folder. Please refer to the `README.md` file in the sub-directories for instructions on how to set up and run the experiments.

1. `unit-test` - Costs of private softmax and GELU protocols.
2. `text-classification` - Private inference costs of BERT-base and BERT-large.
3. `text-generation` - Private inference cost of GPT-2.
4. `image-classification` - Private inference cost of ViT-base.

## Citation
You can cite our paper as follows:
```bibtex
@inproceedings{ndss/KeiC25,
    author = {Andes Y. L. Kei and Sherman S. M. Chow},
    title = {{SHAFT}: {Secure}, Handy, Accurate, and Fast Transformer Inference},
    booktitle = {{NDSS}},
    year = {2025}
}
```

## License
SHAFT is MIT licensed, as found in the LICENSE file.