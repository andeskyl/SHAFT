# coding=utf-8
# Modified by SHAFT's team: Private Text Classification on ImageNet-1k.
# 
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Finetuning any 🤗 Transformers model for image classification leveraging 🤗 Accelerate."""

import argparse
import logging
import torch

import datasets
import crypten as ct
from crypten.config import cfg
from multiprocess_launcher import MultiProcessLauncher

from tqdm.auto import tqdm

import transformers
from transformers import AutoConfig, AutoModelForImageClassification
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.42.0.dev0")

require_version("datasets>=2.0.0", "To fix: pip install -r examples/pytorch/image-classification/requirements.txt")


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a Transformers model on an image classification dataset")
    parser.add_argument(
        "--comp",
        action="store_true",
        help="If passed, estimate computation time (without communication).",
    )
    parser.add_argument("--validation_dir", type=str, default=None, help="A folder containing the validation data.")
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        default="google/vit-base-patch16-224-in21k",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible result.")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()

    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=1000,
        finetuning_task="image-classification",
    )
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    device = "cuda"
    ct.init()

    dummy_input = torch.rand([1, 3, 224, 224])
    private_model = ct.nn.from_pytorch(model, dummy_input).encrypt().to(device)

    model.eval()
    for _ in range(args.max_eval_samples):
        input = torch.rand([1, 3, 224, 224], device=device)
        input_enc = ct.cryptensor(input).to(device)
        with ct.no_grad():
            private_model(input_enc)


if __name__ == "__main__":
    args = parse_args()
    if args.comp:
        # run without communication
        with cfg.temp_override({"cost.estimate_cost": True, "cost.estimate_mode": "comp"}):
            main()
    else:
        # run with communication
        launcher = MultiProcessLauncher(2, main)
        launcher.start()
        launcher.join()
        launcher.terminate()
