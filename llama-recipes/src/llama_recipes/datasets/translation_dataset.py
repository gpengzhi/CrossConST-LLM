# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://crfm.stanford.edu/2023/03/13/alpaca.html

import copy
import json
import random

import torch
from torch.utils.data import Dataset

PROMPT_DICT = {
    "prompt_input_gpt-mt_src2tgt": (
        "Translate this from {input_lang} to {output_lang}:\n{input_lang}: {input}\n{output_lang}:"
    ),
    "prompt_input_gpt-mt_tgt2tgt": (
        "Translate this from {output_lang} to {output_lang}:\n{output_lang}: {output}\n{output_lang}:"
    ),
    "prompt_input_t-enc_src2tgt": (
        "{output_lang}: {input}\n"
    ),
    "prompt_input_t-enc_tgt2tgt": (
        "{output_lang}: {output}\n"
    ),
    "prompt_input_t-dec_src2tgt": (
        "{input}\n{output_lang}:"
    ),
    "prompt_input_t-dec_tgt2tgt": (
        "{output}\n{output_lang}:"
    ),
    "prompt_input_s-enc-t-enc_src2tgt": (
        "{input_lang} {output_lang}: {input}\n"
    ),
    "prompt_input_s-enc-t-enc_tgt2tgt": (
        "{output_lang} {output_lang}: {output}\n"
    ),
    "prompt_input_s-enc-t-dec_src2tgt": (
        "{input_lang}: {input}\n{output_lang}:"
    ),
    "prompt_input_s-enc-t-dec_tgt2tgt": (
        "{output_lang}: {output}\n{output_lang}:"
    )
}


class InstructionDataset(Dataset):

    def __init__(self, dataset_config, tokenizer, partition="train", max_words=512):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == "train":
            self.ann = self.ann
        else:
            self.ann = self.ann[:100]

        self.max_words = max_words
        self.tokenizer = tokenizer
        self.prompt_type = dataset_config.prompt_type

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss

        ann = self.ann[index]

        sample = {}

        if self.prompt_type == "random":
            prompt_type = random.choice(["gpt-mt", "t-enc", "t-dec", "s-enc-t-enc", "s-enc-t-dec"])
        else:
            prompt_type = self.prompt_type

        # src -> tgt
        prompt = PROMPT_DICT["prompt_input_{}_src2tgt".format(prompt_type)].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        sample["input_ids"] = example
        sample["labels"] = labels
        sample["attention_mask"] = example_mask
        sample["prompt_length"] = torch.tensor(len(prompt), dtype=torch.int64)

        # tgt -> tgt
        prompt = PROMPT_DICT["prompt_input_{}_tgt2tgt".format(prompt_type)].format_map(ann)
        example = prompt + ann["output"]
        prompt = torch.tensor(self.tokenizer.encode(prompt), dtype=torch.int64)
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(example, dtype=torch.int64)
        padding = self.max_words - example.shape[0]
        if padding > 0:
            example = torch.cat((example, torch.zeros(padding, dtype=torch.int64) - 1))
        elif padding < 0:
            example = example[: self.max_words]
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX
        example_mask = example_mask.float()
        label_mask = label_mask.float()
        sample["input_ids_"] = example
        sample["labels_"] = labels
        sample["attention_mask_"] = example_mask
        sample["prompt_length_"] = torch.tensor(len(prompt), dtype=torch.int64)

        return sample
