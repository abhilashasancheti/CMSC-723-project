# -*- coding: utf-8 -*-

import random
from collections import namedtuple
from typing import Dict

import torch

from beaver.data.field import Field

Batch = namedtuple("Batch", ['src', 'summary_cn', 'summary_en', 'batch_size'])
Example = namedtuple("Example", ['src', 'summary_cn', 'summary_en'])


class SumTransDataset(object):

    def __init__(self,
                 source_path: str,
                 summary_cn_path: str,
                 summary_en_path: str,
                 batch_size: int,
                 device: torch.device,
                 train: bool,
                 fields: Dict[str, Field]):

        self.batch_size = batch_size
        self.train = train
        self.device = device
        self.fields = fields
        self.sort_key = lambda ex: (len(ex.src), len(ex.summary_cn), len(ex.summary_en))

        examples = []
        for src_line, cn_line, en_line in zip(read_file(source_path),
                                              read_file(summary_cn_path),
                                              read_file(summary_en_path)):
            examples.append(Example(src_line, cn_line, en_line))
        examples, self.seed = self.sort(examples)

        self.num_examples = len(examples)
        self.batches = list(batch(examples, self.batch_size))

    def __iter__(self):
        while True:
            if self.train:
                random.shuffle(self.batches)
            for minibatch in self.batches:
                source = self.fields["src"].process([x.src for x in minibatch], self.device)
                summary_cn = self.fields["summary_cn"].process([x.summary_cn for x in minibatch], self.device)
                summary_en = self.fields["summary_en"].process([x.summary_en for x in minibatch], self.device)
                yield Batch(src=source, summary_cn=summary_cn, summary_en=summary_en, batch_size=len(minibatch))
            if not self.train:
                break

    def sort(self, examples):
        seed = sorted(range(len(examples)), key=lambda idx: self.sort_key(examples[idx]))
        return sorted(examples, key=self.sort_key), seed


def read_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()


def batch(data, batch_size):
    minibatch, cur_source_len, cur_target_len = [], 0, 0
    for ex in data:
        minibatch.append(ex)
        cur_source_len = max(cur_source_len, len(ex.src))
        cur_target_len = max(cur_target_len, len(ex.summary_en), len(ex.summary_cn))
        if (cur_target_len + cur_source_len) * len(minibatch) > batch_size:
            yield minibatch[:-1]
            minibatch, cur_source_len, cur_target_len = [ex], len(ex.src), max(len(ex.summary_cn), len(ex.summary_en))
    if minibatch:
        yield minibatch