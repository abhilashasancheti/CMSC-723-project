# -*- coding: utf-8 -*-
import itertools

from beaver.data.dataset import TranslationDataset
from beaver.data.dataset2 import SumTransDataset

class Dataset(object):
    def __init__(self, task1_dataset: TranslationDataset, task2_3_dataset: SumTransDataset ):
        self.task1_dataset = task1_dataset
        self.task2_3_dataset = task2_3_dataset


        self.fields = {
            "src": task1_dataset.fields["src"],
            "task1_tgt": task1_dataset.fields["tgt"],
            "task2_tgt": task2_3_dataset.fields["summary_cn"],
            "task3_tgt": task2_3_dataset.fields["summary_en"]            
        }

    def __iter__(self):
        for batch1, batch2 in itertools.zip_longest(self.task1_dataset, self.task2_3_dataset):
            if batch1 is not None:
                yield batch1, True
            if batch2 is not None:
                yield batch2, False



