# -*- coding: utf-8 -*-

from beaver.data.dataset import TranslationDataset
from beaver.data.dataset2 import SumTransDataset
from beaver.data.dataset_wrapper import Dataset
from beaver.data.field import Field


def build_dataset(opt, data_path, vocab_path, device, train=True):
    # task 1 is MT, task2 is MS and task 3 is CLS
    task1_source_path = data_path[0]
    task1_target_path = data_path[1]
    task2_source_path = task3_source_path = data_path[2]
    task2_target_path = data_path[3]
    task3_target_path = data_path[4]

    source_field = Field(unk=True, pad=True, bos=False, eos=False)
    translation_target_field = Field(unk=True, pad=True, bos=True, eos=True)
    summary_target_field = Field(unk=True, pad=True, bos=True, eos=True)
    summary_mono_field = Field(unk=True, pad=True, bos=True, eos=True)

    source_vocab, target_vocab = vocab_path
    source_special = source_field.special
    translation_target_special = translation_target_field.special
    summary_target_special = summary_target_field.special
    summary_mono_special = summary_mono_field.special

    with open(source_vocab, encoding="UTF-8") as f:
        source_words = [line.strip() for line in f]
    with open(target_vocab, encoding="UTF-8") as f:
        target_words = [line.strip() for line in f]

    if opt.mono:
        # source和摘要target共享词表
        source_special = summary_target_special = sorted(set(source_special + summary_target_special))

    if opt.share_cn_embedding:
        summary_mono_special = source_special = sorted(set(source_special + summary_mono_special))

    source_field.load_vocab(source_words, source_special)
    translation_target_field.load_vocab(target_words, translation_target_special)

    if opt.mono:
        summary_target_field.load_vocab(source_words, summary_target_special)
    else:
        summary_target_field.load_vocab(target_words, summary_target_special)

    summary_mono_field.load_vocab(source_words, summary_mono_special)


    data1 = TranslationDataset(task1_source_path, task1_target_path, opt.batch_size, device, train,
                               {'src': source_field, 'tgt': translation_target_field})

    data2 = SumTransDataset(task2_source_path, task2_target_path, task3_target_path, opt.batch_size, device, train,
                            {'src': source_field, 'summary_cn': summary_mono_field, 'summary_en': summary_target_field})

    return Dataset(data1, data2)
