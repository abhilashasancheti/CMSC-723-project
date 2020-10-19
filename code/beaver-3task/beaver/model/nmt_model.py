# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn

from beaver.model.embeddings import Embedding
from beaver.model.transformer import Decoder, Encoder


class Generator(nn.Module):
    def __init__(self, hidden_size: int, tgt_vocab_size: int):
        self.vocab_size = tgt_vocab_size
        super(Generator, self).__init__()
        self.linear_hidden = nn.Linear(hidden_size, tgt_vocab_size)
        self.lsm = nn.LogSoftmax(dim=-1)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_hidden.weight)

    def forward(self, dec_out):
        score = self.linear_hidden(dec_out)
        lsm_score = self.lsm(score)
        return lsm_score


class NMTModel(nn.Module):
    # task 1 MT, task 2 MS and task 3 CLS
    def __init__(self, encoder: Encoder,
                 task1_decoder: Decoder,
                 task2_decoder: Decoder,
                 task3_decoder: Decoder,
                 task1_generator: Generator,
                 task2_generator: Generator,
                 task3_generator: Generator):
        super(NMTModel, self).__init__()

        self.encoder = encoder
        self.task1_decoder = task1_decoder
        self.task2_decoder = task2_decoder
        self.task3_decoder = task3_decoder
        self.task1_generator = task1_generator
        self.task2_generator = task2_generator
        self.task3_generator = task3_generator

    def forward(self, source, target, summary_cn, summary_en, flag):
        source_pad = source.eq(self.encoder.embedding.word_padding_idx)
        
        if flag:
          # MT is trained when flag =1
          target = target[:, :-1]  # shift left
          target_pad = target.eq(self.task1_decoder.embedding.word_padding_idx)
        else:
          summary_cn = summary_cn[:, :-1]  # shift left
          summary_en = summary_en[:, :-1]  # shift left
          summary_cn_pad = summary_cn.eq(self.task2_decoder.embedding.word_padding_idx)
          summary_en_pad = summary_en.eq(self.task3_decoder.embedding.word_padding_idx)

        enc_out = self.encoder(source, source_pad)

        if flag:  # task1
          decoder_outputs, _ = self.task1_decoder(target, enc_out, source_pad, target_pad)
          return self.task1_generator(decoder_outputs)
        else:  # task2+3
          task2_decoder_outputs, _ = self.task2_decoder(summary_cn, enc_out, source_pad, summary_cn_pad)
          task3_decoder_outputs, _ = self.task3_decoder(summary_en, enc_out, source_pad, summary_en_pad)
          task2_scores = self.task2_generator(task2_decoder_outputs)
          task3_scores = self.task3_generator(task3_decoder_outputs)
          return task2_scores, task3_scores

      

    @classmethod
    def load_model(cls, model_opt,
                   pad_ids: Dict[str, int],
                   vocab_sizes: Dict[str, int],
                   checkpoint=None):
        source_embedding = Embedding(embedding_dim=model_opt.hidden_size,
                                     dropout=model_opt.dropout,
                                     padding_idx=pad_ids["src"],
                                     vocab_size=vocab_sizes["src"])
        
        target_embedding_task3 = Embedding(embedding_dim=model_opt.hidden_size,
                                           dropout=model_opt.dropout,
                                           padding_idx=pad_ids["task3_tgt"],
                                           vocab_size=vocab_sizes["task3_tgt"])

        if model_opt.mono:
            # 单语摘要，task1 share source embedding
          target_embedding_task1 = source_embedding
        else:
          target_embedding_task1 = target_embedding_task3

        if model_opt.share_cn_embedding:
          target_embedding_task2 = source_embedding
        else:
          target_embedding_task2 = Embedding(embedding_dim=model_opt.hidden_size,
                                           dropout=model_opt.dropout,
                                           padding_idx=pad_ids["task2_tgt"],
                                           vocab_size=vocab_sizes["task2_tgt"])


        encoder = Encoder(model_opt.layers,
                          model_opt.heads,
                          model_opt.hidden_size,
                          model_opt.dropout,
                          model_opt.ff_size,
                          source_embedding)

        task1_decoder = Decoder(model_opt.layers,
                                model_opt.heads,
                                model_opt.hidden_size,
                                model_opt.dropout,
                                model_opt.ff_size,
                                target_embedding_task1)

        task2_decoder = Decoder(model_opt.layers,
                                model_opt.heads,
                                model_opt.hidden_size,
                                model_opt.dropout,
                                model_opt.ff_size,
                                target_embedding_task2)
 
        task3_decoder = Decoder(model_opt.layers,
                                model_opt.heads,
                                model_opt.hidden_size,
                                model_opt.dropout,
                                model_opt.ff_size,
                                target_embedding_task3)        

        task1_generator = Generator(model_opt.hidden_size, vocab_sizes["task1_tgt"])
        task2_generator = Generator(model_opt.hidden_size, vocab_sizes["task2_tgt"])
        task3_generator = Generator(model_opt.hidden_size, vocab_sizes["task3_tgt"])

        model = cls(encoder, task1_decoder, task2_decoder, task3_decoder, task1_generator, task2_generator, task3_generator)
        if checkpoint is None and model_opt.train_from:
            checkpoint = torch.load(model_opt.train_from, map_location=lambda storage, loc: storage)
            model.load_state_dict(checkpoint["model"])
        elif checkpoint is not None:
            model.load_state_dict(checkpoint)
        return model
