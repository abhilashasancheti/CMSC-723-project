# -*- coding: utf-8 -*-
import logging

import torch
import os

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.model import NMTModel
from beaver.utils import parseopt, get_device, calculate_bleu

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_translate_args()
device = get_device()


def translate(dataset, fields, model):

    already_1, hypothesis_1, references_1 = 0, [], []
    already_2_3, hypothesis_2, hypothesis_3 = 0, [], []

    for batch, flag in dataset:
        if flag:
            predictions = beam_search(opt, model, batch.src, fields, flag)
            hypothesis_1 += [fields["task1_tgt"].decode(p) for p in predictions]
            already_1 += len(predictions)
            logging.info("Task 1: %7d/%7d" % (already_1, dataset.task1_dataset.num_examples))
        else:
            predictions_1, predictions_2 = beam_search(opt, model, batch.src, fields, flag)
            hypothesis_2 += [fields["task2_tgt"].decode(p) for p in predictions_1]
            hypothesis_3 += [fields["task3_tgt"].decode(p) for p in predictions_2]
            already_2_3 += len(predictions_1)
            logging.info("Finished: %7d/%7d" % (already_2_3, dataset.task2_3_dataset.num_examples))

    origin_1 = sorted(zip(hypothesis_1, dataset.task1_dataset.seed), key=lambda t: t[1])
    hypothesis_1 = [h for h, _ in origin_1]

    with open(opt.output[0], "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis_1))
        out_file.write("\n")

    origin_2_3 = sorted(zip(hypothesis_2, hypothesis_3, dataset.task2_3_dataset.seed), key=lambda t: t[2])
    hypothesis_2 = [h for h, _, _ in origin_2_3]
    hypothesis_3 = [h for _, h, _ in origin_2_3]
    with open(opt.output[1], "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis_2))
        out_file.write("\n")
    with open(opt.output[2], "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis_3))
        out_file.write("\n")

    logging.info("All finished. ")



def main():
    logging.info("Build dataset...")
    dataset = build_dataset(opt, [opt.input[0], opt.input[0], opt.input[1], opt.input[1], opt.input[1]], opt.vocab, device, train=False)

    fields = dataset.fields

    pad_ids = {"src": fields["src"].pad_id,
               "task1_tgt": fields["task1_tgt"].pad_id,
               "task2_tgt": fields["task2_tgt"].pad_id,
               "task3_tgt": fields["task3_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task2_tgt"].vocab),
                   "task3_tgt": len(fields["task3_tgt"].vocab)}

    # load checkpoint from model_path
    logging.info("Load checkpoint from %s." % opt.model_path)
    checkpoint = torch.load(opt.model_path, map_location=lambda storage, loc: storage)

    logging.info("Build model...")
    model = NMTModel.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()

    logging.info("Start translation...")
    with torch.set_grad_enabled(False):
        translate(dataset, fields, model)


if __name__ == '__main__':
    main()

