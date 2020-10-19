# -*- coding: utf-8 -*-
import logging

import torch
import torch.cuda

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils.metric import calculate_rouge
from beaver.utils import parseopt, get_device, printing_opt

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


def write_file(hypothesis, references, step):
    with open("hypothesis_"+str(step), "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    with open("references_"+str(step), "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(references))
        out_file.write("\n")

def valid(model, criterion_task1, criterion_task2, valid_dataset, step):
    model.eval()
    total_n = 0
    total_task1_loss = total_task2_loss = 0.0
    task1_hypothesis, task1_references = [], []
    task2_hypothesis, task2_references = [], []

    for i, (batch, flag) in enumerate(valid_dataset):
        scores = model(batch.src, batch.tgt, flag)
        if flag:
            loss = criterion_task1(scores, batch.tgt)
        else:
            loss = criterion_task2(scores, batch.tgt)
        _, predictions = scores.topk(k=1, dim=-1)

        if flag:  # task1
            total_task1_loss += loss.data
            task1_hypothesis += [valid_dataset.fields["task1_tgt"].decode(p) for p in predictions]
            task1_references += [valid_dataset.fields["task1_tgt"].decode(t) for t in batch.tgt]
        else:
            total_task2_loss += loss.data
            task2_hypothesis += [valid_dataset.fields["task2_tgt"].decode(p) for p in predictions]
            task2_references += [valid_dataset.fields["task2_tgt"].decode(t) for t in batch.tgt]

        total_n += 1
        del loss
    print(total_n)
    bleu_task1 = calculate_bleu(task1_hypothesis, task1_references)
    bleu_task2 = calculate_bleu(task2_hypothesis, task2_references)
    rouge1_task1, rouge2_task1 = calculate_rouge(task1_hypothesis, task1_references)
    rouge1_task2, rouge2_task2 = calculate_rouge(task2_hypothesis, task2_references)
    mean_task1_loss = total_task1_loss / len(task1_references)
    mean_task2_loss = total_task2_loss / len(task2_references)
    logging.info("loss-task1: %.2f \t loss-task2 %.2f \t bleu-task1: %3.2f\t bleu-task2: %3.2f \t rouge1-task1: %3.2f \t rouge1-task2: %3.2f \t rouge2-task1: %3.2f \t rouge2-task2: %3.2f"
                 % (mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2)

def main():
    logging.info("Build dataset...")
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id,
               "task1_tgt": fields["task1_tgt"].pad_id,
               "task2_tgt": fields["task2_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task2_tgt"].vocab)}
    print(vocab_sizes)
    for ckpt in range(805000,1000001, 5000):
        model_path = opt.model_path + '/checkpoint-step-' + str(ckpt)
        logging.info("Load checkpoint from %s." % model_path)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        model = NMTModel.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()
        criterion_task1 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task1_tgt"], pad_ids["task1_tgt"]).to(device)
        criterion_task2 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task2_tgt"], pad_ids["task2_tgt"]).to(device)

        n_step = ckpt
        logging.info("Start translation...")
        with torch.set_grad_enabled(False):
            valid(model, criterion_task1, criterion_task2, valid_dataset, n_step)
            


if __name__ == '__main__':
    main()
