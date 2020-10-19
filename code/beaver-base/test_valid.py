# -*- coding: utf-8 -*-
import logging

import torch
import torch.cuda

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import SaverTest
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = SaverTest(opt)


def write_file(hypothesis, references, step):
    with open("hypothesis_"+str(step), "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(hypothesis))
        out_file.write("\n")

    with open("references_"+str(step), "w", encoding="UTF-8") as out_file:
        out_file.write("\n".join(references))
        out_file.write("\n")

def valid(model, criterion, valid_dataset, step):
    model.eval()
    total_loss, total = 0.0, 0

    hypothesis, references = [], []

    for batch in valid_dataset:
        scores = model(batch.src, batch.tgt)
        loss = criterion(scores, batch.tgt)
        total_loss += loss.data
        total += 1

        if opt.tf:
            _, predictions = scores.topk(k=1, dim=-1)
        else:
            predictions = beam_search(opt, model, batch.src, valid_dataset.fields)

        hypothesis += [valid_dataset.fields["tgt"].decode(p) for p in predictions]
        references += [valid_dataset.fields["tgt"].decode(t) for t in batch.tgt]
        del loss
        
    bleu = calculate_bleu(hypothesis, references)
    #write_file(hypothesis, references, step)
    logging.info("Valid loss: %.2f\tValid BLEU: %3.2f" % (total_loss / total, bleu))
    #checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(step, bleu, total_loss / total)


def main():
    logging.info("Build dataset...")
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}
    for ckpt in range(755000,1000001, 5000):
        model_path = opt.model_path + '/checkpoint-step-' + str(ckpt)
        logging.info("Load checkpoint from %s." % model_path)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        model = NMTModel.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()

        criterion = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["tgt"], pad_ids["tgt"]).to(device)

        n_step = ckpt
        logging.info("Start translation...")
        with torch.set_grad_enabled(False):
            valid(model, criterion, valid_dataset, n_step)


if __name__ == '__main__':
    main()
