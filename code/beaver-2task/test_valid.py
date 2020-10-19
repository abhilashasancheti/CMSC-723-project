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
from beaver.utils.metric import calculate_rouge


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

def valid(model, criterion_cn, criterion_en, valid_dataset, step):
    model.eval()
    total_loss = total_cn_loss = total_en_loss = 0.0
    total_n = 0
    cn_hypothesis, cn_references = [], []
    en_hypothesis, en_references = [], []

    for batch in valid_dataset:
        cn_scores, en_scores = model(batch.source, batch.summary_cn, batch.summary_en)
        cn_loss = criterion_cn(cn_scores, batch.summary_cn)
        en_loss = criterion_en(en_scores, batch.summary_en)

        loss = cn_loss + en_loss
        total_loss += loss.data
        total_cn_loss += cn_loss.data
        total_en_loss += en_loss.data
        total_n += 1

        _, cn_predictions = cn_scores.topk(k=1, dim=-1)
        cn_hypothesis += [valid_dataset.fields["summary_cn"].decode(p) for p in cn_predictions]
        cn_references += [valid_dataset.fields["summary_cn"].decode(t) for t in batch.summary_cn]

        _, en_predictions = en_scores.topk(k=1, dim=-1)
        en_hypothesis += [valid_dataset.fields["summary_en"].decode(p) for p in en_predictions]
        en_references += [valid_dataset.fields["summary_en"].decode(t) for t in batch.summary_en]

    #write_file(en_hypothesis, en_references, step)
    bleu_cn = calculate_bleu(cn_hypothesis, cn_references)
    bleu_en = calculate_bleu(en_hypothesis, en_references)
    rouge1_cn, rouge2_cn = calculate_rouge(cn_hypothesis, cn_references)
    rouge1_en, rouge2_en = calculate_rouge(en_hypothesis, en_references)
    mean_loss = total_loss / total_n
    mean_en_loss = total_en_loss / total_n
    mean_cn_loss = total_cn_loss / total_n
    logging.info("loss: %.2f\t loss-cn: %.2f \t loss-en %.2f \t bleu-cn: %3.2f\t bleu-en: %3.2f \t rouge1-cn: %3.2f \t rouge1-en: %3.2f \t rouge2-cn: %3.2f \t rouge2-en: %3.2f"
                 % (mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en))
    saver.save(step, mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en)



def main():
    logging.info("Build dataset...")
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields

    pad_ids = {"source": fields["source"].pad_id,
               "summary_cn": fields["summary_cn"].pad_id,
               "summary_en": fields["summary_en"].pad_id}
    vocab_sizes = {"source": len(fields["source"].vocab),
                   "summary_cn": len(fields["summary_cn"].vocab),
                   "summary_en": len(fields["summary_en"].vocab)}

    # load checkpoint from model_path
    for ckpt in range(1100000,1345001, 5000):
        model_path = opt.model_path + '/checkpoint-step-' + str(ckpt)
        logging.info("Load checkpoint from %s." % model_path)
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)

        logging.info("Build model...")
        model = NMTModel.load_model(checkpoint["opt"], pad_ids, vocab_sizes, checkpoint["model"]).to(device).eval()
        criterion_cn = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["summary_cn"], pad_ids["summary_cn"]).to(device)
        criterion_en = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["summary_en"], pad_ids["summary_en"]).to(device)

        n_step = ckpt


        logging.info("Start translation...")
        with torch.set_grad_enabled(False):
            valid(model, criterion_cn, criterion_en, valid_dataset, n_step)


if __name__ == '__main__':
    main()

