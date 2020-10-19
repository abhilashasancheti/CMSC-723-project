# -*- coding: utf-8 -*-
import logging
import datetime

import torch
import torch.cuda

from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt
from beaver.utils.metric import calculate_rouge

import torch.nn as nn
import torch.distributed as dist
## multi processing libraries
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP




logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()
logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


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

    bleu_cn = calculate_bleu(cn_hypothesis, cn_references)
    bleu_en = calculate_bleu(en_hypothesis, en_references)
    rouge1_cn, rouge2_cn = calculate_rouge(cn_hypothesis, cn_references)
    rouge1_en, rouge2_en = calculate_rouge(en_hypothesis, en_references)
    mean_loss = total_loss / total_n
    mean_en_loss = total_en_loss / total_n
    mean_cn_loss = total_cn_loss / total_n
    logging.info("loss: %.2f\t loss-cn: %.2f \t loss-en %.2f \t bleu-cn: %3.2f\t bleu-en: %3.2f \t rouge1-cn: %3.2f \t rouge1-en: %3.2f \t rouge2-cn: %3.2f \t rouge2-en: %3.2f"
                 % (mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en)
    return mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en

def train(model, pad_ids, vocab_sizes, criterion_cn, criterion_en, optimizer, train_dataset, valid_dataset):
    total_loss = total_cn_loss = total_en_loss = 0.0
    hist_valid_scores = []
    num_trial = 0
    iteration = 0
    patience = 0
    best_n_step = 0
    
    model.zero_grad()
    for i, batch in enumerate(train_dataset):
        cn_scores, en_scores = model(batch.source, batch.summary_cn, batch.summary_en)
        cn_loss = criterion_cn(cn_scores, batch.summary_cn)
        en_loss = criterion_en(en_scores, batch.summary_en)

        loss = cn_loss + en_loss
        loss.backward()
        total_loss += loss.data
        total_cn_loss += cn_loss.data
        total_en_loss += en_loss.data
        iteration = i+1

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_loss = total_loss / opt.report_every / opt.grad_accum
                mean_en_loss = total_en_loss / opt.report_every / opt.grad_accum
                mean_cn_loss = total_cn_loss / opt.report_every / opt.grad_accum
                logging.info("step: %7d\t loss: %.4f \t loss-cn: %.4f \t loss-en: %.4f"
                             % (optimizer.n_step, mean_loss, mean_cn_loss, mean_en_loss))
                with open(saver.model_path +"/train.log", "a", encoding="UTF-8") as log:
                    log.write("%s\t step: %7d\t loss: %.4f \t loss-cn: %.4f \t loss-en: %.4f\n"
                             % (datetime.datetime.now(), optimizer.n_step, mean_loss, mean_cn_loss, mean_en_loss))
                total_loss = total_cn_loss = total_en_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    logging.info('begin validation ...')
                    mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en = valid(model, criterion_cn, criterion_en, valid_dataset, optimizer.n_step)

#                 valid_metric = rouge1_en

#                 is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
#                 hist_valid_scores.append(valid_metric)

#                 if is_better:
#                     patience = 0
#                     logging.info('save currently the best model to [%s]' % saver.model_path)
#                     checkpoint = {"model": model.state_dict(), "opt": opt}
#                     saver.save(checkpoint, optimizer.n_step, mean_loss, mean_cn_loss, mean_en_loss, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en)
#                     best_n_step = optimizer.n_step
#                 elif patience < int(opt.patience):
#                     patience += 1
#                     logging.info('hit patience %d' % patience)

#                     if patience == int(opt.patience):
#                         num_trial += 1
#                         logging.info('hit #%d trial' % num_trial)
#                         if num_trial == int(opt.max_num_trial):
#                             logging.info('early stop!')
#                             exit(0)

#                         # set the file name of checkpoint to load
#                         filename = "checkpoint-step-%06d" % best_n_step
#                         opt.train_from = os.path.join(saver.model_path, filename)
#                         # optimizer, lr, and restore from previously best checkpoint load model
#                         model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
#                         logging.info('restore parameters of the optimizers')
#                         optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, best_n_step)

#                         # reset patience
#                         patience = 0

                if iteration == int(opt.max_iterations):
                    logging.info('reached maximum number of iterations!')
                    exit(0)
                model.train()
        del loss


def main():
    logging.info("Build dataset...")
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    logging.info("Built dataset valid ...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    logging.info("Built dataset train ...")
    
    fields = valid_dataset.fields = train_dataset.fields
    logging.info("Build model...")

    pad_ids = {"source": fields["source"].pad_id,
               "summary_cn": fields["summary_cn"].pad_id,
               "summary_en": fields["summary_en"].pad_id}
    vocab_sizes = {"source": len(fields["source"].vocab),
                   "summary_cn": len(fields["summary_cn"].vocab),
                   "summary_en": len(fields["summary_en"].vocab)}
    print(vocab_sizes)

    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
    criterion_cn = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["summary_cn"], pad_ids["summary_cn"]).to(device)
    criterion_en = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["summary_en"], pad_ids["summary_en"]).to(device)

    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, pad_ids, vocab_sizes, criterion_cn, criterion_en, optimizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
