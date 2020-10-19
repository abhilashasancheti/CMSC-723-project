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

import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


def valid(model, criterion_task1, criterion_task2, criterion_task3, valid_dataset, step):
    model.eval()
    total_n = 0
    total_task1_loss = total_task2_loss = total_task3_loss = total_task2_3_loss = 0.0
    task1_hypothesis, task1_references = [], []
    task2_hypothesis, task2_references = [], []
    task3_hypothesis, task3_references = [], []

    for i, (batch, flag) in enumerate(valid_dataset):

        if flag:
            task1_scores = model(batch.src, batch.tgt, None, None, flag)
            loss = criterion_task1(task1_scores, batch.tgt)
        else:
            task2_scores, task3_scores = model(batch.src, None, batch.summary_cn, batch.summary_en, flag)
            task2_loss = criterion_task2(task2_scores, batch.summary_cn)
            task3_loss = criterion_task3(task3_scores, batch.summary_en)
            loss = task2_loss + task3_loss

        if flag:
            _, task1_predictions = task1_scores.topk(k=1, dim=-1)
        else:
            _, task2_predictions = task2_scores.topk(k=1, dim=-1)
            _, task3_predictions = task3_scores.topk(k=1, dim=-1)



        if flag:  # task1
            total_task1_loss += loss.data
            task1_hypothesis += [valid_dataset.fields["task1_tgt"].decode(p) for p in task1_predictions]
            task1_references += [valid_dataset.fields["task1_tgt"].decode(t) for t in batch.tgt]
        else:
            total_task2_3_loss += loss.data
            total_task2_loss += task2_loss.data
            total_task3_loss += task3_loss.data
            task2_hypothesis += [valid_dataset.fields["task2_tgt"].decode(p) for p in task2_predictions]
            task2_references += [valid_dataset.fields["task2_tgt"].decode(t) for t in batch.summary_cn]
            task3_hypothesis += [valid_dataset.fields["task3_tgt"].decode(p) for p in task3_predictions]
            task3_references += [valid_dataset.fields["task3_tgt"].decode(t) for t in batch.summary_en]

        total_n += 1
        del loss

    bleu_task1 = calculate_bleu(task1_hypothesis, task1_references) if len(task1_hypothesis)>0 else 0
    bleu_task2 = calculate_bleu(task2_hypothesis, task2_references) if len(task2_hypothesis)>0 else 0
    bleu_task3 = calculate_bleu(task3_hypothesis, task3_references) if len(task3_hypothesis)>0 else 0

    if len(task1_hypothesis)>0:
        rouge1_task1, rouge2_task1 = calculate_rouge(task1_hypothesis, task1_references)
    else:
        rouge1_task1, rouge2_task1 = 0,0
    
    if len(task2_hypothesis)>0:
        rouge1_task2, rouge2_task2 = calculate_rouge(task2_hypothesis, task2_references)
    else:
        rouge1_task2, rouge2_task2 = 0,0
        
    if len(task3_hypothesis)>0:
        rouge1_task3, rouge2_task3 = calculate_rouge(task3_hypothesis, task3_references) 
    else:
        rouge1_task3, rouge2_task3 = 0,0
    
    mean_task1_loss = total_task1_loss / total_n
    mean_task2_loss = total_task2_loss / total_n
    mean_task3_loss = total_task3_loss / total_n
    logging.info("loss-task1: %.2f \t loss-task2 %.2f \t loss-task3 %.2f \t bleu-task1: %3.2f \t bleu-task2: %3.2f \t bleu-task3: %3.2f \t rouge1-task1: %3.2f \t rouge1-task2: %3.2f \t rouge1-task3: %3.2f \t rouge2-task1: %3.2f \t rouge2-task2: %3.2f \t rouge2-task3: %3.2f"
                 % (mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3 ))
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3)
    return mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3

def train(model, pad_ids, vocab_sizes, criterion_task1, criterion_task2, criterion_task3, optimizer, train_dataset, valid_dataset):
    total_task1_loss = total_task2_loss = total_task3_loss = total_task2_3_loss = 0.0
    hist_valid_scores = []
    num_trial = 0
    iteration = 0
    patience = 0
    best_n_step = 0

    model.zero_grad()
    for i, (batch, flag) in enumerate(train_dataset):


        if flag:
            task1_scores = model(batch.src, batch.tgt, None, None, flag)
            loss = criterion_task1(task1_scores, batch.tgt)
        else:
            task2_scores, task3_scores = model(batch.src, None, batch.summary_cn, batch.summary_en, flag)
            task2_loss = criterion_task2(task2_scores, batch.summary_cn)
            task3_loss = criterion_task3(task3_scores, batch.summary_en)
            loss = task2_loss + task3_loss

        loss.backward()

        if flag:  # task1
            total_task1_loss += loss.data
        else:
            total_task2_3_loss += loss.data
            total_task2_loss += task2_loss.data
            total_task3_loss += task3_loss.data

        iteration = i+1

        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_task1_loss = total_task1_loss / opt.report_every / opt.grad_accum * 2
                mean_task2_loss = total_task2_loss / opt.report_every / opt.grad_accum * 2
                mean_task3_loss = total_task3_loss / opt.report_every / opt.grad_accum * 2
                logging.info("step: %7d\t loss-task1: %.4f \t loss-task2: %.4f \t loss-task3: %.4f"
                             % (optimizer.n_step, mean_task1_loss, mean_task2_loss,mean_task3_loss))
                with open(saver.model_path +"/train.log", "a", encoding="UTF-8") as log:
                    log.write("%s\t step: %7d\t loss-task1: %.4f \t loss-task2: %.4f \t loss-task3: %.4f\n"
                             % (datetime.datetime.now(), optimizer.n_step, mean_task1_loss, mean_task2_loss, mean_task3_loss))
                total_task1_loss = total_task2_loss = total_task3_loss = total_task2_3_loss= 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    logging.info('begin validation ...')
                    mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3 = valid(model, criterion_task1, criterion_task2, criterion_task3, valid_dataset, optimizer.n_step)

#                 valid_metric = rouge1_task3

#                 is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
#                 hist_valid_scores.append(valid_metric)

#                 if is_better:
#                     patience = 0
#                     logging.info('save currently the best model to [%s]' % saver.model_path)
#                     checkpoint = {"model": model.state_dict(), "opt": opt}
#                     saver.save(checkpoint, optimizer.n_step, mean_task1_loss, mean_task2_loss, mean_task3_loss, bleu_task1, bleu_task2, bleu_task3, rouge1_task1, rouge1_task2, rouge1_task3, rouge2_task1, rouge2_task2, rouge2_task3)
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

#                 if iteration == int(opt.max_iterations):
#                     logging.info('reached maximum number of iterations!')
#                     exit(0)
                model.train()
        del loss


def main():
    logging.info("Build dataset...")
    train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
    valid_dataset = build_dataset(opt, opt.valid, opt.vocab, device, train=False)
    fields = valid_dataset.fields = train_dataset.fields
    logging.info("Build model...")

    pad_ids = {"src": fields["src"].pad_id,
               "task1_tgt": fields["task1_tgt"].pad_id,
               "task2_tgt": fields["task2_tgt"].pad_id,
               "task3_tgt": fields["task3_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task2_tgt"].vocab),
                   "task3_tgt": len(fields["task3_tgt"].vocab)}

    print(vocab_sizes)

    model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
#         model = nn.DataParallel(model, device_ids=[0,1])
    
    
    # for MT
    criterion_task1 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task1_tgt"], pad_ids["task1_tgt"]).to(device)
    # for MS
    criterion_task2 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task2_tgt"], pad_ids["task2_tgt"]).to(device)
    # for CLS
    criterion_task3 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task3_tgt"], pad_ids["task3_tgt"]).to(device)

    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, pad_ids, vocab_sizes, criterion_task1, criterion_task2, criterion_task3, optimizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
