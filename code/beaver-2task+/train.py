# -*- coding: utf-8 -*-
import logging
import datetime
import os

import torch
import torch.cuda
import torch.nn as nn
import torch.distributed as dist
## multi processing libraries
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP



from beaver.data import build_dataset
from beaver.infer import beam_search
from beaver.loss import WarmAdam, LabelSmoothingLoss
from beaver.model import NMTModel
from beaver.utils import Saver
from beaver.utils import calculate_bleu
from beaver.utils import parseopt, get_device, printing_opt
from beaver.utils.metric import calculate_rouge




logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)
opt = parseopt.parse_train_args()

device = get_device()

logging.info("\n" + printing_opt(opt))

saver = Saver(opt)


def valid(model, criterion_task1, criterion_task2, valid_dataset, step):
    # if torch.cuda.device_count() >1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0,1])

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
    mean_task1_loss = total_task1_loss / total_n
    mean_task2_loss = total_task2_loss / total_n
    logging.info("loss-task1: %.2f \t loss-task2 %.2f \t bleu-task1: %3.2f\t bleu-task2: %3.2f \t rouge1-task1: %3.2f \t rouge1-task2: %3.2f \t rouge2-task1: %3.2f \t rouge2-task2: %3.2f"
                 % (mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2))
    
    checkpoint = {"model": model.state_dict(), "opt": opt}
    saver.save(checkpoint, step, mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2)

    return mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2


def train(model, pad_ids, vocab_sizes, criterion_task1, criterion_task2, optimizer, train_dataset, valid_dataset):
    total_task1_loss = total_task2_loss = 0.0
    hist_valid_scores = []
    num_trial = 0
    iteration = 0
    patience = 0
    best_n_step = 0
    # if torch.cuda.device_count() >1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = nn.DataParallel(model, device_ids=[0,1])

    model.zero_grad()
    for i, (batch, flag) in enumerate(train_dataset):
        #print("Outside: input size", batch.src.size(), "output_size", batch.tgt.size())
        scores = model(batch.src, batch.tgt, flag)

        if flag:
            loss = criterion_task1(scores, batch.tgt)
        else:
            loss = criterion_task2(scores, batch.tgt)
        loss.backward()
        if flag:  # task1
            total_task1_loss += loss.data
        else:
            total_task2_loss += loss.data

        iteration+=1
        if (i + 1) % opt.grad_accum == 0:
            optimizer.step()
            model.zero_grad()

            if optimizer.n_step % opt.report_every == 0:
                mean_task1_loss = total_task1_loss / opt.report_every / opt.grad_accum * 2
                mean_task2_loss = total_task2_loss / opt.report_every / opt.grad_accum * 2
                logging.info("step: %7d\t loss-task1: %.4f \t loss-task2: %.4f"
                             % (optimizer.n_step, mean_task1_loss, mean_task2_loss))
                with open(saver.model_path +"/train.log", "a", encoding="UTF-8") as log:
                    log.write("%s\t step: %7d\t loss-task1: %.4f \t loss-task2: %.4f\n"
                             % (datetime.datetime.now(), optimizer.n_step, mean_task1_loss, mean_task2_loss))
                total_task1_loss = total_task2_loss = 0.0

            if optimizer.n_step % opt.save_every == 0:
                with torch.set_grad_enabled(False):
                    logging.info('begin validation ...')
                    mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2 = valid(model, criterion_task1, criterion_task2, valid_dataset, optimizer.n_step)

                # valid_metric = rouge1_task2

                # is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                # hist_valid_scores.append(valid_metric)

                # if is_better:
                #     patience = 0
                #     logging.info('save currently the best model to [%s]' % saver.model_path)
                #     checkpoint = {"model": model.state_dict(), "opt": opt}
                #     saver.save(checkpoint, optimizer.n_step, mean_task1_loss, mean_task2_loss, bleu_task1, bleu_task2, rouge1_task1, rouge1_task2, rouge2_task1, rouge2_task2)
                #     best_n_step = optimizer.n_step
                # elif patience < int(opt.patience):
                #     patience += 1
                #     logging.info('hit patience %d' % patience)

                #     if patience == int(opt.patience):
                #         num_trial += 1
                #         logging.info('hit #%d trial' % num_trial)
                #         if num_trial == int(opt.max_num_trial):
                #             logging.info('early stop!')
                #             exit(0)

                #         # set the file name of checkpoint to load
                #         filename = "checkpoint-step-%06d" % best_n_step
                #         opt.train_from = os.path.join(saver.model_path, filename)
                #         # optimizer, lr, and restore from previously best checkpoint load model
                #         model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
                #         logging.info('restore parameters of the optimizers')
                #         optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, best_n_step)

                #         # reset patience
                #         patience = 0

                if iteration == int(opt.max_iterations):
                    logging.info('reached maximum number of iterations!')
                    exit(0)
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
               "task2_tgt": fields["task2_tgt"].pad_id}
    vocab_sizes = {"src": len(fields["src"].vocab),
                   "task1_tgt": len(fields["task1_tgt"].vocab),
                   "task2_tgt": len(fields["task2_tgt"].vocab)}
    print(vocab_sizes)
    model = NMTModel.load_model(opt, pad_ids, vocab_sizes)

    # for multi-gpu processing, this will copy the model on multiple gpus and then split the data on the gpus
#     if torch.cuda.device_count() > 1:
#         print("Let's use", torch.cuda.device_count(), "GPUs!")
        
#         model = nn.DataParallel(model, device_ids=[0,1])
    
    model = model.to(device)
    criterion_task1 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task1_tgt"], pad_ids["task1_tgt"]).to(device)
    criterion_task2 = LabelSmoothingLoss(opt.label_smoothing, vocab_sizes["task2_tgt"], pad_ids["task2_tgt"]).to(device)

    n_step = int(opt.train_from.split("-")[-1]) if opt.train_from else 1
    optimizer = WarmAdam(model.parameters(), opt.lr, opt.hidden_size, opt.warm_up, n_step)

    logging.info("start training...")
    train(model, pad_ids, vocab_sizes, criterion_task1, criterion_task2, optimizer, train_dataset, valid_dataset)


if __name__ == '__main__':
    main()
