import json

import torch
import os
import datetime


class SaverTest(object):
    def __init__(self, opt):
        self.ckpt_names = []
        self.model_path = opt.model_path + datetime.datetime.now().strftime("-%y%m%d-%H%M%S")
        self.max_to_keep = opt.max_to_keep
        os.mkdir(self.model_path)

        with open(os.path.join(self.model_path, "params.json"), "w", encoding="UTF-8") as log:
            log.write(json.dumps(vars(opt), indent=4) + "\n")

    def save(self, step, loss, loss_cn, loss_en, bleu_cn, bleu_en, rouge1_cn, rouge1_en, rouge2_cn, rouge2_en):

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t" % datetime.datetime.now())
            log.write("step: %6d\t" % step)
            log.write("loss: %.2f\t" % loss)
            log.write("loss-cn: %.2f\t" % loss_cn)
            log.write("loss-en: %.2f\t" % loss_en)
            log.write("bleu-cn: %3.2f\t" % bleu_cn)
            log.write("bleu-en: %3.2f\t" % bleu_en)
            log.write("rouge1-cn: %3.2f\t" % rouge1_cn)
            log.write("rouge1-en: %3.2f\t" % rouge1_en)
            log.write("rouge2-cn: %3.2f\t" % rouge2_cn)
            log.write("rouge2-en: %3.2f\t" % rouge2_en)
            log.write("\n")

