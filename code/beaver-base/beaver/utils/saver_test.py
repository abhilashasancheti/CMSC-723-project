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

    def save(self, step, bleu, loss):

        with open(os.path.join(self.model_path, "log"), "a", encoding="UTF-8") as log:
            log.write("%s\t step: %6d\t loss: %.2f\t bleu: %.2f\n" % (datetime.datetime.now(), step, loss, bleu))
            
