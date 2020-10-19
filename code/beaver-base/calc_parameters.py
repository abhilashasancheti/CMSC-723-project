# -*- coding: utf-8 -*-
import logging
import datetime
import os

import torch
import torch.cuda

from beaver.data import build_dataset
from beaver.model import NMTModel
from beaver.utils import parseopt, get_device, printing_opt


device = get_device()
opt = parseopt.parse_train_args()

def calc_parameters_torch(model):
	tensor_dict = model.state_dict()#torch.load(model, map_location='cpu') # OrderedDict
	tensor_list = list(tensor_dict.items())
	total = 0
	decoder = 0
	encoder = 0
	for layer_tensor_name, tensor in tensor_list:
		#print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
		total+=torch.numel(tensor)
		if 'encoder' in layer_tensor_name:
			encoder+=torch.numel(tensor)
		if 'decoder' in layer_tensor_name:
			decoder+=torch.numel(tensor)
	print('encoder: {}, Decoder: {}, Total parameters: {}'.format(encoder, decoder, total))
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(pytorch_total_params)


if __name__=='__main__':
	train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
	fields = train_dataset.fields
	logging.info("Build model...")
	pad_ids = {"src": fields["src"].pad_id, "tgt": fields["tgt"].pad_id}
	vocab_sizes = {"src": len(fields["src"].vocab), "tgt": len(fields["tgt"].vocab)}
	model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
	calc_parameters_torch(model)


