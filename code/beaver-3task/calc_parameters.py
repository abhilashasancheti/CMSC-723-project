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
	task1 = 0
	task2 = 0
	task3 = 0
	encoder = 0
	for layer_tensor_name, tensor in tensor_list:
		#print('Layer {}: {} elements'.format(layer_tensor_name, torch.numel(tensor)))
		total+=torch.numel(tensor)
		print(tensor)
		break
		if 'task3_decoder' in layer_tensor_name:
			task3+=torch.numel(tensor)
		if 'task1_decoder' in layer_tensor_name:
			task1+=torch.numel(tensor)
		if 'task2_decoder' in layer_tensor_name:
			task2+=torch.numel(tensor)
		if 'encoder' in layer_tensor_name:
			encoder+=torch.numel(tensor)

	print('task1: {}, task2: {}, task3: {}, encoder: {}, Total parameters: {}'.format(task1, task2, task3, encoder, total))

 	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# 	for p in model.parameters():
# 		print(p)
# 		break
# 		if p.requires_grad:
# 			print(p.numel()) 
	print(pytorch_total_params)



if __name__=='__main__':
	train_dataset = build_dataset(opt, opt.train, opt.vocab, device, train=True)
	fields = train_dataset.fields
	logging.info("Build model...")
	pad_ids = {"src": fields["src"].pad_id,"task1_tgt": fields["task1_tgt"].pad_id,"task2_tgt": fields["task2_tgt"].pad_id,"task3_tgt": fields["task3_tgt"].pad_id}
	vocab_sizes = {"src": len(fields["src"].vocab),"task1_tgt": len(fields["task1_tgt"].vocab),"task2_tgt": len(fields["task2_tgt"].vocab),"task3_tgt": len(fields["task3_tgt"].vocab)}
	model = NMTModel.load_model(opt, pad_ids, vocab_sizes).to(device)
	calc_parameters_torch(model)
	

