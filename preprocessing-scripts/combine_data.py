import os

def read_file(filename):
	with open(filename, 'r') as f:
		lines = f.readlines()
		lines = [line.strip() for line in lines]
	return lines


if __name__=='__main__':

	
	# for source language
	lines1 = read_file('../data/MT/processed/train.source.mt')  # modify path to train.source.mt
	lines2 = read_file('../data/processed/train.source')		# modify path to train.source
	with open('../data/MT/processed/train.source.vocab', 'w') as f:
		for line in lines1:
			f.write('{}\n'.format(line.strip()))
		for line in lines2:
		f.write('{}\n'.format(line.strip()))
					
	# for target language
	lines1 = read_file('../data/MT/processed/train.target.mt')	# modify path to train.target.mt
	lines2 = read_file('../data/processed/train.target.cls')  # modify path to train.target.cls
	with open('../data/MT/processed/train.target.vocab', 'w') as f:  # modify accordingly
		for line in lines1:
			f.write('{}\n'.format(line.strip()))
		for line in lines2:
			f.write('{}\n'.format(line.strip()))