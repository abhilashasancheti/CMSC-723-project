from collections import namedtuple

src_path = '../data/processed/train.source'
tgt_path = '../data/processed/train.target.ms'
def read_file(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield line.strip().split()

Example = namedtuple("Example", ['src', 'tgt'])

examples = []
for src_line, tgt_line in zip(read_file(src_path), read_file(tgt_path)):
    print("read..", len(examples), len(src_line))
    if len(src_line) > 100:
    	continue
    try:
        examples.append(Example(src_line, tgt_line))
    except:
    	continue
