import os
import codecs


def undo_bpe(sentences):
    new_sentences = []
    for sentence in sentences:
        sentence = sentence.replace(' ', '')
        sentence = sentence.replace('▁', ' ')
        sentence = sentence.replace('__________', '')
        sentence = sentence.strip()
        new_sentences.append(sentence)
    return new_sentences



def read_file(filename):
    with codecs.open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        lines = [line.lower().strip() for line in lines]
    return lines

def write_files(filename, lines):
    with codecs.open(filename, 'w', encoding='utf-8', errors='ignore') as f:
        for line in lines:
            f.write('{}\n'.format(line.strip().lower()))

print("read_files........")
train_target_mt = read_file('./MT/processed/train.target.mt')
train_source_mt = read_file('./MT/processed/train.source.mt')

valid_target_mt = read_file('./MT/processed/valid.target.mt')
valid_source_mt = read_file('./MT/processed/valid.source.mt')

test_target_mt = read_file('./MT/processed/test.target.mt')
test_source_mt = read_file('./MT/processed/test.source.mt')


## undo bpe 
print("undo_bpe........")
undo_bpe_train_target_mt =  undo_bpe(train_target_mt)
undo_bpe_train_source_mt =  undo_bpe(train_source_mt)

undo_bpe_valid_target_mt =  undo_bpe(valid_target_mt)
undo_bpe_valid_source_mt =  undo_bpe(valid_source_mt)

undo_bpe_test_target_mt =  undo_bpe(test_target_mt)
undo_bpe_test_source_mt =  undo_bpe(test_source_mt)


write_files('./MT/pre-processed/train.target.mt', undo_bpe_train_target_mt)
write_files('./MT/pre-processed/train.source.mt', undo_bpe_train_source_mt)
write_files('./MT/pre-processed/valid.target.mt', undo_bpe_valid_target_mt)
write_files('./MT/pre-processed/valid.source.mt', undo_bpe_valid_source_mt)
write_files('./MT/pre-processed/test.target.mt', undo_bpe_test_target_mt)
write_files('./MT/pre-processed/test.source.mt', undo_bpe_test_source_mt)



