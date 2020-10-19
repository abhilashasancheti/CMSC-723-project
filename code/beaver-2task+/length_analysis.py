import matplotlib.pyplot as plt
import codecs
import numpy as np

def plot_length(length_ref, source, length_mt_cls, length_cls, length_ms_cls, length_ms_mt_cls):
	plt.plot(length_mt_cls, color='red')
	plt.plot(length_ref, color='pink')
# 	plt.plot(length_cls, color='green')
# 	plt.plot(length_ms_cls, color='orange')
# 	plt.plot(length_ms_mt_cls, color='blue')
	plt.savefig('length_analysis_1.pdf')


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

source_lines = read_file('/home/code-base/archive/data/processed/test.source.corrected.50k')
ref_lines = read_file('/home/code-base/archive/data/processed/test.target.cls')
mt_cls_lines = read_file('/home/code-base/archive/outputs/test.2+task.cls.50k.output.1395000') 
cls_lines = read_file('/home/code-base/archive/outputs/test.2+task.cls.50k.output.890000')
ms_cls_lines = read_file('/home/code-base/archive/outputs/test.2task.cls.output.1200000') 
ms_mt_cls_lines = read_file('/home/code-base/archive/outputs/test.3task.cls.output.1060000')

processed_source_lines = undo_bpe(source_lines)
processed_ref_lines = undo_bpe(ref_lines)
processed_mt_cls_lines = undo_bpe(mt_cls_lines)
processed_cls_lines = undo_bpe(cls_lines)
processed_ms_cls_lines = undo_bpe(ms_cls_lines)
processed_ms_mt_cls_lines = undo_bpe(ms_mt_cls_lines)

length_mt_cls = []
length_cls = []
length_ms_cls = []
length_ms_mt_cls = []
length_source = []
length_ref = []

for ref, source, mt_cls_line, cls_line, ms_cls_line, ms_mt_cls_line in zip(processed_ref_lines, processed_source_lines, processed_mt_cls_lines, processed_cls_lines, processed_ms_cls_lines, processed_ms_mt_cls_lines):
    length_mt_cls.append(len(mt_cls_line.strip().split()))
    length_cls.append(len(cls_line.strip().split()))
    length_ms_cls.append(len(ms_cls_line.strip().split()))
    length_ms_mt_cls.append(len(ms_mt_cls_line.strip().split()))
    length_source.append(len(source.strip().split()))
    length_ref.append(len(ref.strip().split()))
plot_length(length_ref, length_source, length_mt_cls, length_cls, length_ms_cls, length_ms_mt_cls)
print("Avg. source %.2f, Avg. reference %.2f, Avg. CLS+MT %.2f, Avg. CLS %.2f, Avg. CLS+MS %.2f, Avg. CLS+MS+MT %.2f" % (np.mean(length_source), np.mean(length_ref), np.mean(length_mt_cls), np.mean(length_cls), np.mean(length_ms_cls), np.mean(length_ms_mt_cls)))