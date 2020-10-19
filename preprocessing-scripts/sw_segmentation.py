"""SentencePiece Tokenization for Wiki Dataset

Example:
  * python sw_segmentation.py --word --unigram/bpe --phase train/test/valid
"""
import gzip
import json
import jieba
import subprocess
import argparse
from pathlib import Path

import sentencepiece as spm
import numpy as np
from tqdm import tqdm


VOC_SIZE=40000

# need to change the paths accordingly
DATAPATH = TMPPATH = "../data/MT/pre-processed/test.target.mt"
#TMPPATH = "/tmp/tmp_texts.txt"
TMPPATH_WORD = "../data/MT/pre-processed/tmp/tmp_words.txt"
MODEL_PREFIX = "../data/processed/{algorithm}_{seg_word}_model"


def json_to_txt():
    print('in function')


def fit_model(seg_word=True, algorithm="bpe", phase='train'):
    if not Path(TMPPATH).exists():
        json_to_txt()

    if seg_word:
        print("Performing word segmentation...")
        with open(DATAPATH,'r') as f:
            data = f.readlines()
        
        f = open(TMPPATH_WORD, 'w')
        for text in data:
            text = text.lower()
            text_word = jieba.cut(text, cut_all=False)
            #print(text_word)
            new_text = (" ".join(text_word))
            f.write("{}\n".format(new_text.strip()))
        f.close()


    # Train Model
    if phase=='train':
        print("Training model...")
        spm.SentencePieceTrainer.Train(
            '--input={} --model_prefix={} --vocab_size={} --character_coverage=0.995 --model_type={algo}'.format(
                TMPPATH_WORD if seg_word else TMPPATH,
                MODEL_PREFIX.format(algorithm=algorithm, seg_word=seg_word), VOC_SIZE, algo=algorithm
            )
        )


def tokenize(phase, seg_word=True, algorithm="bpe"):
    print("Tokenizing...")
    sp = spm.SentencePieceProcessor()
    sp.Load(MODEL_PREFIX.format(
        algorithm=algorithm, seg_word=seg_word) + ".model")
    # need to modify paths accordingly
    with open(TMPPATH_WORD if seg_word else TMPPATH) as f, open("../data/MT/processed/"+phase+".target.mt",'w') as fw:
        for _, sentence in tqdm(enumerate(f.readlines())):
            fw.write("{}\n".format(" ".join(sp.EncodeAsPieces(sentence.lower())).strip()))



def main(word, bpe, phase):
    seg_word = True if word else False
    algorithm = "bpe" if bpe else "unigram"
    #if phase=='train':
    fit_model(seg_word, algorithm, phase)
    tokenize(phase,seg_word, algorithm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="sub-word segmentation", description="Segments words into subwords")
    parser.add_argument('-word', '--word', action='store_true', help="Add this argument if word segmentation needs to be done")
    parser.add_argument('-bpe', '--bpe', action='store_true', help="Add this argument if unigram subword segmentation algorithm")
    parser.add_argument('-unigram', '--unigram', action='store_true', help="Add this argument if BPE subword segmentation algorithm")
    parser.add_argument('-phase', '--phase', required=True, help="train/test/valid data")

    args = parser.parse_args()
    word = args.word
    phase = args.phase
    bpe = args.bpe

    main(word,bpe,phase)
