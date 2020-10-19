import os
import re
import subprocess
import tempfile
import codecs
#from pyrouge import Rouge155
#from rouge_scorer import calculateRouge
#import rouge
from pythonrouge.pythonrouge import Pythonrouge
import os
import re
import subprocess
import tempfile




def calculate_bleu(hypotheses, references, lowercase=False):
    hypothesis_file = tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", delete=False)
    hypothesis_file.write("\n".join(hypotheses) + "\n")
    hypothesis_file.close()
    reference_file = tempfile.NamedTemporaryFile(mode="w", encoding="UTF-8", delete=False)
    reference_file.write("\n".join(references) + "\n")
    reference_file.close()
    return file_bleu(hypothesis_file.name, reference_file.name, lowercase)


def file_bleu(hypothesis, reference, lowercase=False):
    # ../../../tools/multi-bleu.perl, so take 3 levels up.
    beaver_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    multi_bleu_path = os.path.join(beaver_path, "tools", "multi-bleu.perl")
    with open(hypothesis, "r") as read_pred, open(os.devnull, "w") as black_hole:
        bleu_cmd = ["perl", multi_bleu_path]
        if lowercase:
            bleu_cmd += ["-lc"]
        bleu_cmd += [reference]
        try:
            bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=black_hole).decode("utf-8")
            bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
        except subprocess.CalledProcessError:
            bleu_score = -1.0
        return float(bleu_score)


    
N_TEST = 3000

ROUGE_PATH = '/home/code-base/runtime/CLS/code/evaluation/pythonrouge/pythonrouge/RELEASE-1.5.5/ROUGE-1.5.5.pl'
DATA_PATH  = '/home/code-base/runtime/CLS/code/evaluation/pythonrouge/pythonrouge/RELEASE-1.5.5/data/'


OUT_DIR = '/home/code-base/archive/outputs/'
INP_DIR = '/home/code-base/archive/data/processed/'

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
            f.write('{}\n'.format(line.strip()))
    

def liPaperEvaluation(pred_y, true_y):
    rouge = Pythonrouge(summary_file_exist=False, summary=pred_y, reference=true_y, f_measure_only=True, n_gram=2, ROUGE_SU4=False, ROUGE_L=True, stemming=False, stopwords=False, word_level=False, length_limit=False, use_cf=False, cf=95, scoring_formula="average", resampling=True, samples=1000)
    result = rouge.calc_score()
    print(result)


if __name__=='__main__':
    # for CLS task
    ref = read_file(INP_DIR + 'test.target.cls')
    base_output = read_file(OUT_DIR + 'test.2+task.cls.50k.output.890000')
    task2_output = read_file(OUT_DIR + 'test.2task.cls.output.1200000')
    #ms_task2_output = read_file(OUT_DIR + 'test.2task.ms.output.')
    task3_output = read_file(OUT_DIR + 'test.3task.cls.output.2050000')
    task2_plus_output = read_file(OUT_DIR + 'test.2+task.cls.50k.output.1055000')
    task2_2p_3task_output = read_file(OUT_DIR + 'test.2.2p.3task.cls.output.1140000')
    task2p_2_3task_output = read_file(OUT_DIR + 'test.2p.2.3task.cls.output.1460000')
    #pipeline_output = read_file(OUT_DIR + 'test.pipeline.cls.output')
    #ms_pipeline_output = read_file(OUT_DIR + 'test.ms.pipeline.output.560000')
    #task_first20 = read_file(OUT_DIR + 'test.first20.cls.output')
    
    # for MS task
    #ms_ref = read_file(INP_DIR + 'test.target.ms')
    #task3_ms_output = read_file(OUT_DIR + 'test.3task.ms.output.1900000')
    #task2_ms_output = read_file(OUT_DIR + 'test.2task.ms.output.1440000')
    # for MT task

    sentences_ref = undo_bpe(ref)
    base_sentences_output = undo_bpe(base_output)
    task2_sentences_output = undo_bpe(task2_output)
    task3_sentences_output = undo_bpe(task3_output)
    task2_plus_sentences_output = undo_bpe(task2_plus_output)
    task2_2p_3task_sentences_output = undo_bpe(task2_2p_3task_output)
    task2p_2_3task_sentences_output = undo_bpe(task2p_2_3task_output)
    
    #pipeline_sentences_output = pipeline_output
    #ms_pipeline_sentences_output = undo_bpe(ms_pipeline_output)

    #ms_sentences_ref = undo_bpe(ms_ref)
    #task2_ms_sentences_output = undo_bpe(task2_ms_output)
    #task3_ms_sentences_output = undo_bpe(task3_ms_output)

#     if not os.path.exists(OUT_DIR + 'decoded/'):
#         os.mkdir(OUT_DIR + 'decoded/')

#     if not os.path.exists(INP_DIR + 'reference/'):
#         os.mkdir(INP_DIR + 'reference/')

    # write files in normal format- undo bpe

    #write_files( INP_DIR + 'reference/ref.reference.txt', sentences_ref)
    #write_files( OUT_DIR + 'decoded/cls.base.decoded.txt', base_sentences_output)
    #write_files( OUT_DIR + 'decoded/cls.2task.decoded.txt', task2_sentences_output)
    #write_files( OUT_DIR + 'decoded/ms.2task.decoded.txt', ms_task2_sentences_output)
    #write_files( OUT_DIR + 'decoded/cls.2+task.50k.decoded.txt', task2_plus_sentences_output)
    #write_files( OUT_DIR + 'decoded/cls.3task.decoded.txt', task3_sentences_output)
    #write_files( OUT_DIR + 'decoded/ms.pipeline.decoded.txt', ms_pipeline_sentences_output)
    #write_files( OUT_DIR + 'decoded/cls.pipeline.decoded.txt', pipeline_sentences_output)
    #write_files( OUT_DIR + 'decoded/cls.first20.decoded.txt', task_first20)

    #write_files( OUT_DIR + 'decoded/ms.2task.decoded.txt', task2_ms_sentences_output)
    #write_files( OUT_DIR + 'decoded/ms.3task.decoded.txt', task3_ms_sentences_output)
    #write_files( OUT_DIR + 'decoded/ms.ref.decoded.txt', ms_sentences_ref)


    #print("Rouge scores for first20 model...........")
    #liPaperEvaluation([[sentence] for sentence in task_first20], [[[sentence]] for sentence in sentences_ref] )
    #print("Rouge scores for pipeline model...........")
    #liPaperEvaluation([[sentence] for sentence in pipeline_sentences_output], [[[sentence]] for sentence in sentences_ref] )
    #print("Rouge scores for supervised model...........")
    #liPaperEvaluation([[sentence] for sentence in base_sentences_output], [[[sentence]] for sentence in sentences_ref] )
    #print("Rouge scores for CLS+MS model...........")
    #liPaperEvaluation([[sentence] for sentence in task2_sentences_output], [[[sentence]] for sentence in sentences_ref] )
    #print("Rouge scores for CLS+MT model...........")
    #liPaperEvaluation([[sentence] for sentence in task2_plus_sentences_output], [[[sentence]] for sentence in sentences_ref] )
    #print("Rouge scores for CLS+MS+MT model...........")
    #liPaperEvaluation([[sentence] for sentence in task3_sentences_output], [[[sentence]] for sentence in sentences_ref] )

    # for MS
    #print("Rouge scores for CLS+MS model...........")
    #liPaperEvaluation([[sentence] for sentence in task2_ms_sentences_output], [[[sentence]] for sentence in ms_sentences_ref] )
    #print("Rouge scores for CLS+MS+MT model...........")
    #liPaperEvaluation([[sentence] for sentence in task3_ms_sentences_output], [[[sentence]] for sentence in ms_sentences_ref] )

    print("Bleu scores for CLS.......")
    print(calculate_bleu(base_sentences_output, sentences_ref))
    print("Bleu scores for MS+CLS.......")
    print(calculate_bleu(task2_sentences_output, sentences_ref))
    print("Bleu scores for MT+CLS.......")
    print(calculate_bleu(task2_plus_sentences_output, sentences_ref))
    print("Bleu scores for CLS+MS+MT.......")
    print(calculate_bleu(task3_sentences_output, sentences_ref))
    print("Bleu scores for CLS+MS->CLS+MT.......")
    print(calculate_bleu(task2_2p_3task_sentences_output, sentences_ref))
    print("Bleu scores for CLS+MT->CLS+MS.......")
    print(calculate_bleu(task2p_2_3task_sentences_output, sentences_ref))






