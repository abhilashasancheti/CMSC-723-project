#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rouge import Rouge 
import sys

def calculateRouge(argv):
    
    file2 = open(argv[1], 'r', encoding='UTF-8')
    rouge = Rouge()

    rouge1 = 0
    rouge2 = 0
    rougel = 0

    i = 0

    with open(argv[0], 'r', encoding='UTF-8') as file1:
        for line in file1:
            hypothesis = line.lower()
            reference = file2.readline()
            n = reference.count(reference[0])
            reference = reference.replace(reference[0], '', n)
            #print(hypothesis)
            #print(reference)   
            scores = rouge.get_scores(hypothesis, reference)
            rouge1 = rouge1 + scores[0]['rouge-1']['f']
            rouge2 = rouge2 + scores[0]['rouge-2']['f']
            rougel = rougel + scores[0]['rouge-l']['f']
            #print(rouge1, rouge2, rougel)
            i = i + 1
            #if i == 4:
            #    break

    print('Number of Lines read:', i)
    print('Rouge-1 :', rouge1/i, 'Rouge-2:', rouge2/i, 'Rouge-l:', rougel/i)

if __name__ == '__main__':
    calculateRouge(sys.argv)





