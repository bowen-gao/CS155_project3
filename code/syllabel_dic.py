import numpy as np
import os
'''
transform Syllable_dictionary.txt to a dictionary
'''
dic = open(os.path.join(os.getcwd(), '../data/Syllable_dictionary.txt')).read()
lines = [line.split() for line in dic.split('\n') if line.split()]

syl_dic = {}

for line in lines:
    normal_syl = []
    end_syl = []
    for i, word in enumerate(line):

        if i == 0:
            continue

        if word[0] == 'E':
            end_syl.append(int(word[1:]))
        else:
            normal_syl.append(int(word))

    if len(end_syl) == 0:
            end_syl = normal_syl
    syl_dic[line[0]] = [normal_syl, end_syl]




