
import os
import pickle
import os
import numpy as np
from IPython.display import HTML

from HMM import unsupervised_HMM
from HMM_helper import (
    text_to_wordcloud,
    states_to_wordclouds,
    parse_observations,
    sample_sentence,
    visualize_sparsities,
    animate_emission
)

#######
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
#########

file = open('hmm.txt', "rb")
HMM = pickle.load(file)

visualize_sparsities(HMM, O_max_cols=50)

text = open(os.path.join(os.getcwd(), '../data/shakespeare.txt')).read()
wordcloud = text_to_wordcloud(text, title='shakespeare')
obs, obs_map, stress_dic = parse_observations(text)
wordclouds = states_to_wordclouds(HMM, obs_map, syl_dic, stress_dic)


