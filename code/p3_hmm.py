########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

import os
import pickle
from HMM import unsupervised_HMM
from HMM_helper import (
    parse_observations,
    sample_sentence,
    visualize_sparsities,
)
from Utility import Utility



    # Print the transition matrix.


if __name__ == '__main__':
    train = False
    n_states = 10
    N_iters = 50
    text = open(os.path.join(os.getcwd(), '../data/shakespeare.txt')).read()
    obs, obs_map, stress_dic = parse_observations(text)
    #print(obs)
    # Train the HMM.
    if train:
        HMM = unsupervised_HMM(obs, n_states, N_iters)
        file = open('hmm_10.txt','wb')
        pickle.dump(HMM, file)
        file.close()
    else:
        file = open("hmm_10.txt", "rb")
        HMM = pickle.load(file)

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
    #print(syl_dic)
    for j in range(20):
        for i in range(12):
            print(sample_sentence(HMM, obs_map, syl_dic, stress_dic))
        for i in range(2):
            print('  ' + sample_sentence(HMM, obs_map, syl_dic, stress_dic))
        print('')


