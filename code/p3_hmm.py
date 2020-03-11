########################################
# CS/CNS/EE 155 2017
# Problem Set 5
#
# Author:       Andrew Kang
# Description:  Set 5
########################################

import os
from Utility import Utility
from HMM import unsupervised_HMM
from HMM_helper import (
    parse_observations,
    sample_sentence,
    visualize_sparsities,
)

def unsupervised_learning(n_states, N_iters):
    '''
    Trains an HMM using supervised learning on the file 'ron.txt' and
    prints the results.

    Arguments:
        n_states:   Number of hidden states that the HMM should have.
    '''
    text = open(os.path.join(os.getcwd(), '../data/shakespeare.txt')).read()
    print(text)
    obs, obs_map = parse_observations(text)
    print(obs)
    # Train the HMM.
    HMM = unsupervised_HMM(obs, n_states, N_iters)
    print(sample_sentence(HMM, obs_map, n_words=25))

    # Print the transition matrix.


if __name__ == '__main__':


    unsupervised_learning(4, 100)
