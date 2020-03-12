########################################
# CS/CNS/EE 155 2018
# Problem Set 6
#
# Author:       Andrew Kang
# Description:  Set 6 skeleton code
########################################

# You can use this (optional) skeleton code to complete the HMM
# implementation of set 5. Once each part is implemented, you can simply
# execute the related problem scripts (e.g. run 'python 2G.py') to quickly
# see the results from your code.
#
# Some pointers to get you started:
#
#     - Choose your notation carefully and consistently! Readable
#       notation will make all the difference in the time it takes you
#       to implement this class, as well as how difficult it is to debug.
#
#     - Read the documentation in this file! Make sure you know what
#       is expected from each function and what each variable is.
#
#     - Any reference to "the (i, j)^th" element of a matrix T means that
#       you should use T[i][j].
#
#     - Note that in our solution code, no NumPy was used. That is, there
#       are no fancy tricks here, just basic coding. If you understand HMMs
#       to a thorough extent, the rest of this implementation should come
#       naturally. However, if you'd like to use NumPy, feel free to.
#
#     - Take one step at a time! Move onto the next algorithm to implement
#       only if you're absolutely sure that all previous algorithms are
#       correct. We are providing you waypoints for this reason.
#
# To get started, just fill in code where indicated. Best of luck!

import random


class HiddenMarkovModel:
    '''
    Class implementation of Hidden Markov Models.
    '''

    def __init__(self, A, O):
        '''
        Initializes an HMM. Assumes the following:
            - States and observations are integers starting from 0. 
            - There is a start state (see notes on A_start below). There
              is no integer associated with the start state, only
              probabilities in the vector A_start.
            - There is no end state.

        Arguments:
            A:          Transition matrix with dimensions L x L.
                        The (i, j)^th element is the probability of
                        transitioning from state i to state j. Note that
                        this does not include the starting probabilities.

            O:          Observation matrix with dimensions L x D.
                        The (i, j)^th element is the probability of
                        emitting observation j given state i.

        Parameters:
            L:          Number of states.
            
            D:          Number of observations.
            
            A:          The transition matrix.
            
            O:          The observation matrix.
            
            A_start:    Starting transition probabilities. The i^th element
                        is the probability of transitioning from the start
                        state to state i. For simplicity, we assume that
                        this distribution is uniform.
        '''

        self.L = len(A)
        self.D = len(O[0])
        self.A = A
        self.O = O
        self.A_start = [1. / self.L for _ in range(self.L)]

    def viterbi(self, x):
        '''
        Uses the Viterbi algorithm to find the max probability state 
        sequence corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            max_seq:    State sequence corresponding to x with the highest
                        probability.
        '''

        M = len(x)  # Length of sequence.
        # The (i, j)^th elements of probs and seqs are the max probability
        # of the prefix of length i ending in state j and the prefix
        # that gives this probability, respectively.
        #
        # For instance, probs[1][0] is the probability of the prefix of
        # length 1 ending in state 0.
        probs = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        seqs = [['' for _ in range(self.L)] for _ in range(M + 1)]

        for state in range(self.L):
            probs[1][state] = self.A_start[state] * self.O[state][x[0]]
        for t in range(2, M + 1):
            for cur_state in range(self.L):
                max_prob = 0
                max_prev = 0
                for prev_state in range(self.L):
                    cur_prob = probs[t - 1][prev_state] * self.A[prev_state][cur_state] * self.O[cur_state][x[t - 1]]
                    if cur_prob > max_prob:
                        max_prob = cur_prob
                        max_prev = prev_state
                probs[t][cur_state] = max_prob
                seqs[t][cur_state] = max_prev
        max_prob = 0
        last_state = 0
        max_seq = ''
        for state in range(self.L):
            cur_prob = probs[-1][state]
            if cur_prob > max_prob:
                last_state = state
                max_prob = cur_prob
        cur_state = last_state
        for i in range(1, M + 1):
            prev_state = seqs[M + 1 - i][cur_state]
            max_seq = str(cur_state) + max_seq
            cur_state = prev_state
        return max_seq

    def forward(self, x, normalize=False):
        '''
        Uses the forward algorithm to calculate the alpha probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            alphas:     Vector of alphas.

                        The (i, j)^th element of alphas is alpha_j(i),
                        i.e. the probability of observing prefix x^1:i
                        and state y^i = j.

                        e.g. alphas[1][0] corresponds to the probability
                        of observing x^1:1, i.e. the first observation,
                        given that y^1 = 0, i.e. the first state is 0.
        '''

        M = len(x)  # Length of sequence.
        alphas = [[0. for _ in range(self.L)] for _ in range(M + 1)]
        total = 0
        for state in range(self.L):
            alphas[1][state] = self.A_start[state] * self.O[state][x[0]]
            total += alphas[1][state]
        if normalize:
            for state in range(self.L):
                alphas[1][state] /= total
        for t in range(2, M + 1):
            total = 0
            for cur_state in range(self.L):
                alphas[t][cur_state] = self.O[cur_state][x[t - 1]]
                prob = 0
                for prev_state in range(self.L):
                    prob += self.A[prev_state][cur_state] * alphas[t - 1][prev_state]
                alphas[t][cur_state] *= prob
                total += alphas[t][cur_state]
            if normalize:
                for cur_state in range(self.L):
                    alphas[t][cur_state] /= total
        return alphas

    def backward(self, x, normalize=False):
        '''
        Uses the backward algorithm to calculate the beta probability
        vectors corresponding to a given input sequence.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

            normalize:  Whether to normalize each set of alpha_j(i) vectors
                        at each i. This is useful to avoid underflow in
                        unsupervised learning.

        Returns:
            betas:      Vector of betas.

                        The (i, j)^th element of betas is beta_j(i), i.e.
                        the probability of observing prefix x^(i+1):M and
                        state y^i = j.

                        e.g. betas[M][0] corresponds to the probability
                        of observing x^M+1:M, i.e. no observations,
                        given that y^M = 0, i.e. the last state is 0.
        '''

        M = len(x)  # Length of sequence.
        betas = [[0. for _ in range(self.L)] for _ in range(M + 1)]

        total = 0
        for state in range(self.L):
            betas[-1][state] = 1
            total += betas[-1][state]
        if normalize:
            for state in range(self.L):
                betas[-1][state] /= total
        for t in range(M - 1, 0, -1):
            total = 0
            for cur_state in range(self.L):
                for next_state in range(self.L):
                    betas[t][cur_state] += self.A[cur_state][next_state] * betas[t + 1][next_state] * \
                                           self.O[next_state][x[t]]
                total += betas[t][cur_state]
            if normalize:
                for cur_state in range(self.L):
                    betas[t][cur_state] /= total

        return betas

    def supervised_learning(self, X, Y):
        '''
        Trains the HMM using the Maximum Likelihood closed form solutions
        for the transition and observation matrices on a labeled
        datset (X, Y). Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to D - 1. In other words, a list of
                        lists.

            Y:          A dataset consisting of state sequences in the form
                        of lists of variable length, consisting of integers 
                        ranging from 0 to L - 1. In other words, a list of
                        lists.

                        Note that the elements in X line up with those in Y.
        '''

        # Calculate each element of A using the M-step formulas.

        for prev_state in range(self.L):
            for next_state in range(self.L):
                nume = 0
                deno = 0
                for i in range(len(X)):
                    obs = X[i]
                    sts = Y[i]
                    for t in range(len(obs) - 1):
                        if sts[t] == prev_state:
                            deno += 1
                            if sts[t + 1] == next_state:
                                nume += 1
                self.A[prev_state][next_state] = float(nume) / deno

        # Calculate each element of O using the M-step formulas.

        for state in range(self.L):
            for ob in range(self.D):
                nume = 0
                deno = 0
                for i in range(len(X)):
                    obs = X[i]
                    sts = Y[i]
                    for t in range(len(obs)):
                        if sts[t] == state:
                            deno += 1
                            if obs[t] == ob:
                                nume += 1
                self.O[state][ob] = float(nume) / deno

    def unsupervised_learning(self, X, N_iters):
        '''
        Trains the HMM using the Baum-Welch algorithm on an unlabeled
        datset X. Note that this method does not return anything, but
        instead updates the attributes of the HMM object.

        Arguments:
            X:          A dataset consisting of input sequences in the form
                        of lists of length M, consisting of integers ranging
                        from 0 to D - 1. In other words, a list of lists.

            N_iters:    The number of iterations to train on.
        '''

        for iter in range(N_iters):
            print(iter)
            A_nume = [[0. for _ in range(self.L)] for _ in range(self.L)]
            A_deno = [[0. for _ in range(self.L)] for _ in range(self.L)]
            O_nume = [[0. for _ in range(self.D)] for _ in range(self.L)]
            O_deno = [[0. for _ in range(self.D)] for _ in range(self.L)]
            for obs in X:
                alphas = self.forward(obs, normalize=True)
                betas = self.backward(obs, normalize=True)
                for t in range(len(obs) - 1):
                    deno = 0.0
                    norm = sum([a * b for a, b in zip(alphas[t + 1], betas[t + 1])])
                    for state1 in range(self.L):
                        for state2 in range(self.L):
                            deno += alphas[t + 1][state1] * self.A[state1][state2] * \
                                    self.O[state2][obs[t + 1]] * \
                                    betas[t + 2][state2]
                    for prev_state in range(self.L):
                        for next_state in range(self.L):
                            A_deno[prev_state][next_state] += \
                                alphas[t + 1][prev_state] * \
                                betas[t + 1][prev_state] / norm
                            nume = alphas[t + 1][prev_state] * \
                                   self.A[prev_state][next_state] * \
                                   self.O[next_state][obs[t + 1]] * \
                                   betas[t + 2][next_state]
                            A_nume[prev_state][next_state] += nume / deno
                for t in range(len(obs)):
                    norm = sum([a * b for a, b in zip(alphas[t + 1], betas[t + 1])])
                    for state in range(self.L):
                        for ob in range(self.D):
                            O_deno[state][ob] += alphas[t + 1][state] * \
                                                 betas[t + 1][state] / norm
                            if ob == obs[t]:
                                O_nume[state][ob] += alphas[t + 1][state] * \
                                                     betas[t + 1][state] / norm
            for prev_state in range(self.L):
                for next_state in range(self.L):
                    self.A[prev_state][next_state] = A_nume[prev_state][next_state] / \
                                                     A_deno[prev_state][next_state]
            for state in range(self.L):
                for ob in range(self.D):
                    self.O[state][ob] = O_nume[state][ob] / O_deno[state][ob]

    def dfs(self, word1, word2, stress_dic, visited):
        print(word1, word2)
        if word1 in visited:
            return False
        if word2 in stress_dic[word1]:
            return True
        visited.add(word1)
        for neighbor in stress_dic[word1]:
            if self.dfs(neighbor, word2, stress_dic, visited):
                return True
        return False


    def recur(self, curr_emission, count, syl_dic, cur_state, obs_map_r, emission, state, stress_dic):
        if count > 10:
            return False, None, None
        word = obs_map_r[curr_emission]
        normal_syl, end_syl = syl_dic[word]
        for syl in end_syl:
            if count + syl == 10:
                emission.append(curr_emission)
                state.append(cur_state)
                #print(count + syl)
                return True, emission[:], state[:]
        for syl in normal_syl:
            emission.append(curr_emission)
            state.append(cur_state)
            pre_state = cur_state
            while True:
                s = set()
                next_state = random.choices([i for i in range(self.L)], self.A[pre_state])[0]
                next_emission = random.choices([i for i in range(self.D)], self.O[cur_state])[0]
                if self.dfs(obs_map_r[curr_emission], obs_map_r[next_emission], stress_dic, s):
                    break
            if count + syl < 10:
                flag, res_emission, res_state = self.recur(next_emission, count+syl, syl_dic, next_state, obs_map_r, emission, state, stress_dic)
                if flag:
                    return True, res_emission[:], res_state[:]
                else:
                    emission.pop()
                    state.pop()
            else:
                emission.pop()
                state.pop()
                return False, None, None
        return False, None, None



    #recur(curr_emission, 0, syl_dic, cur_state, obs_map_r, [], [])

    def generate_emission(self, obs_map_r, syl_dic, stress_dic):
        '''
        Generates an emission of length M, assuming that the starting state
        is chosen uniformly at random. 

        Arguments:
            M:          Length of the emission to generate.

        Returns:
            emission:   The randomly generated emission as a list.

            states:     The randomly generated states as a list.
        '''

        ter_flag = 1
        count = 0
        num = 10
        emission = []
        states = []

        while True:
            state_0 = random.choices([i for i in range(self.L)], self.A_start)[0]
            emission_0 = random.choices([i for i in range(self.D)], self.O[state_0])[0]
            flag, emission_result, states_result = self.recur(emission_0, count, syl_dic, state_0, obs_map_r, emission, states, stress_dic)
            if flag:
                break


        return emission_result, states_result

    def probability_alphas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the forward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        # Calculate alpha vectors.
        alphas = self.forward(x)

        # alpha_j(M) gives the probability that the state sequence ends
        # in j. Summing this value over all possible states j gives the
        # total probability of x paired with any state sequence, i.e.
        # the probability of x.
        prob = sum(alphas[-1])
        return prob

    def probability_betas(self, x):
        '''
        Finds the maximum probability of a given input sequence using
        the backward algorithm.

        Arguments:
            x:          Input sequence in the form of a list of length M,
                        consisting of integers ranging from 0 to D - 1.

        Returns:
            prob:       Total probability that x can occur.
        '''

        betas = self.backward(x)

        # beta_j(1) gives the probability that the state sequence starts
        # with j. Summing this, multiplied by the starting transition
        # probability and the observation probability, over all states
        # gives the total probability of x paired with any state
        # sequence, i.e. the probability of x.
        prob = sum([betas[1][j] * self.A_start[j] * self.O[j][x[0]] \
                    for j in range(self.L)])

        return prob


def supervised_HMM(X, Y):
    '''
    Helper function to train a supervised HMM. The function determines the
    number of unique states and observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for supervised learning.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        Y:          A dataset consisting of state sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to L - 1. In other words, a list of lists.
                    Note that the elements in X line up with those in Y.
    '''
    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Make a set of states.
    states = set()
    for y in Y:
        states |= set(y)

    # Compute L and D.
    L = len(states)
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with labeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.supervised_learning(X, Y)

    return HMM


def unsupervised_HMM(X, n_states, N_iters):
    '''
    Helper function to train an unsupervised HMM. The function determines the
    number of unique observations in the given data, initializes
    the transition and observation matrices, creates the HMM, and then runs
    the training function for unsupervised learing.

    Arguments:
        X:          A dataset consisting of input sequences in the form
                    of lists of variable length, consisting of integers 
                    ranging from 0 to D - 1. In other words, a list of lists.

        n_states:   Number of hidden states to use in training.
        
        N_iters:    The number of iterations to train on.
    '''

    # Make a set of observations.
    observations = set()
    for x in X:
        observations |= set(x)

    # Compute L and D.
    L = n_states
    D = len(observations)

    # Randomly initialize and normalize matrix A.
    random.seed(2020)
    A = [[random.random() for i in range(L)] for j in range(L)]

    for i in range(len(A)):
        norm = sum(A[i])
        for j in range(len(A[i])):
            A[i][j] /= norm

    # Randomly initialize and normalize matrix O.
    random.seed(155)
    O = [[random.random() for i in range(D)] for j in range(L)]

    for i in range(len(O)):
        norm = sum(O[i])
        for j in range(len(O[i])):
            O[i][j] /= norm

    # Train an HMM with unlabeled data.
    HMM = HiddenMarkovModel(A, O)
    HMM.unsupervised_learning(X, N_iters)

    return HMM
