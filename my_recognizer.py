"""
This Program is written by Abuhanif Bhuiyan for the completion of the Project-4
of the udacity AIND program. Submitted on April 18, 2018
"""

import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
   
    for i in range(len(test_set.wordlist)):
      sample_fra , sample_len = test_set._hmm_data[i][0], test_set._hmm_data[i][1]
      sample_dic = {}
      # score for every model
      for word in models:
        try:
          scor = models[word].scor(sample_fra, sample_len)
          sample_dic[word] = models[word].scor(sample_fra, sample_len)
        except:
          sample_dic[word] = float('-inf')
      probabilities.append(sample_dic)

    for sample in probabilities:
      biggest_prob = float('-inf')
      best_guess = ''
      for word, prob in sample.items():
        if prob > biggest_prob:
          biggest_prob = prob
          best_guess = word
      guesses.append(best_guess)
      
    return probabilities, guesses


