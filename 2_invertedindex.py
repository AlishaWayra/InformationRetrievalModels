import pandas as pd
import numpy as np
from collections import Counter

from nltk.tokenize import word_tokenize
import string
from nltk.stem import PorterStemmer

#%% 1. preprocess text data

def preprocessing(text): 
    
    # 1. remove punctuation
    punc = set(string.punctuation)
    text_nopunc = ''.join([p for p in text if p not in punc])
    
    # 2. all words to lower case
    text_lowcase = text_nopunc.lower()
    
    # 3. tokenize whole text/string into single tokens/words
    tokens = word_tokenize(text_lowcase) # a list of strings now
    
    # initialize WordNet
    stemmer= PorterStemmer()
    
    # 4. stemming
    tokens_stemmed = [ stemmer.stem(token) for token in tokens ]
    
    return tokens_stemmed

#%% 2. create inverted index from imported data

def create_inv_idx(data, vocab): 
    """ 
    takes dataframe and vocabulary list as argumetns, creates dictionary of passages & pids and returns inverted index as dictionary
    """
    
    # map each passage to unique pid in dictioanry
    psg_pid = dict(list(zip(data.passage, data.pid)))   # list of tuples (passage, pid) transformed into dictionary with passages as keys and pids as values

    # inv. idx with unique words as keys and empty nested list as values
    inv_idx = {token: {} for token in vocab}
    
    # for each passage and corresponding pid  
    for passage, pid in psg_pid.items():
        # tokenize passage into single words and preprocess
        words = preprocessing(passage)
        
        # count number of times each word occurs in passage
        count = Counter(words)
        
        for word in count: # for each unique word in passage
            try:
                # append pid and word count if word in passage is part of vocabulary / dictionary
                inv_idx[word][pid] = count[word]
            except:
                # if word not part of vocabulary / dictionary, continue with loop
                pass
    
    return inv_idx

#%% run script

if( __name__ == "__main__" ):
    # import query-passage data and voabulary list
    VOCAB = 'vocab_freq.npy'                            # file name of vocabualary list
    FILE = 'candidate-passages-top1000.tsv'             # file name with query and passage text
    vocab = list( np.load(VOCAB, allow_pickle='TRUE').item().keys() )
    data = pd.read_csv(FILE, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    data = data.drop_duplicates(subset=['pid'])         # keep only unique passages (pid as identifier)
    
    # create inverted index
    inv_idx = create_inv_idx(data, vocab)
    np.save('inv_idx.npy', inv_idx)
    print('created and saved inverted index successfully!')
    
