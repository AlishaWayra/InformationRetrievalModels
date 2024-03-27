
import pandas as pd
import numpy as np

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import operator
import csv
import math

#%% 1. text processing helper functions 

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

def get_stopwords():
    """ combine and preprocess stopwords of two packages """
    stopw_nltk = stopwords.words('english')                                                 # nltk stop words
    stopw_sc = list(ENGLISH_STOP_WORDS)                                                     # sklearn stop words
    stopw_sc.remove('even')
    stopw = set(stopw_nltk + stopw_sc)
    
    punc = set(string.punctuation)                                                          # remove punctuations from stopwords (e.g "should've" -> "shouldve" )
    stopw_nopunc = [''.join([p for p in word if p not in punc]) for word in stopw]

    stemmer= PorterStemmer()                                                                # stemming
    stopw_processed = set([stemmer.stem(word.lower()) for word in stopw_nopunc])
    
    return stopw_processed
    
def remove_stopw(text, stopwordlist):
    """ only keep tokens that are not in list of stopwords """
    tokens = [ token for token in text if token not in stopwordlist ]

    return tokens

#%% 2. tokenizer helper functions 

def tokenizer(passages, stopwordlist):
    """ tokenize all potential passages for one query """
    passages_tokenized = [ remove_stopw(preprocessing(passage), stopwordlist) for passage in passages ]
    
    return passages_tokenized

def create_tokens(chunk, qid, query, stopwordlist):
    """ tokenize all potential passages for one query and query as well"""
    psg_tokens = tokenizer(chunk['passage'].tolist(), stopwordlist)                            # tokenize relevant passages
    qry_tokens = tokenizer([query], stopwordlist)                                              # tokenize query
  
    return psg_tokens, qry_tokens

#%% 3. Query likelihood language model - Lacplace Smoothing

def laplace_estimate(word, pid, psg_tokens_i, inv_idx):
    """
    estimating p(qi|D) = probability that query word qi is generated from Document language model 
    """
    V = len(inv_idx)                                                                            # nr of unique words in corpus
    D = len(psg_tokens_i)                                                                       # document length
    
    if word in inv_idx and pid in inv_idx[word].keys():
        fi = inv_idx[word][pid]
        estimate = (fi + 1) / (D + V)
    elif word in inv_idx:
        fi = 0
        estimate = (fi + 1) / (D + V)
    else:
        fi = 0
        estimate = (fi + 1) / (D + V + 1) # 1 added to denominator to account for the fact that the query word is not in the vocabulary
    
    return estimate

def laplace_score(qry_tokens, pid, psg_tokens_i, inv_idx):
    """
    estimating p(q|D) = probability that whole query is generated from Document langauge model
    
        -> using naive bayes: p(q|D) = p(q1|D) * p(q2|D) * ... * p(qn|D)        n = number of words in query
    """
    score = 1
    
    for word in qry_tokens:
        score *= laplace_estimate(word, pid, psg_tokens_i, inv_idx)

    return math.log(score)     

def laplace_probabilities(chunk, qid, qry_tokens, psg_tokens, inv_idx):
    """
    estimating p(q|D) pairwise for one query and top 1000 passages and rank them
    """
    
    scores = []
    for pid, psg_tokens_i in zip(chunk['pid'], psg_tokens):
        score = laplace_score(qry_tokens, pid, psg_tokens_i, inv_idx)
        scores.append([qid, pid, score])
    
    scores_sorted = sorted(scores, key=operator.itemgetter(2), reverse=True)
    
    return scores_sorted

def get_top100(scores):
    """
    Extracting 100 best passages for one query
    """
    return scores[:min(100, len(scores))]

def create_csvfile(OUTPUTFILE, output):
    """ write csv file of top 100 passages """
    with open(OUTPUTFILE, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerows(output)                                                             
        
    del output
    
#%% 4. Query likelihood language model - Lidstone Correction

def lidstone_estimate(word, pid, psg_tokens_i, epsilon, inv_idx):
    """
    estimating p(qi|D) = probability that query word qi is generated from Document language model 
    """
    V = len(inv_idx)                                                                            
    D = len(psg_tokens_i)
    
    if word in inv_idx and pid in inv_idx[word].keys():
        fi = inv_idx[word][pid]
        estimate = (fi + epsilon) / (D + V*epsilon)
    elif word in inv_idx:
        fi = 0
        estimate = (fi + epsilon) / (D + V*epsilon)
    else:
        fi = 0
        estimate = (fi + epsilon) / (D + V*epsilon + epsilon)
    
    return estimate

def listone_score(qry_tokens, pid, psg_tokens_i, inv_idx, epsilon):
    """
    estimating p(q|D) = probability that whole query is generated from Document langauge model
    
        -> using naive bayes: p(q|D) = p(q1|D) * p(q2|D) * ... * p(qn|D)        n = number of words in query
    """
    
    score = 1
    
    for word in qry_tokens:
        score *= lidstone_estimate(word, pid, psg_tokens_i, epsilon, inv_idx)

    return math.log(score)     

def lidstone_probabilities(chunk, qid, qry_tokens, psg_tokens, inv_idx, epsilon):
    """
    estimating p(q|D) pairwise for one query and top 1000 passages and rank them
    """
    
    scores = []
    for pid, psg_tokens_i in zip(chunk['pid'], psg_tokens):
        score = listone_score(qry_tokens, pid, psg_tokens_i, inv_idx, epsilon)
        scores.append([qid, pid, score])
    
    scores_sorted = sorted(scores, key=operator.itemgetter(2), reverse=True)
    
    return scores_sorted

#%% 5. Query likelihood language model - Dirichtlet smoothing

def dirichtlet_estimate(word, pid, psg_tokens_i, mu, inv_idx, voca_freq):
    """
    estimating p(qi|D) = probability that query word qi is generated from Document language model 
    """
    
    N = len(psg_tokens_i)
    L = sum(vocab_freq.values())
    
    if word in inv_idx and pid in inv_idx[word].keys():                             # define word frequency in document
        fi = inv_idx[word][pid]
    else:
        fi = 0
    
    if word in inv_idx:                                                             # define word frequency in whole corpus
        ci = vocab_freq[word]
    else:
        ci = 1                                                                      # in case word is not in vocab
    
    return (N / (N+mu)) * (fi / N) + (mu / (N+mu)) * (ci/L)

def dirichtlet_score(qry_tokens, pid, psg_tokens_i, inv_idx, vocab_freq, mu):
    """
    estimating p(q|D) = probability that whole query is generated from Document langauge model
    
        -> using naive bayes: p(q|D) = p(q1|D) * p(q2|D) * ... * p(qn|D)        n = number of words in query
    """
    
    score = 1
    
    for word in qry_tokens:
        score *= dirichtlet_estimate(word, pid, psg_tokens_i, mu, inv_idx, vocab_freq)

    return math.log(score)     

def dirichlet_probabilities(chunk, qid, qry_tokens, psg_tokens, inv_idx, vocab_freq, mu):
    """
    estimating p(q|D) pairwise for one query and top 1000 passages and rank them
    """
    
    scores = []
    for pid, psg_tokens_i in zip(chunk['pid'], psg_tokens):
        score = dirichtlet_score(qry_tokens, pid, psg_tokens_i, inv_idx, vocab_freq, mu)
        scores.append([qid, pid, score])
    
    scores_sorted = sorted(scores, key=operator.itemgetter(2), reverse=True)
    
    return scores_sorted

#%% 6. Extracting and ranking passages

if( __name__ == "__main__" ):
    # import data files and output from previous scripts
    PASSAGEFILE = 'candidate-passages-top1000.tsv'
    QUERYFILE = 'test-queries.tsv'
    INVIDXFILE = 'inv_idx.npy'
    VOCABFREQFILE = 'vocab_freq.npy'
    OUTPUTFILE_LAPLACE = 'laplace.csv'
    OUTPUTFILE_LIDSTONE = 'lidstone.csv'
    OUTPUTFILE_DIRICHLET = 'dirichlet.csv'
    
    df_psg = pd.read_csv(PASSAGEFILE, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    df_qry = pd.read_csv(QUERYFILE, sep='\t', header=None, names=['qid', 'query'])
    inv_idx = np.load(INVIDXFILE, allow_pickle='TRUE').item()
    vocab_freq = np.load(VOCABFREQFILE, allow_pickle='TRUE').item()
    
    # retrieve at most 100 passages for each query
    stopwordlist = get_stopwords()
    laplace_results = []
    lidstone_results = []
    dirichlet_results = []
    
    for i, (qid, query) in enumerate( zip(df_qry['qid'], df_qry['query']) ):
        total_qry = df_qry.shape[0]
        print(f'Tokenizing texts and estimating query likelihoods for query {qid} ({i+1}/{total_qry})')
        print(f'    Query: {query}')
        
        ### tokenizing query and passages ###
        chunk = df_psg.groupby('qid').get_group(qid)                                # reduce dataframe to relevant passages for query
        psg_tokens, qry_tokens = create_tokens(chunk, qid, query, stopwordlist)
        
        ### Laplace probability results ###
        laplace_scores = laplace_probabilities(chunk, qid, qry_tokens[0], psg_tokens, inv_idx)
        laplace_top100 = get_top100(laplace_scores)
        laplace_results = laplace_results + laplace_top100                          # append top 100 to results list
        
        ### Lidstone probability results ###
        lidstone_scores = lidstone_probabilities(chunk, qid, qry_tokens[0], psg_tokens, inv_idx, epsilon=0.1)
        lidstone_top100 = get_top100(lidstone_scores)
        lidstone_results = lidstone_results + lidstone_top100
        
        ### Dirichlet probability results ###
        dirichlet_scores = dirichlet_probabilities(chunk, qid, qry_tokens[0], psg_tokens, inv_idx, vocab_freq, mu=50)
        dirichlet_top100 = get_top100(dirichlet_scores)
        dirichlet_results = dirichlet_results + dirichlet_top100
        
    create_csvfile(OUTPUTFILE_LAPLACE, laplace_results)
    print('successfully saved laplace results to csv!')
    create_csvfile(OUTPUTFILE_LIDSTONE, lidstone_results)
    print('successfully saved lidstone results to csv!')
    create_csvfile(OUTPUTFILE_DIRICHLET, dirichlet_results)
    print('successfully saved dirichlet results to csv!')
    
    
