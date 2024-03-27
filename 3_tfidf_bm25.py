import pandas as pd
import numpy as np

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import scipy.sparse as sp
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
import csv
import operator
from statistics import mean

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

#%% 2. tfidf helper functions 

def tokenizer(passages, stopwordlist):
    """ tokenize all potential passages for one query """
    passages_tokenized = [ remove_stopw(preprocessing(passage), stopwordlist) for passage in passages ]
    
    return passages_tokenized

def create_tokens(chunk, qid, query, stopwordlist):
    """ tokenize all potential passages for one query and query as well"""
    psg_tokens = tokenizer(chunk['passage'].tolist(), stopwordlist)                            # tokenize relevant passages
    qry_tokens = tokenizer([query], stopwordlist)                                              # tokenize query
  
    return psg_tokens, qry_tokens

def calc_idf(df, inv_idx):
    """ calculate idf for each word as dictionary """
    idf = {}
    N = len(df['pid'].unique())                                                             # number of total passages
    for word in inv_idx:
        n = len( inv_idx[word] )                                                            # number of passages a given word occurs in 
        idf[word] = np.log( N / n)  
    
    return idf

def calc_tfidf_vector(text_tknized, idf, inv_idx, ID, passage=True):
    """ represents tokenized text as tfidf vector """
    
    tfidf_vec = np.zeros((len(inv_idx), 1))                                                 # tfidf vector with default 0 values 
    text_size = len(text_tknized)                                                           # number of words in text 
         
    for token in text_tknized:
        if token in inv_idx:
            row_idx = list(inv_idx.keys()).index(token)                                     # get row/token index in tfidf vector
            if passage == True:
                token_count = inv_idx[token][ID]                                            # nr. of times a token occurs in text in case of a passage
            else:
                token_count = Counter(text_tknized)[token]                                  # nr. of times a token occurs in text in case of a query
            token_idf = idf[token]                      
            token_tfidf = np.single( (token_count / text_size) * token_idf )                # tfidf score for token
            tfidf_vec[row_idx] = token_tfidf                                                # insert score into vector
            del row_idx, token_idf, token_tfidf                                             # delete variables to save memory

    return tfidf_vec

def create_tfidf_embeddings(chunk, qid, psg_tokens, qry_tokens):
    """ computes tfidf embeddings for all potential passages for one query and query as well """
    psg_embeddings = np.hstack([calc_tfidf_vector(psg_tokens_i, idf, inv_idx, ID=pid, passage=True) for psg_tokens_i, pid in zip(psg_tokens, chunk['pid'].tolist()) ])  # tfidf representation matrix of all relevant passages
    qry_embedding = calc_tfidf_vector(qry_tokens[0], idf, inv_idx, qid, passage=False)                                                                                  # tfidf representation vector of query
    
    return psg_embeddings, qry_embedding

#%% 3. Cosine similarity helper functions

def calc_cosine_sim(qry_embedding, psg_embeddings):    
    """ returns array with pairwise cosine similarity score between one query and all potential passages """
    return cosine_similarity(sp.csr_matrix(qry_embedding).T, sp.csr_matrix(psg_embeddings).T)

def get_top100(chunk, scores):
    """ extract 100 most relevant passages """
    pids_scores = dict(zip(chunk['pid'], scores[0].tolist()))                                 # map pids to respective cosine score
    pids_scores = ( sorted(pids_scores.items(), key=operator.itemgetter(1),reverse=True) )    # sort (pid, cosine) according to cosine score in descending order
    pids_scores_100 = [list(i) for i in pids_scores[:min(100, len(pids_scores))]]             # extract best 100 passages
    for l in pids_scores_100: 
        l.insert(0, qid)
        
    return pids_scores_100

def create_csvfile(OUTPUTFILE_TFIDF, output):
    """ write csv file of top 100 passages """
    with open(OUTPUTFILE_TFIDF, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',', quoting=csv.QUOTE_NONE)
        writer.writerows(output)                                                              # write several rows
        
    del output
    
#%% 4. BM25 helper functions

def calc_modifiedidf(df, word, inv_idx):
    """ calculate 1. part of the score """
    N = len(df['pid'].unique())                                                               # total number of documents in corpus
    ni = len( inv_idx[word] )                                                                 # number of documents containing word
    idf_word = np.log(  (N - ni + 0.5) / (ni + 0.5) )
    
    return idf_word

def calc_tf(word, k1, k2, b, count, pid, dl, inv_idx, psg_tokens):
    """ calculate 2. part of the score """
    qfi = count[word]                                                                           # how often word occurs in query
    fi = inv_idx[word][pid]                                                                     # how often word occurs in text
    avdl = mean(len(psg_tokens_i) for psg_tokens_i in psg_tokens)                               # average document length
    K = k1*( (1-b) + b*dl/avdl )
    tf_word = ( ( (k1 + 1) * fi) / (K+fi) ) * ( ((k2 + 1) * qfi) / (k2 + qfi) )
    
    return tf_word

def bm25(df, qid, qry_tokens, psg_tokens_i, inv_idx, pid, k1, k2, b):
    """  calculate score between one query and one document """
    count = Counter(qry_tokens[0])                                                              # how often words occur in query
    dl = len(psg_tokens_i)                                                                      # document length

    score = 0
    for word in qry_tokens[0]:
        if word in inv_idx and pid in inv_idx[word].keys():                                     # if passage contains the word, calculate score
            score += calc_modifiedidf(df, word, inv_idx) * calc_tf(word, k1, k2, b, count, pid, dl, inv_idx, psg_tokens)
        
        else:                                                                                   # otherwise score is 0
            score +=0
            
    return score

def bm25_similarity(df, qid, qry_tokens, psg_tokens, inv_idx, chunk, k1, k2, b):
    """ return array with pairwise bm25 score (between one query and all potential passages) """
    return  np.array([ bm25(df, qid, qry_tokens, psg_tokens_i, inv_idx, pid, k1, k2, b) for pid, psg_tokens_i in zip(chunk['pid'], psg_tokens)]).reshape(1, -1)

#%% 5. Extracting and ranking passages

if( __name__ == "__main__" ):
    # import data files and output from previous scripts
    PASSAGEFILE = 'candidate-passages-top1000.tsv'
    QUERYFILE = 'test-queries.tsv'
    INVIDXFILE = 'inv_idx.npy'
    OUTPUTFILE_TFIDF = 'cosine.csv'
    OUTPUTFILE_BM25 = 'bm25.csv'
    
    df_psg = pd.read_csv(PASSAGEFILE, sep='\t', header=None, names=['qid', 'pid', 'query', 'passage'])
    df_qry = pd.read_csv(QUERYFILE, sep='\t', header=None, names=['qid', 'query'])
    inv_idx = np.load(INVIDXFILE, allow_pickle='TRUE').item()

    # retrieve at most 100 passages for each query
    stopwordlist = get_stopwords()
    idf = calc_idf(df_psg, inv_idx)
    cosine_results = []
    bm25_results = []
    
    for i, (qid, query) in enumerate( zip(df_qry['qid'], df_qry['query']) ):
        total_qry = df_qry.shape[0]
        print(f'Creating embeddings and similarities for query {qid} ({i}/{total_qry})')
        print(f'    Query: {query}')
        
        ### tfidf embeddings ###
        chunk = df_psg.groupby('qid').get_group(qid)                      # reduce dataframe to relevant passages for query
        psg_tokens, qry_tokens = create_tokens(chunk, qid, query, stopwordlist)
        psg_embeddings, qry_embedding = create_tfidf_embeddings(chunk, qid, psg_tokens, qry_tokens)
        
        ### Cosine similarity results ###
        cosine_scores = calc_cosine_sim(qry_embedding, psg_embeddings)
        cosine_top100 = get_top100(df_psg.groupby('qid').get_group(qid), cosine_scores)
        cosine_results = cosine_results + cosine_top100                       # append top 100 to results list
        
        ### Bm25 similarity results ###
        bm25_scores = bm25_similarity(df_psg, qid, qry_tokens, psg_tokens, inv_idx, chunk, k1=1.2, k2=100, b=0.75)
        bm25_top100 = get_top100(chunk, bm25_scores)
        bm25_results = bm25_results + bm25_top100
    
    create_csvfile(OUTPUTFILE_TFIDF, cosine_results)
    print('successfully saved cosine results to csv!')
    create_csvfile(OUTPUTFILE_BM25, bm25_results)
    print('successfully saved bm25 results to csv!')


