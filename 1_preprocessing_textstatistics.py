
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import string
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

nltk.download('wordnet')
nltk.download('stopwords')                                              # importing english stop words

#%% 1. Read passage data helper function

def import_textdata(FILE):
    """ open text file in read mode """
    with open(FILE) as file: 
        # read all characters of the file into one string
        passages = file.read() 
        
    return passages
    
#%% 2. Text preprocessing helper function 

def preprocessing(text): 
    """
    preprocessing text data
    
    Parameters
    ----------
    text : STRING
        any text composed of characters.

    Returns
    -------
    tokens_stemmed : LIST of strings
        contains text as separated tokens 

    """
    
    # 1. removing punctuation
    punc = set(string.punctuation)
    text_nopunc = ''.join([p for p in text if p not in punc])
    
    # 2. setting all words to lower case
    text_lowcase = text_nopunc.lower()
    
    # 3. tokenizing whole text (string) into single tokens
    tokens = word_tokenize(text_lowcase)
    
    # initialize stemming method
    stemmer= PorterStemmer()
    
    # 4. Apply Stemming
    tokens_stemmed = [ stemmer.stem(token) for token in tokens ]
    
    return tokens_stemmed
        
#%% 3. Vocabulary size & normalized term frequency helper function

def absolute_freq(processed_tks): 
    """
    counting number of times each term occurs in text
    
    Parameters
    ----------
    processed_tks : LIST
        contains separate tokens

    Returns
    -------
    occurences : FREQDIST
        number of times each term of processed_tks occurs

    """

    occurences = FreqDist(processed_tks)
    
    return occurences
 
def normalized_freq(tks_count): 
    """
    computing normalized term frequency
    
    Parameters
    ----------
    term_count : FREQDIST
        number of times a selection of terms occur each.

    Returns
    -------
    term_freq : ARRAY
        frequency of each term normalized by the sum of all frequencies.
    ranks : ARRAY
        frequency rank of each term, starting with 1 for the most frequent term
        in ascending order.

    """
    
    # absolute frequencies of terms
    freq = np.array(list(tks_count.values()))
    
    # sort indices of frequencies in descending order & get their rank
    freq_idx_sorted = np.argsort(-freq)
    ranks = np.empty(len(freq), dtype=int)
    ranks[freq_idx_sorted] = np.arange(1, len(freq) + 1)
    
    # normalized frequency
    freq_norm = freq / sum(freq)

    return freq_norm, ranks


#%% 4. Plotting normalized frequency against rank helper function

# set style and fontstyle for plots
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = 'Georgia'
plt.rcParams['font.family'] = 'sans-serif'

### function plotting normalized frequency
def plot_norm_freq():
    
    fig = plt.figure()
    sns.lineplot( x=np.log(ranks), y=np.log(tks_freq), color='darkblue')
    plt.suptitle('Term Frequency vs. Term Rank', fontweight='bold')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    sns.despine()

    return None

#%% 5. normalized term frequency vs. Zipfs distribution helper functions

def zipf_distr(ranks, s=1):
    """
    calculating zipfian distribution of terms
    
    Parameters
    ----------
    ranks : ARRAY
        frequency rank of each term, starting with 1 for the most frequent term
        in ascending order.
    s : INTEGER
        distribution parameter, by default 1
    N : INTEGER
        vocabulary size

    Returns
    -------
    zipf_freq : ARRAY
        term frequencies expected by a zipfian distribution.

    """
    N = len(ranks)
    rank_sum = sum( 1 / (np.linspace(1, N, N))**s )
    zipf_freq = [ 1 / (k**s * rank_sum ) for k in ranks ] 
    
    return zipf_freq

def plot_norm_zipf():
    """ ploting normalized term distribution against zipfian distribution """
    fig = plt.figure()
    sns.lineplot( x=np.log(ranks), y=np.log(tks_freq), color='darkblue')
    sns.lineplot( x=np.log(ranks), y=np.log(zipf_freq), color='red', linestyle='dashed')
    plt.suptitle("Empirical Term Frequency vs. Zipf's Law", fontweight='bold')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    plt.legend(['Empirical Distribution', "Zipfian Distribution" ])
    sns.despine()
    
    return None
 
def calc_MSE(term_freq, zipf_freq, ranks):
    """ Mean squared error between normalized and zipfian term frequency """
    term_freq_sorted = abs(np.sort(-term_freq))
    zipf_freq_sorted = abs(np.sort(-zipf_freq))
    squared_error = (term_freq_sorted - zipf_freq_sorted)**2
    
    MSE = (1/len(ranks)) * sum(squared_error)
    
    return MSE

#%% 6. normalized term frequency vs. Zipfs distribution after removing stopword helper functions

def remove_stopw(word_list):
    """ preprocesses stop words and then removes them from a list of words """
    # initialize stemmer
    stemmer= PorterStemmer()
    
    # nltk stop words
    stopw_nltk = stopwords.words('english')
    # sklearn stop words
    stopw_sc = list(ENGLISH_STOP_WORDS)
    stopw_sc.remove('even')
    # combine sklearn and nltk stop words
    stopw = set(stopw_nltk + stopw_sc)
    
    # remove punctuations from stopwords
    punc = set(string.punctuation)
    stopw_nopunc = [''.join([c for c in word if c not in punc]) for word in stopw]
    stopw_processed = set([stemmer.stem(word.lower()) for word in stopw_nopunc])

    clean = [ token for token in word_list if token not in stopw_processed ]

    return clean

def plot_norm_zipf_without_stpwrds():
    """ plotting normalized distribution after stopwords removal against zipf distribution  """
    fig = plt.figure()
    sns.lineplot( x=np.log(ranks_wsw), y=np.log(tks_freq_wsw), color='darkblue')
    sns.lineplot( x=np.log(ranks_wsw), y=np.log(zipf_freq_wsw), color='red', linestyle='dashed')
    plt.suptitle("Empirical Term Frequency vs. Zipf's Law", fontweight='bold')
    plt.title('Stop words removed')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    plt.legend(['Empirical Distribution', "Zipfian Distribution" ])
    sns.despine()
    
    return None

#%% 7. text statistics output

if( __name__ == "__main__" ):
    # import text data
    FILE = 'passage-collection.txt'                                     # file name 
    passages = import_textdata(FILE)
    
    ### preprocessing and text statistics including stop words ###
    passages_tks = preprocessing(passages)
    tks_count = absolute_freq(passages_tks)
    print('--------------------------------------------------------------------------------')
    print('Text Statistics')
    print('    vocabulary size:', len(tks_count), 'terms')
    tks_freq, ranks = normalized_freq(tks_count)
    plot_norm_freq()
    
    ### generating zipfian distribution ###
    zipf_freq = np.array(zipf_distr(ranks))
    plot_norm_zipf()
    print('    MSE between normalized term frequency/distribution and zipfian distribution:', calc_MSE(tks_freq, zipf_freq, ranks))

    ### preprocessing and text statistics wihtout stopwords ###
    passages_tks_wsw = remove_stopw(passages_tks)
    tks_count_wsw = absolute_freq(passages_tks_wsw)
    del tks_count_wsw['’'], tks_count_wsw['”'], tks_count_wsw['“']      # deleting remaining punctuation 
    np.save('vocab_freq.npy', tks_count_wsw)                            # save absolute term frequency in directory as .npy 

    tks_freq_wsw, ranks_wsw = normalized_freq(tks_count_wsw)
    zipf_freq_wsw = np.array(zipf_distr(ranks_wsw))
    plot_norm_zipf_without_stpwrds()
    print('    MSE between normalized and Zipfian distribution after removing stop words:  ', calc_MSE(tks_freq_wsw, zipf_freq_wsw, ranks_wsw))

    



