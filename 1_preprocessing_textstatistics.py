#%% 0. Import Packages
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

#%% 1. Read passage data 

# open text file in read mode
with open('passage-collection.txt') as file: 
    # read all characters of the file into one string
    passages = file.read() 
    
#%% 2. Preprocessing 

### a function preprocessing text data
def preprocessing(text): 
    """

    Parameters
    ----------
    text : STRING
        any text composed of characters.

    Returns
    -------
    tokens_stemmed : LIST
        contains input text in separated tokens 

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

processed_tks = preprocessing(passages)
        
#%% 3. Vocabulary size & normalized term frequency

### counting number of times each term occurs in text
def calc_term_count(processed_tks): 
    """

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

term_count = calc_term_count(processed_tks)
print('vocabulary size:', len(term_count), 'terms')

### normalized term frequency
def normalized_freq(term_count): 
    """

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
    freq = np.array(list(term_count.values()))
    
    # sort indices of counts values & get their rank
    freq_sorted = np.argsort(-freq)
    ranks = np.empty(len(freq), dtype=int)
    ranks[freq_sorted] = np.arange(1, len(freq) + 1)
    
    # normalized frequency
    freq_norm = freq / sum(freq)

    return freq_norm, ranks

term_freq, ranks = normalized_freq(term_count)

#%% 4. Plotting normalized frequency against rank

# set style and fontstyle for plots
sns.set_style('whitegrid')
plt.rcParams['font.sans-serif'] = 'Georgia'
plt.rcParams['font.family'] = 'sans-serif'

### function plotting normalized frequency
def plot_norm_freq():
    
    fig = plt.figure()
    sns.lineplot( x=np.log(ranks), y=np.log(term_freq), color='darkblue')
    plt.suptitle('Term Frequency vs. Term Rank', fontweight='bold')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    sns.despine()

    return None

plot_norm_freq()

#%% 5. Comparing normalized frequency with Zipfs law distribution

### function calculating zipfian distribution of terms
def zipf_distr(ranks, s=1, N=len(ranks)):
    
    """

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
    
    rank_sum = sum( 1 / (np.linspace(1, N, N))**s )
    zipf_freq = [ 1 / (k**s * rank_sum ) for k in ranks ] 
    
    return zipf_freq

zipf_freq = np.array(zipf_distr(ranks))

### ploting normalized distribution of terms against zipfian distribution 
def plot_norm_zipf():
    
    fig = plt.figure()
    sns.lineplot( x=np.log(ranks), y=np.log(term_freq), color='darkblue')
    sns.lineplot( x=np.log(ranks), y=np.log(zipf_freq), color='red', linestyle='dashed')
    plt.suptitle("Empirical Term Frequency vs. Zipf's Law", fontweight='bold')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    plt.legend(['Empirical Distribution', "Zipfian Distribution" ])
    sns.despine()
    
    return None

plot_norm_zipf()

### quantifying difference between normalized and zipfian term frequency with with MSE
term_freq_sorted = abs(np.sort(-term_freq))
zipf_freq_sorted = abs(np.sort(-zipf_freq))
squared_error = (term_freq_sorted - zipf_freq_sorted)**2

MSE = (1/len(ranks)) * sum(squared_error)
print('MSE between normalized and zipfian distribution is:', MSE)

#%% 6. Removing stop words and comparing to Zipfian distribution

# importing english stop words
nltk.download('stopwords')

### function that preprocesses stop words and then removes them from a list of words
def remove_stopw(word_list):
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

# list of terms in passages without stop words  
clean_tks = remove_stopw(processed_tks)

# updated term frequency
clean_term_count = calc_term_count(clean_tks)

# deleting remaining punctuation 
top100 = clean_term_count.most_common(-100)
del clean_term_count['’'], clean_term_count['”'], clean_term_count['“']

### recalculating normalized and zipfian term frequency and ranks 
clean_freq, clean_ranks = normalized_freq(clean_term_count)
clean_zipf_freq = zipf_distr(clean_ranks)


### plotting empirical distribution after stopwords removal against zipf distribution 
def plot_norm_zipf_without_stpwrds():
    
    fig = plt.figure()
    sns.lineplot( x=np.log(clean_ranks), y=np.log(clean_freq), color='darkblue')
    sns.lineplot( x=np.log(clean_ranks), y=np.log(clean_zipf_freq), color='red', linestyle='dashed')
    plt.suptitle("Empirical Term Frequency vs. Zipf's Law", fontweight='bold')
    plt.title('Stop words removed')
    plt.xlabel('Rank k (log)')
    plt.ylabel('Term probability (log)')
    plt.legend(['Empirical Distribution', "Zipfian Distribution" ])
    sns.despine()
    
    return None

plot_norm_zipf_without_stpwrds()

### quantifying difference after stop words removal with MSE
clean_freq_sorted = abs(np.sort(-clean_freq))
clean_squared_error = (clean_freq_sorted - clean_zipf_freq)**2
clean_MSE = (1/len(clean_ranks)) * sum(clean_squared_error)
print('MSE after removing stop words is:', clean_MSE)

#%%


