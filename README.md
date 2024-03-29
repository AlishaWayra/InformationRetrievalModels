# Information Retrieval Models

## Project Description
In this project, I focused on **implementing selected information retrieval and language models** covered during my MSc in Data Science. 
Most models are already available as functions in python packages, but the aim was to deepen my understanding in this field by 
building simplified version from scratch and **applying them to a query-passage text dataset**. Overall, the models retrieve a specified
amount of passages for a given search query which are relevant to the query's content, and sort passages in descending order based on the relevancy score 
that models assign to each passage. 

## Dataset
There are three datasets used to apply the models. The first dataset *candidate-passages-top1000.tsv* contains **200 unique search queries** and for each query up 
to **1'000 pre-selected suitable passages**. The passages are not unique to the query and can occur several times throughout the dataset 
if they are suitable for more than one query. The dataset contains **4 features**:
- **qid**: idenfication number of a query
- **pid**: identification number of a passage
- **query**: the query text
- **passage**: the passage text

The dataset *passage-collection.txt* only contains the passage text in a text file, whereas *test-queries.tsv* only contains the unique
queries, again with a **qid** and **query** column as in the first dataset. All data was provided by the UCL course "Information Retrieval and Data Mining" (COMP0084).

## Models
The following models are implemented:
- vector space model with TF-IDF representation of passages and queries and cosine similarity 
- BM25
- query likelihood language model with:
  - Laplace smoothing
  - Lidstone correction
  - Dirichlet smoothing

Out of up to 1'000 passages, the models retrieve at most 100 passages for each query. At the end, a *.csv* file is created for each model containing each query and the top 100 passages in ascending order depending on their score. For those interested, here are some details about the different models:

### Vector space model with TF-IDF representation and cosine similarity
The main goal of the retrieval models is to evaluate how relevant a passage is for a given search query. A vector space model achieves this by assessing how similar two vectors are in a d-dimensional space, in this case how similar a passage is to a query. Hence, query and passage first need to be represented as vectors capturing the charactaristics of their content. The TF-IDF method is used to represent documents as vectors by computing the **term frequency** (TF) of words occuring in a document and the **inverted document frequency** (IDF) by using
$$tfidf(w_i,D) = tf(w_i,D) \times idf(w_i),$$
$$tf(w_i,D) = \frac{v_i}{m},$$
$$idf(w_i) = log\frac{N}{n}.$$
$w_i$ with $i \in [1,d]$ denotes a word from the whole corpus of $d$ unique words, $D$ denotes a document with length $m$, $v_i$ denotes how often $w_i$ occurs in $D$, $N$ denotes the total number of documents in the corpus, and $n$ denotes the number of documents where word $w_i$ occurs in. If a word doesn't occur in $D$, the $tfidf(w_i, D)$ is 0. Any document, whether a passage or a query, can then be presented as a tfidf-vector of 

$$tfidf_D = [tfidf(w_1, D), tfidf(w_2,D), ... , tfidf(w_d,D)] $$
with $tfidf_D \in \mathbb{R}^{d \times 1}$.

The similarity between a query vector $tfidf_Q$ and passage vector $tfidf_P$ can then be assessed using the cosine similarity metric, which computes the cosine of the angle $\theta$ between two vectors by using the dot product and vector magnitudes:
$$cos(\theta) =  \frac{tfidf_Q \cdot tfidf_P}{|tfidf_Q||tfidf_P|}.$$

The cosine similarity can take on values between $[-1,1]$, whereby $1$ signifies maximum similarity of two vectors, implying proportionality.

### BM25 model
BM25 is a probabilistic ranking model based the term frequency of query words in a passage. Query and passages are not represented as vectors and the relevancy score between a query and a passage is directly computed. For the query words $q_1, q_2, ... , q_z$ in query $Q$, BM25 score is calculated as

$$bm25(Q,P) = \sum_{i=1}^z log\frac{N-n_i+0.5}{n_i + 0.5} \frac{(k_1+1)f_i}{K+f_i} \frac{(k_2+1)qf_i}{k_2+qf_i}.$$

The parameters $k_1$ is typically set to $1.2$ and $k_2$ is given a value between $[0,1000]$. $K$ is set empirically according to $K=k_1((1-b)+b \cdot \frac{m}{avdl})$, $avdl$ denoting the average document length and $b=0.7$ typically. $qf_i$ denotes the term frequency of $q_i$ in query $Q$ and $f_i$ denotes the term frequency of $q_i$ in passage $P$.

## Structure of Repository
First, text data is preprocessed, vocabulary of terms gets idenfitified, and some basic analysis is run to evaluate the frequency
of unique term accross all passages. The distribution of term frequencies is also compared to the Zipfian distribution, since it is
common that term frequencies follow Zipf's Law. All of this can be found in file *1_preprocessing_textstatistic*. Second, an inverted
index is created to retrieve passages efficiently in file *2_invertedindex.py*. Third, the vector space model and BM25 are 
implemented and applied to the dataset in file *3_tfidf_bm25.py*. Lastly, the query likelihood language models are executed in 
*4_querylikelihoodl_lm.py*.

## How to run my code

