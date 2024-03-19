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

Out of up to 1'000 passages, the models retrieve at most 100 passages for each query. At the end, a *.csv* file is created for each model containing each query and the top 100 passages in ascending order depending on their score.

## Structure of Repository
First, text data is preprocessed, vocabulary of terms gets idenfitified, and some basic analysis is run to evaluate the frequency
of unique term accross all passages. The distribution of term frequencies is also compared to the Zipfian distribution, since it is
common that term frequencies follow Zipf's Law. All of this can be found in file *1_preprocessing_textstatistic*. Second, an inverted
index is created to retrieve passages efficiently in file *2_invertedindex.py*. Third, the vector space model and BM25 are 
implemented and applied to the dataset in file *3_tfidf_bm25.py*. Lastly, the query likelihood language models are executed in 
*4_querylikelihoodl_lm.py*.

## How to run my code

