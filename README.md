# Word Embeddings & Text Classification 

This project explores **word embeddings**, **dimensionality reduction**, and **text classification** techniques in **Natural Language Processing (NLP)**.  
It consists of two main parts:
1. **Exploring Word Embeddings and Visualization**
2. **Classifying Prohibited Comments using Text Features**


---

## Overview

This project demonstrates:
- Tokenization and text preprocessing  
- Training and loading pre-trained word embeddings  
- Visualizing embeddings using PCA and t-SNE  
- Representing phrases with averaged embeddings  
- Text classification with Bag-of-Words (BOW), TF-IDF, and Word Embeddings  
- Evaluating models with ROC curves and AUC metrics  

---

## Dataset

### 1. Quora Dataset
Downloaded automatically using:

wget https://www.dropbox.com/s/obaitrix9jyu84r/quora.txt?dl=1 -O ./quora.txt

## Part 1 — Working with Word Embeddings
Tokenization

Tokenization performed using nltk.WordPunctTokenizer().

All text is converted to lowercase.

Word Embeddings

Three types of embeddings are explored:

Custom embeddings trained on small corpora

GloVe pre-trained embeddings (glove-twitter-100)

FastText embeddings (fasttext-wiki-news-subwords-300)

## Example usage:

import gensim.downloader as api
model = api.load('glove-twitter-100')
model.most_similar(positive=["coder", "money"], negative=["brain"])

##  Visualization

Embeddings are reduced to 2D using:

PCA (Principal Component Analysis)

t-SNE (t-distributed Stochastic Neighbor Embedding)

Visualization with Bokeh:

draw_vectors(word_vectors_pca[:, 0], word_vectors_pca[:, 1], token=words)


Clusters reveal semantic relationships between words (e.g., synonyms and topic groups).

##  Part 2 — Text Classification
Preprocessing

Tokenization with nltk.TweetTokenizer()

Lowercasing and cleaning of comments

Splitting into train/test using train_test_split

Bag of Words (BOW)

Count word frequencies in the training set.

Build a vocabulary of the top k=10000 most frequent words.

Convert each text into a vector of word counts.

##  Model:

from sklearn.linear_model import LogisticRegression
bow_model = LogisticRegression(max_iter=10000, C=94)

##  TF-IDF Features

Implements manual TF-IDF computation:

Term Frequency (TF): word count normalized by document length

Inverse Document Frequency (IDF): log((N+1)/(df+1)) + 1

Model achieves higher AUC compared to basic BOW.

Word Embedding Features

Uses FastText embeddings to represent each comment as the sum of its word vectors.
This method significantly reduces dimensionality and improves generalization.

##  Example:

embeddings = gensim.downloader.load("fasttext-wiki-news-subwords-300")
features = np.sum([embeddings[tok] for tok in tokens if tok in embeddings], axis=0)

Evaluation

Each model’s performance is compared via ROC curves and AUC scores:

BOW model: baseline (~0.77 AUC)

TF-IDF model: improved (~0.85 AUC)

Embedding model: excellent (~0.92+ AUC)

##  Results Summary
Model Type	Feature Size	Test AUC	Comments
Bag of Words	10,000	~0.77	Baseline linear model
TF-IDF	10,000	~0.85	Better weighting of rare words
Word Embeddings	300	>0.92	Compact, semantically rich features

tqdm

