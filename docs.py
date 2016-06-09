# coding=utf-8
import re
from gensim import corpora
from abstract import Abstract

__author__ = 'rcastro'

from gensim.models import Word2Vec, LdaModel, LsiModel
from sklearn.feature_extraction.text import CountVectorizer

import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import numpy as np

#model = Word2Vec.load_word2vec_format("/Users/rcastro/nltk_data/word2vec_models/GoogleNews-vectors-negative300.bin", binary=True)
#print(model.most_similar('Crayfish', topn=5))


# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
#train["reviews"]

# def review_to_words( raw_review ):
#     # Function to convert a raw review to a string of words
#     # The input is a single string (a raw movie review), and
#     # the output is a single string (a preprocessed movie review)
#     #
#     # 2. Remove non-letters
#     #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
#     #
#     # 3. Convert to lower case, split into individual words
#     words = raw_review.lower().split()
#     #
#     # 4. In Python, searching a set is much faster than searching
#     #   a list, so convert the stop words to a set
#     stops = set(stopwords.words("english"))
#     #
#     # 5. Remove stop words
#     meaningful_words = [w for w in words if not w in stops]
#     #
#     # 6. Join the words back into one string separated by space,
#     # and return the result.
#     return( " ".join( meaningful_words ))


print "get the abstracts"
text = ''
try:
  with open('/Users/rcastro/dev/abstracts.txt', 'r') as abstracts_file:
    text = abstracts_file.read().strip()
except IOError as e:
  print 'Operation failed: %s' % e.strerror

abstracts = [Abstract(x) for x in text.split("\r\n\r\n")]




print "Cleaning and parsing the training set movie reviews...\n"
clean_train_reviews = []
num_reviews = len(abstracts)
clean_train_reviews = [x.text for x in abstracts]
stops = set(stopwords.words("english"))

def get_tokens_list(my_text):
    words = [w for w in nltk.word_tokenize(my_text) if not w in stops]
    return words + [' '.join(x) for x in nltk.bigrams(words)]

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             lowercase=True, \
                             ngram_range=(1, 2), \
                             max_features = 155000)
analyzer = vectorizer.build_analyzer()

review_lists = [analyzer(w) for w in clean_train_reviews]



#
# for i in xrange( 0, num_reviews ):
#     # If the index is evenly divisible by 1000, print a message
#     if( (i+1)%1000 == 0 ):
#         print "Review %d of %d\n" % ( i+1, num_reviews )
#     clean_train_reviews.append( texts[i]) #review_to_words( texts[i] ))




# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
train_data_features = vectorizer.fit_transform(clean_train_reviews)

# Numpy arrays are easy to work with, so convert the result to an
# array
train_data_features = train_data_features.toarray()
# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# Take a look at the words in the vocabulary
vocab = vectorizer.get_feature_names()


# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print count, tag


#print "Training the random forest..."
#from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# forest = forest.fit( train_data_features, train["sentiment"] )

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             lowercase=True, \
                             ngram_range=(1, 2), \
                             max_features = 155000)
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_train_reviews)
terms = tfidf_vectorizer.get_feature_names()
#dictionary = corpora.Dictionary(clean_train_reviews)


from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)
print
print

from sklearn.cluster import KMeans
#
num_clusters = 5
#
km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

#uncomment the below to save your model
#since I've already run my model I am loading from the pickle

joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

import pandas as pd

films = { 'title': [x.title for x in abstracts], 'cluster': clusters}

frame = pd.DataFrame(films, index = [clusters] , columns = ['title', 'cluster'])

print frame['cluster'].value_counts()


# from __future__ import print_function

# print("Top terms per cluster:")
# #sort cluster centers by proximity to centroid
# order_centroids = km.cluster_centers_.argsort()[:, ::-1]
#
# for i in range(num_clusters):
#     print "Cluster %d words:" % i
#
#     for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
#         print ' %s' % terms[ind].encode('utf-8', 'ignore')
#
#
#     print "Cluster %d titles:" % i
#     for title in frame.ix[i]['title'].values.tolist()[:5]:
#         print ' %s,' % title




#create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(review_lists)

#remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)
#
# #convert the dictionary to a bag of words corpus for reference
# corpus = [dictionary.doc2bow(review) for review in review_lists]
# corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

corpus = corpora.MmCorpus('/tmp/deerwester.mm')

lsi = LsiModel(corpus, id2word=dictionary, num_topics=5) # initialize an LSI transformation
lsi.print_topics(5)

lsi.save('/tmp/model.lsi') # same for tfidf, lda, ...
lsi = LsiModel.load('/tmp/model.lsi')

lda = LdaModel(corpus, num_topics=5,
                            id2word=dictionary,
                            update_every=5,
                            chunksize=10000,
                            passes=10)
lda.show_topics()
topics_matrix = lda.show_topics(formatted=False, num_words=20)
# topics_matrix = np.array(topics_matrix)
#
# topic_words = topics_matrix[:,:,1]
# for i in topic_words:
#     print([str(word) for word in i])