# coding=utf-8
import re

__author__ = 'rcastro'

from gensim.models import Word2Vec
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

def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 2. Remove non-letters
    #letters_only = re.sub("[^a-zA-Z]", " ", raw_review)
    #
    # 3. Convert to lower case, split into individual words
    words = raw_review.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))


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
clean_train_reviews = [x.title for x in abstracts]
#
# for i in xrange( 0, num_reviews ):
#     # If the index is evenly divisible by 1000, print a message
#     if( (i+1)%1000 == 0 ):
#         print "Review %d of %d\n" % ( i+1, num_reviews )
#     clean_train_reviews.append( texts[i]) #review_to_words( texts[i] ))


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = 'english',   \
                             lowercase=True, \
                             ngram_range=(1, 2), \
                             max_features = 155000)

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
print vocab

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print count, tag


print "Training the random forest..."
from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# forest = forest.fit( train_data_features, train["sentiment"] )

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(clean_train_reviews)