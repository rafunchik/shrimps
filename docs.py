# coding=utf-8
from __future__ import print_function
import codecs
import re
from gensim import corpora, matutils
from abstract import Abstract
import numpy

__author__ = 'rcastro'

from gensim.models import Word2Vec, LdaModel, LsiModel, HdpModel
from sklearn.feature_extraction.text import CountVectorizer

import nltk
# nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords  # Import the stop word list
import numpy as np

# model = Word2Vec.load_word2vec_format("/Users/rcastro/nltk_data/word2vec_models/GoogleNews-vectors-negative300.bin", binary=True)
# print(model.most_similar('Crayfish', topn=5))

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Initialize an empty list to hold the clean reviews
clean_train_reviews = []
# train["reviews"]

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


print ("get the abstracts")
text = ''
try:
    with codecs.open('/Users/rcastro/dev/abstracts.txt', 'r', encoding='utf8') as abstracts_file:
        text = abstracts_file.read().strip()
except IOError as e:
    print ('Operation failed: %s' % e.strerror)

abstracts = [Abstract(x) for x in text.split("\r\n\r\n")]

print ("Cleaning and parsing the training set movie reviews...\n")
clean_train_reviews = []
num_reviews = len(abstracts)
clean_train_reviews = [x.text for x in abstracts]
# stops = set(stopwords.words("english"))
#
# def get_tokens_list(my_text):
#     words = [w for w in nltk.word_tokenize(my_text) if not w in stops]
#     return words + [' '.join(x) for x in nltk.bigrams(words)]


def remove_numeric_tokens(string):
    return re.sub(r'\d+[^\w|-]+', ' ', string)


# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
# vectorizer = CountVectorizer(analyzer="word",
#                              tokenizer=None,
#                              preprocessor=remove_numeric_tokens,
#                              stop_words='english',
#                              lowercase=True,
#                              ngram_range=(1, 2),
#                              min_df=0,
#                              max_df=1.0,  # quizas probar con 0.8 x ahi
#                              token_pattern=r"(?u)\b[\w][\w|-]+\b",
#                              max_features=155000)
# analyzer = vectorizer.build_analyzer()
#
# review_lists = [analyzer(w) for w in clean_train_reviews]
# TfidfTransformer() ->


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
# train_data_features = vectorizer.fit_transform(clean_train_reviews)
#
# # Numpy arrays are easy to work with, so convert the result to an
# # array
# train_data_features = train_data_features.toarray()
# # Sum up the counts of each vocabulary word
# dist = np.sum(train_data_features, axis=0)
#
# # Take a look at the words in the vocabulary
# vocab = vectorizer.get_feature_names()


# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print count, tag


# print "Training the random forest..."
# from sklearn.ensemble import RandomForestClassifier

# Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators = 100)

# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# forest = forest.fit( train_data_features, train["sentiment"] )

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=remove_numeric_tokens,
                             stop_words='english',
                             lowercase=True,
                             ngram_range=(1, 2),
                             min_df=1,
                             max_df=1,  # quizas probar con 0.8 x ahi
                             token_pattern=r"(?u)\b[\w][\w|-]+\b",
                             max_features=155000)
analyzer = vectorizer.build_analyzer()

review_lists = [analyzer(w) for w in clean_train_reviews]

tfidf_matrix = vectorizer.fit_transform(clean_train_reviews)
terms = vectorizer.get_feature_names()
# dictionary = corpora.Dictionary(clean_train_reviews)


from sklearn.metrics.pairwise import cosine_similarity

dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
#
num_clusters = 3
#
km = KMeans(n_clusters=num_clusters)

km.fit(tfidf_matrix)

clusters = km.labels_.tolist()

from sklearn.externals import joblib

# uncomment the below to save your model
# since I've already run my model I am loading from the pickle

joblib.dump(km, 'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

import pandas as pd

films = {'title': [x.title for x in abstracts], 'cluster': clusters}

frame = pd.DataFrame(films, index=[clusters], columns=['title', 'cluster'])

print (frame['cluster'].value_counts())



print("Top terms per cluster:")
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print ("Cluster %d words:" % i)

    for ind in order_centroids[i, :6]: #replace 6 with n words per cluster
        print( ' %s' % terms[ind].encode('utf-8', 'ignore'))

    #
    # print( "Cluster %d titles:" % i)
    # for title in frame.ix[i]['title'].values.tolist()[:5]:
    #     print (' %s,' % title)




# create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(review_lists)

# remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)
#
# #convert the dictionary to a bag of words corpus for reference
corpus = [dictionary.doc2bow(review) for review in review_lists]
corpora.MmCorpus.serialize('/tmp/deerwester.mm', corpus)

corpus = corpora.MmCorpus('/tmp/deerwester.mm')

print("lsi")
lsi = LsiModel(corpus, id2word=dictionary, num_topics=5)  # initialize an LSI transformation
#
lsi.save('/tmp/model.lsi')  # same for tfidf, lda, ...
lsi = LsiModel.load('/tmp/model.lsi')


def print_topic(lsi, topicno, topn=10):
    """
        Return a single topic as a formatted string. See `show_topic()` for parameters.

        >>> lsimodel.print_topic(10, topn=5)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
    return ' + '.join(['%.3f*"%s"' % (v, k) for k, v in show_topic(lsi, topicno, topn)])


def show_topic(lsi, topicno, topn=10):
    """
        Return a specified topic (=left singular vector), 0 <= `topicno` < `self.num_topics`,
        as a string.

        Return only the `topn` words which contribute the most to the direction
        of the topic (both negative and positive).

        >>> lsimodel.show_topic(10, topn=5)
        [("category", -0.340), ("$M$", 0.298), ("algebra", 0.183), ("functor", -0.174), ("operator", -0.168)]

        """
    # size of the projection matrix can actually be smaller than `self.num_topics`,
    # if there were not enough factors (real rank of input matrix smaller than
    # `self.num_topics`). in that case, return an empty string
    if topicno >= len(lsi.projection.u.T):
        return ''
    c = numpy.asarray(lsi.projection.u.T[topicno, :]).flatten()
    norm = numpy.sqrt(numpy.sum(numpy.dot(c, c)))
    most = matutils.argsort(numpy.abs(c), topn, reverse=True)
    return [(lsi.id2word[val], 1.0 * c[val] / norm) for val in most]


def show_topics(num_topics=3, num_words=10, log=True, formatted=True, lsi=None):
    """
        Return `num_topics` most significant topics (return all by default).
        For each topic, show `num_words` most significant words (10 words by default).

        The topics are returned as a list -- a list of strings if `formatted` is
        True, or a list of `(word, probability)` 2-tuples if False.

        If `log` is True, also output this result to log.

        """
    shown = []
    for i in xrange(min(num_topics, lsi.num_topics)):
        if i < len(lsi.projection.s):
            if formatted:
                topic = print_topic(lsi, i, topn=num_words)
            else:
                topic = lsi.show_topic(i, topn=num_words)
            shown.append((i, topic))
            if log:
                print("topic #%i(%.3f): %s", i, lsi.projection.s[i], topic)
    return shown


show_topics(lsi=lsi)


# try with BoW vectors too

print("lda")
lda = LdaModel(corpus, num_topics=3,
                            id2word=dictionary,
                            update_every=5,
                            chunksize=10000,
                            passes=100)
lda.save('/tmp/model.lda')

lda = LdaModel.load('/tmp/model.lda')
lda.show_topics()
topics_matrix = lda.show_topics(formatted=False, num_words=20)

print(topics_matrix)
print(len(topics_matrix))

for topic in topics_matrix:
    i = topic[1]
    print([str(word) for word in i])
#
# topics_matrix = np.array(topics_matrix)
#
# topic_words = topics_matrix[:, :, 1]
# for i in topic_words:
#     print([str(word) for word in i])


print("HDP")
model = HdpModel(corpus, id2word=dictionary)
model.show_topics(log=True, topics=3)