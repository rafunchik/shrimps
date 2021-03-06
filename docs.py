# coding=utf-8
from __future__ import print_function
import codecs
import os
import re
from gensim import corpora, matutils
from abstract import Abstract
import numpy

__author__ = 'rcastro'

from gensim.models import LdaModel, LsiModel, HdpModel

# model = Word2Vec.load_word2vec_format("/Users/rcastro/nltk_data/word2vec_models/GoogleNews-vectors-negative300.bin", binary=True)
# print(model.most_similar('Crayfish', topn=5))

import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

print("get the abstracts")
text = ''
clean_abstracts_filename = 'clean_abstracts.txt'
if not os.path.isfile(clean_abstracts_filename):
    try:
        with codecs.open('abstracts.txt', 'r', encoding='utf8') as abstracts_file:
            text = abstracts_file.read().strip()
    except IOError as e:
        print('Operation failed: %s' % e.strerror)
else:
    pass  # serialize the clean abstracts

abstracts = [Abstract(x) for x in text.split("\r\n\r\n")]

num_abstracts = len(abstracts)
clean_abstracts = [x.text for x in abstracts]


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
# abstract_vectors = [analyzer(w) for w in clean_abstracts]
# TfidfTransformer() ->


#
# for i in xrange( 0, num_abstracts ):
#     # If the index is evenly divisible by 1000, print a message
#     if( (i+1)%1000 == 0 ):
#         print "Review %d of %d\n" % ( i+1, num_abstracts )
#     clean_abstracts.append( texts[i]) #review_to_words( texts[i] ))




# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of
# strings.
# train_data_features = vectorizer.fit_transform(clean_abstracts)
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

"""
Aqui vectorizamos el texto de los articulos usando TF/IDF quitando primero los tokens que son solo numericos,
y las stopwords en ingles.
Selecciona casi todos los unigramas y los bigramas de dos caracteres (donde el segundo caracter puede ser -) al menos
(en minusculas).
"""
vectorizer = TfidfVectorizer(analyzer="word",
                             tokenizer=None,
                             preprocessor=remove_numeric_tokens,
                             stop_words='english',
                             lowercase=True,
                             ngram_range=(1, 2),
                             min_df=1,
                             max_df=1.0, # se puede disminuir el umbral para ignorar terminos que aparecen en muchos docs
                             token_pattern=r"(?u)\b[\w][\w|-]+\b",
                             max_features=155000)
analyzer = vectorizer.build_analyzer()

abstract_vectors = [analyzer(w) for w in clean_abstracts]

tfidf_matrix = vectorizer.fit_transform(clean_abstracts)
terms = vectorizer.get_feature_names()  # todos los terminos (unigramas y bigramas)
# dictionary = corpora.Dictionary(clean_abstracts)
#
#
# from sklearn.metrics.pairwise import cosine_similarity
#
# dist = 1 - cosine_similarity(tfidf_matrix)

from sklearn.cluster import KMeans
from sklearn.externals import joblib

num_clusters = 5  # numero predefinido de clusters, hay que probar en un rango
if not os.path.isfile('doc_cluster.pkl'):  # carga del disco si lo corriste ya una vez, comentalo si lo quieres reescribir
    km = KMeans(n_clusters=num_clusters)  # kmeans usando cosine distance, agrupa los abstracts similares
    km.fit(tfidf_matrix)
    joblib.dump(km, 'doc_cluster.pkl')
else:
    km = joblib.load('doc_cluster.pkl')

clusters = km.labels_.tolist()

import pandas as pd

article_titles = {'title': [x.title for x in abstracts], 'cluster': clusters}

frame = pd.DataFrame(article_titles, index=[clusters], columns=['title', 'cluster'])

print(frame['cluster'].value_counts())

print("Top terms per cluster:")
# sort cluster centers by proximity to centroid (usando cosine distance)
order_centroids = km.cluster_centers_.argsort()[:, ::-1]

for i in range(num_clusters):
    print("Cluster %d words:" % i)
    for ind in order_centroids[i, :7]:  # replace 5 with n words per cluster
        print(' %s' % terms[ind].encode('utf-8', 'ignore'))  # las 7 palabras mas representativas de cada cluster

        #
        # print( "Cluster %d titles:" % i)
        # for title in frame.ix[i]['title'].values.tolist()[:5]:
        #     print (' %s,' % title)




# create a Gensim dictionary from the texts
dictionary = corpora.Dictionary(abstract_vectors)

# remove extremes (similar to the min/max df step used when creating the tf-idf matrix)
dictionary.filter_extremes(no_below=1, no_above=0.8)  # filtra los terminos mas comunes
#

corpus_filename = 'deerwester.mm'
if not os.path.isfile(corpus_filename):
    # convert the dictionary to a bag of words corpus for reference
    corpus = [dictionary.doc2bow(review) for review in abstract_vectors]
    corpora.MmCorpus.serialize(corpus_filename, corpus)
else:
    corpus = corpora.MmCorpus(corpus_filename)



#  vamos a utilizar Latent semantic indexing para tratar categorizar los abstracts

print("lsi")
lsi_filename = 'model.lsi'
if not os.path.isfile(lsi_filename):
    lsi = LsiModel(corpus, id2word=dictionary, num_topics=5)  # initialize an LSI transformation, 5 topicos
    #
    lsi.save(lsi_filename)  # same for tfidf, lda, ...
else:
    lsi = LsiModel.load(lsi_filename)

lsi_topics = 5  # numero predefinido de topicos
def print_topic(lsi, topicno, topn=7):
    """
        Return a single topic as a formatted string. See `show_topic()` for parameters.

        >>> lsimodel.print_topic(topicno, topn)
        '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + -0.174 * "functor" + -0.168 * "operator"'

        """
    return ' + '.join(['%.3f*"%s"' % (v, k) for k, v in show_topic(lsi, topicno, topn)])


def show_topic(lsi, topicno, topn=7):
    """
        Return a specified topic (=left singular vector), 0 <= `topicno` < `self.num_topics`,
        as a string.

        Return only the `topn` words which contribute the most to the direction
        of the topic (both negative and positive).

        >>> lsimodel.show_topic(topicno, topn)
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


def show_topics(num_topics=lsi_topics, num_words=7, log=True, formatted=True, lsi=None):
    """
        Return `num_topics` most significant topics (return all by default).
        For each topic, show `num_words` most significant words (7 words by default).

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


show_topics(lsi=lsi)  # imprime los topicos (categorias)


# try with BoW vectors too?



#  vamos a utilizar Latent Dirichlet Allocation para tratar de categorizar los abstracts
# este se demora la primera q lo corres para entrenar el modelo
print("lda")
lda_filename = 'model.lda'
if not os.path.isfile(lda_filename):
    lda = LdaModel(corpus, num_topics=5,
                   id2word=dictionary,
                   update_every=5,
                   chunksize=10000,
                   passes=100)
    lda.save('/tmp/model.lda')
else:
    lda = LdaModel.load('/tmp/model.lda')
lda.show_topics()
topics_matrix = lda.show_topics(formatted=False, num_words=7)

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


# otro modelo mas para categorizar documentos, Hierarchical Dirichlet Process
print("HDP")
model = HdpModel(corpus, id2word=dictionary)
model.show_topics(log=True, topics=5)

#  ver https://radimrehurek.com/gensim/tut2.html
