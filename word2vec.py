# coding=utf-8
import codecs
import re
from abstract import Abstract

__author__ = 'rcastro'

from gensim.models import Word2Vec
from codecs import open
import nltk
#nltk.download()  # Download text data sets, including stop words
from nltk.corpus import stopwords # Import the stop word list
import numpy as np

#model = Word2Vec.load_word2vec_format("/Users/rcastro/nltk_data/word2vec_models/GoogleNews-vectors-negative300.bin", binary=True)
#print(model.most_similar('Crayfish', topn=5))

print ("get the abstracts")
text = ''
try:
    with codecs.open('/Users/rcastro/dev/abstracts.txt', 'r', encoding='utf8') as abstracts_file:
        text = abstracts_file.read().strip()
except IOError as e:
    print ('Operation failed: %s' % e.strerror)

abstracts = [Abstract(x) for x in text.split("\r\n\r\n")]
num_reviews = len(abstracts)
clean_train_reviews = [x.text for x in abstracts]

def remove_numeric_tokens(string):
    return re.sub(r'\d+[^\w|-]+', ' ', string)

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



# Download the punkt tokenizer for sentence splitting
import nltk.data
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')


# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=True ):
    # Function to split a review into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    raw_sentences = tokenizer.tokenize(review.strip())
    #
    # 2. Loop over each sentence
    sentences = []
    for raw_sentence in raw_sentences:
        # If a sentence is empty, skip it
        if len(raw_sentence) > 0:
            # Otherwise, call review_to_wordlist to get a list of words
            sentences.append( )
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

sentences = []  # Initialize an empty list of sentences

print "Parsing sentences from training set"
for review in clean_train_reviews:
    sentences += review_to_sentences(review, tokenizer)


# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

# Set values for various parameters
num_features = 400    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
print "Training model..."

# bigram_transformer = gensim.models.Phrases(sentences)
# >>> model = Word2Vec(bigram_transformer[sentences], size=100, ...)

model = word2vec.Word2Vec(sentences, workers=num_workers,
                          size=num_features, min_count = min_word_count,
                          window = context, sample = downsampling, batch_words = 1000)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "400features_2minwords_20context"
model.save(model_name)

print model.doesnt_match("man woman child kitchen".split())
print model.doesnt_match("france england germany berlin".split())
print model.most_similar("prawn", topn=10)