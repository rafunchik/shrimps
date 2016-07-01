# coding=utf-8
import os
import re
import html2text as html2text
import numpy as np
from abstract import Abstract

__author__ = 'rcastro'

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedLineDocument, TaggedDocument
from codecs import open


def remove_numeric_tokens(string):
    return re.sub(r'\d+[^\w|-]+', ' ', string)


# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text):
    norm_text = text.lower()
    #     control_chars = [chr(0x85)]
    #     for c in control_chars:
    #         norm_text = norm_text.replace(c, ' ')    # Replace breaks with spaces
    #     norm_text = norm_text.replace('<br />', ' ')

    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')

    return norm_text


sentences_keywords = []
docs_filename = '/Users/rcastro/dev/alldata-id.txt6'
if not os.path.isfile(docs_filename):
    print "get the abstracts"
    text = ''
    try:
        with open('/Users/rcastro/dev/abstracts.txt', 'r', encoding='utf8') as abstracts_file:
            text = abstracts_file.read().strip()

    except IOError as e:
        print 'Operation failed: %s' % e.strerror

    abstracts = [Abstract(x) for x in text.split("\r\n\r\n")]
    for article in abstracts:
        sentences_keywords.append([normalize_text(remove_numeric_tokens(x)).strip() for x in article.keywords])
    # with open(docs_filename, 'w', encoding='utf8') as f:
    #     for idx, line in enumerate([normalize_text(remove_numeric_tokens(x.text)) for x in abstracts]):
    #         f.write(line + '\n')
    #         # num_line = "_*{0} {1}\n".format(idx, line)
    #         # f.write(line+'\n')

sentences = TaggedLineDocument('/Users/rcastro/dev/alldata-id.txt')
# sentences = sentences_keywords


from gensim.models import Doc2Vec
import gensim.models.doc2vec
from collections import OrderedDict
import multiprocessing

cores = multiprocessing.cpu_count()
assert gensim.models.doc2vec.FAST_VERSION > -1, "this will be painfully slow otherwise"

# Set values for various parameters
num_features = 400    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 20          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

simple_models = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=50, window=10, negative=10, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=50, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=50, window=10, negative=5, hs=0, min_count=2, workers=cores),
]



simple_models_400 = [
    # PV-DM w/concatenation - window=5 (both sides) approximates paper's 10-word total window size
    Doc2Vec(dm=1, dm_concat=1, size=400, window=10, negative=10, hs=0, min_count=2, workers=cores),
    # PV-DBOW
    Doc2Vec(dm=0, size=400, negative=5, hs=0, min_count=2, workers=cores),
    # PV-DM w/average
    Doc2Vec(dm=1, dm_mean=1, size=400, window=10, negative=5, hs=0, min_count=2, workers=cores),
]

# speed setup by sharing results of 1st model's vocabulary scan
simple_models[0].build_vocab(sentences)  # PV-DM/concat requires one special NULL word so it serves as template
print(simple_models[0])
for model in simple_models[1:]:
    model.reset_from(simple_models[0])
    print(model)

# for model in simple_models_100:
#     model.reset_from(simple_models[0])
#     print(model)

for model in simple_models_400:
    model.reset_from(simple_models[0])
    print(model)

all_models = simple_models+simple_models_400
models_by_name = OrderedDict((str(model), model) for model in all_models)

'''
Following the paper, we also evaluate models in pairs. These wrappers return the concatenation of the vectors from each model. (Only the singular models are trained.)
In [5]:
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
models_by_name['dbow+dmm'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[2]])
models_by_name['dbow+dmc'] = ConcatenatedDoc2Vec([simple_models[1], simple_models[0]])
'''

from random import shuffle
import datetime

# for timing
from contextlib import contextmanager
from timeit import default_timer
import random

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

passes = 20
print("START %s" % datetime.datetime.now())

all_docs = []
for doc in sentences:
    all_docs.append(doc)
for epoch in range(passes):
    shuffle(all_docs)  # shuffling gets best results

# doc_id = np.random.randint(len(sentences))    #
doc_id = np.random.randint(simple_models[0].docvecs.count)  # pick random doc, re-run cell for more examples

for name, model in models_by_name.items()[:3]:
    with elapsed_timer() as elapsed:
        model.train(all_docs)
        # duration = '%.1f' % elapsed()
        # print (name, duration)
        sims = model.docvecs.most_similar(doc_id, topn=model.docvecs.count)  # get *all* similar documents
        print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
        print(u'TARGET : «%s»\n' % (' '.join(all_docs[doc_id].words)))
        print(u'TARGET keywords: «%s»\n' % (' '.join(sentences_keywords[doc_id])))
        for label, index in [('MOST', 0)]: #, ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
            print(u'%s %s: «%s»\n' % (label, sims[index][1], ' '.join(all_docs[sims[index][0]].words)))
            print(u'Similar keywords : «%s»\n' % (' '.join(sentences_keywords[sims[index][0]])))


word_models = all_models[:3]
# while True:
#     word = random.choice(word_models[0].index2word)
#     if word_models[0].vocab[word].count > 10 and len(word)>3:
#         break

word = "aquaculture" #diadromous
similars_per_model = [str(model.most_similar(word, topn=5)).replace('), ','),<br>\n') for model in word_models]
similar_table = ("<table><tr><th>" +
    "</th><th>".join([str(model) for model in word_models]) +
    "</th></tr><tr><td>" +
    "</td><td>".join(similars_per_model) +
    "</td></tr></table>")
print("most similar words for '%s' (%d occurences)" % (word, simple_models[0].vocab[word].count))
print(similar_table)

#TODO import wiki model and add to word_models