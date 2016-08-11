#!/usr/bin/env python
# -*- coding: utf-8 -*-

#--------------------------------------------------------------#
#                                                              #
#  Author: Lucas Miguel S. Ponce  (lucasmsp@gmail.com)         #
#                                                              #
#--------------------------------------------------------------#


# python libs
import os, multiprocessing, linecache
from random import shuffle
import string
import time
from unicodedata import normalize
import re
import matplotlib.pyplot as plt
import numpy as np
import nltk
from dataCleanup import *
from pprint import pprint
import argparse

# gensim modules
from gensim import utils
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec

# sklearn modules
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn import metrics

printable = set(string.printable)



'''
Doc2vec on flat file of articles 
stored at data/articles in format:
<uid> <article text>
...
<uid> <article text>
'''

class TaggedDocuments(object):
    ids = []

    def __init__(self, source, cleaningfns=None):
        self.source = source

        if cleaningfns: self.cleaningfns = cleaningfns
        else: self.cleaningfns = [lambda x: x]

        # make sure that keys are unique
        with utils.smart_open(self.source) as fin:
            for line in fin:
                # split '<id> <text>' to get id
                idd = line.split(' ', 1)[0]
                self.ids.append(self.gen_id(idd))
        # assert all ids are unique
        assert len(set(self.ids)) == len(self.ids), 'prefixes non-unique'
        self.numdocs = len(self.ids)

        self.indices = xrange(self.numdocs)
    
    def __iter__(self):
        for idx in self.indices:
            lineno = idx + 1
            line = linecache.getline(self.source, lineno)
            #linecache.clearcache() # uncomment if storing file in memory isn't feasible
            yield self.tagged_sentence(line)

    def permute(self):
        '''randomly order how documents are iterated'''
        self.indices = np.random.permutation(self.numdocs)

    def tagged_sentence(self, line):
        # split '<id> <text>'
        idd, text = line.split(' ', 1)
        # clean text
        for fn in self.cleaningfns:
            text = fn(text)
        # split on spaces
        text = utils.to_unicode(text).split()
        return TaggedDocument(words=text, tags=[self.gen_id(idd)])

    def docs_perm(self):
        shuffle(self.docs)
        return self.docs

    def gen_id(self, idd):
        return 'DOC_%s' % idd



def Kmeans(n_kmeans,model):
    for doc in xrange(0, len(model.docvecs)):
        doc_vec = model.docvecs[doc]
        # print doc_vec
        vecs.append(doc_vec.reshape((1, 300)))

        #  print vecs[0]
        # print vecs[1]
        # print model.docvecs.offset2doctag
        #  print model.docvecs.doctags.keys()
    doc_vecs = np.array(vecs, dtype='float')  # TSNE expects float type values

    # print doc_vecs
    docs = []
    for i in doc_vecs:
        docs.append(i[0])
    # print  docs

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    # index2word_set = set(model.index2word)
    # print index2word_set


    print("Clustering...")
    startTime = time.time()
    km_clusterer = KMeans(n_clusters=n_kmeans, n_jobs=1, n_init=5)
    ids = km_clusterer.fit_predict(docs)
    endTime = time.time()
    print("Time taken for clustering: {} minutes".format((endTime - startTime) / 60))

    word_centroid_map = dict(zip(model.docvecs.offset2doctag, ids))

    # print "word_centroid_map"
    # print word_centroid_map[1]

    list_labels_cluster = [[] for y in range(n_kmeans)]
    for cluster in xrange(0, n_kmeans):
        # Print the cluster number
        print "Computing cluster %d..." % cluster
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
                list_labels_cluster[cluster].append(word_centroid_map.keys()[i])
                # print list_docs_cluster
                # docs_cluster = [[cluster]]

    list_docs_cluster = [[] for y in range(n_kmeans)]
    num_test_cluster = [0 for y in range(n_kmeans)]  # Traffic elements

    f = open(arg['corpus'], 'r')
    for i in f:
        id = i.split(' ', 1)[0]
        label = "DOC_" + id
        for c in range(n_kmeans):
            if label in list_labels_cluster[c]:
                list_docs_cluster[c].append(i)
                if (int(id) > 8745):
                    num_test_cluster[c] = num_test_cluster[c] + 1
                break
    f.close()

    for i in xrange(0, n_kmeans):
        print "Cluster " + str(i) + ":"
        print  "\tNum itens: %d\t(%.2f%%)" % (
        len(list_docs_cluster[i]), 100 * float(len(list_docs_cluster[i])) / len(ids))
        print  "\tTraffic itens/Cluster doc itens: %.2f%%\t# Porcentagem de tweets de transito em relacao a quantidade global" % (
        100 * float(num_test_cluster[i]) / len(list_docs_cluster[i]))
        precision = float(num_test_cluster[i]) / len(list_docs_cluster[i])
        print  "\tPrecision: %.2f\t# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)" % precision
        recall = float(num_test_cluster[i]) / 6065
        print  "\tRecall: %.2f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)" % recall
        print "\tF-Measure (harmonic avg): %.2f" % (float((2 * precision * recall)) / (precision + recall))

    f = open(arg['output'] + "_clusters.txt", 'w')
    for i in list_docs_cluster:
        f.write("Cluster:\n")
        for z in i:
            f.write(z)

    f.close()


def byDBSCAN(eps,model):

    for doc in xrange(0, len(model.docvecs)):
        doc_vec = model.docvecs[doc]
        # print doc_vec
        vecs.append(doc_vec.reshape((1, 300)))

        #  print vecs[0]
        # print vecs[1]
        # print model.docvecs.offset2doctag
        #  print model.docvecs.doctags.keys()
    doc_vecs = np.array(vecs, dtype='float')  # TSNE expects float type values

    # print doc_vecs
    docs = []
    for i in doc_vecs:
        docs.append(i[0])
    # print  docs

    # Index2word is a list that contains the names of the words in
    # the model's vocabulary. Convert it to a set, for speed
    # index2word_set = set(model.index2word)
    # print index2word_set

    print "Clustering vectors by DBSCAN"
    db = DBSCAN(eps=eps, min_samples=10).fit(docs)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters)

    word_centroid_map = dict(zip(model.docvecs.offset2doctag, labels))

    # print "word_centroid_map"
    # print word_centroid_map[1]

    list_labels_cluster = [[] for y in range(n_clusters)]
    for cluster in xrange(0, n_clusters):
        # Print the cluster number
        print "Computing cluster %d..." % cluster
        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
                list_labels_cluster[cluster].append(word_centroid_map.keys()[i])
                # print list_docs_cluster
                # docs_cluster = [[cluster]]

    list_docs_cluster = [[] for y in range(n_clusters)]
    num_test_cluster = [0 for y in range(n_clusters)]  # Traffic elements

    f = open(arg['corpus'], 'r')

    for i in f:
        id = i.split(' ', 1)[0]
        label = "DOC_" + id
        for c in range(n_clusters):
            if label in list_labels_cluster[c]:
                list_docs_cluster[c].append(i)
                if (int(id) > 8745):
                    num_test_cluster[c] = num_test_cluster[c] + 1
                break
    f.close()

    for i in xrange(0, n_clusters):
        print "Cluster " + str(i) + ":"
        print  "\tNum itens: %d\t(%.2f%%)" % (
        len(list_docs_cluster[i]), 100 * float(len(list_docs_cluster[i])) / len(labels))
        print  "\tTraffic itens/Cluster doc itens: %.2f%%\t# Porcentagem de tweets de transito em relacao a quantidade global" % (
        100 * float(num_test_cluster[i]) / len(list_docs_cluster[i]))
        precision = float(num_test_cluster[i]) / len(list_docs_cluster[i])
        print  "\tPrecision: %.2f\t# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)" % precision
        recall = float(num_test_cluster[i]) / 6065
        print  "\tRecall: %.2f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)" % recall
        if (precision + recall)==0.0:
            soma = 0.00001
        else:
            soma = precision + recall
        print "\tF-Measure (harmonic avg): %.2f" % (float((2 * precision * recall)) / (soma))

    f = open(arg['output'] + "_clusters.txt", 'w')
    for i in list_docs_cluster:
        f.write("Cluster:\n")
        for z in i:
            f.write(z)

    f.close()
    print "Total = %d" % ( np.count_nonzero(labels != -1))

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='Train doc2vec on a corpus')
    # required
    parser.add_argument('-c','--corpus', required=True, help='path to the corpus file on which to train')
    parser.add_argument('-o','--output', required=True, help='file path to output trained model')
    # doc2vec training parameters - not required.
    # NOTE: all defaults match gensims, except --sample.
    parser.add_argument('--dm',        type=int,   default=1,    help='defines training algorithm. 0: distributed bag-of-words, 1: distributed-memory.')
    parser.add_argument('--min_count', type=int,   default=5,    help='ignore all words with total frequency lower than this.')
    parser.add_argument('--window',    type=int,   default=8,    help='the maximum distance between the predicted word and context words used for prediction within a document.')
    parser.add_argument('--size',      type=int,   default=300,  help='is the dimensionality of the feature vectors.')
    parser.add_argument('--sample',    type=float, default=1e-5, help='threshold for configuring which higher-frequency words are randomly downsampled. 0 is off.')
    parser.add_argument('--negative',  type=int,   default=0,    help='if > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drawn (usually between 5-20).')
    # convert Namespace to dict
    arg = vars(parser.parse_args())

    # defines model parameters
    params = { k: arg[k] for k in ['dm','min_count','window','size','sample','negative'] }
    params.update({'workers': multiprocessing.cpu_count() })
    pprint('model parameters:')
    pprint(params)
    model = Doc2Vec(**params)

    # strip punctuation and ensure lower case
    strip_punct = lambda text: filter(lambda x: x in printable, text)
    lower = lambda text: text.lower()

    # builds the vocabulary
    print 'instantiating TaggedDocuments'
    articles = TaggedDocuments(arg['corpus'], [strip_punct, lower])
    print 'building vocabulary'
    model.build_vocab(articles)

    train = 1
    modelfile = arg['output']+".d2v"
    # trains the model
    if train:
        for epoch in range(10):
            print 'epoch:', epoch
            articles.permute()
            model.train(articles)
        model.save(modelfile)

    model.load(modelfile)
    print "Similaridade por palavra : transit"
    print model.most_similar("transit")

    print ()
    print "Similaridade por documento:"
    st = "Houve uma batida na rua Alberto Cintra 322 entre um carro e um caminhao. Transito intenso "
    new_st = clean_text(st, 0)
    print new_st
    new_doc_vec = model.infer_vector(new_st)
    best = model.docvecs.most_similar([new_doc_vec])
    print best

    #for i in best:
    #    j = i[0]
    #    print j

    print "LEN DOCVECS:" + str(len(model.docvecs))

    Method = 0

    vecs = []
    if Method:
        n_kmeans = 15
        Kmeans(n_kmeans,model)
    else:
        eps = 0.1
        byDBSCAN(eps,model)



