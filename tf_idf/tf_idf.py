#!/usr/bin/env python
# -*- coding: utf-8 -*-

#--------------------------------------------------------------#
#                                                              #
#  Author: Lucas Miguel S. Ponce  (lucasmsp@gmail.com)         #
#                                                              #
#--------------------------------------------------------------#

import nltk as nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

from dataCleanup import *

def mount_text():
    documents = []
    for line in open("./data_in/tweet.txt", "r"):
        documents.append(line.split(' ', 1)[1])

    return documents

def  measuring(K,clusters):
    num_docs_cluster  = [ 0 for y in range(K)]       # Number of only traffic docs
    list_docs_cluster = [[] for y in range(K)]       # Docs on each Cluster
    indice = 0

    for line in open("./data_in/tweet.txt", "r"):
        id = line.split(' ')[0]
        list_docs_cluster[clusters[indice]].append("DOC_" + id)
        if int(id) > 8745:
            num_docs_cluster[clusters[indice]] = num_docs_cluster[clusters[indice]] + 1
        indice += 1

    #print list_docs_cluster
    #print num_docs_cluster

    print "Calculando..."
    for i in xrange(0, K):
        if num_docs_cluster[i] == 0:
            num_docs_cluster[i] = 0.00001
        print "Cluster " + str(i) + ":"
        print  "\tNum itens:\t" + str(len(list_docs_cluster[i])) + "\t("+str(float(len(list_docs_cluster[i]))/len(clusters))+"%)"
        print  "\tTraffic itens/Total doc itens:\t" + str(float(num_docs_cluster[i]) /len(clusters))
        precision = float(num_docs_cluster[i]) / len(list_docs_cluster[i])
        print  "\tPrecision:\t" + str(precision) + "\t\t# (relevantes dos recuperados)/recuperados   ==> Traffic itens/Cluster doc itens"
        recall = float(num_docs_cluster[i]) / 8745
        print  "\tRecall:\t" + str(recall) + "\t\t# (relevantes e recuperados)/relevantes"
        print "\tF (harmonic avg):\t" + str(float((2*precision*recall)) / (precision + recall))

    return documents


if __name__ == "__main__":
    print "#\tClustering by TF-IDF"
    documents = mount_text()

    print "Converting to vectors..."
    #vectorize the text i.e. convert the strings to numeric features
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3))
    X = vectorizer.fit_transform(documents)


    true_k = 3
    print "Clustering in " + str(true_k) + " groups..."
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    km.fit_predict(X)

    print "Measuring..."

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(documents, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(documents, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(documents, km.labels_))
    print("Adjusted Rand-Index: %.3f"
          % metrics.adjusted_rand_score(documents, km.labels_))
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(X, km.labels_, sample_size=1000))
    #print top terms per cluster clusters

    clusters = km.labels_.tolist()  # 0 iff term is in cluster0, 1 iff term is in cluster1 ...  (lista de termos)
    #print "Lista de termos pertencentes aos clusters " + str(clusters)
    print "Total de " + str(len(km.labels_)) + " documents"

    #Example to get all documents in cluster 0
    #cluster_0 = np.where(clusters==0) # don't forget import numpy as np
    #print cluster_0
    #cluster_0 now contains all indices of the documents in this cluster, to get the actual documents you'd do:
    #X_cluster_0 = documents[cluster_0]
    terms = vectorizer.get_feature_names()

    #print terms
    measuring(true_k,clusters)
