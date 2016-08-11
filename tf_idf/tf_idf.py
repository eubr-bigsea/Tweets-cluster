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
from sklearn.cluster import DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn import metrics
import numpy as np

import sys
reload(sys)
sys.setdefaultencoding("utf-8")


from dataCleanup import *

def mount_text():
    documents = []
    for line in open("./data_in/id_tweet_pos_mixer.txt", "r"):
        documents.append(line.split(' ', 1)[1])

    return documents

#
#  K-MEANS
#

def  measuring_kmeans(K,clusters):
    num_docs_cluster  = [ 0 for y in range(K)]       # Number of only traffic docs
    list_docs_cluster = [[] for y in range(K)]       # Docs on each Cluster
    indice = 0

    for line in open("./data_in/id_tweet_pos_mixer.txt", "r"):
        id = line.split(' ')[0]
        list_docs_cluster[clusters[indice]].append("DOC_" + id)
        if int(id) > 8745:
            num_docs_cluster[clusters[indice]] = num_docs_cluster[clusters[indice]] + 1
        indice += 1

    #print list_docs_cluster
    #print num_docs_cluster

    print "Testing..."
    for i in xrange(0, K):
        if num_docs_cluster[i] == 0:
            num_docs_cluster[i] = 0.00001
        print "Cluster " + str(i) + ":"
        print  "\tNum itens: %d\t(%.2f%%)"  %(len(list_docs_cluster[i]),100*float(len(list_docs_cluster[i]))/len(clusters))
        print  "\tTraffic itens/Cluster doc itens: %.2f%%\t# Porcentagem de tweets de transito em relacao a quantidade global" % (100*float(num_docs_cluster[i]) /len(clusters))
        precision = float(num_docs_cluster[i]) / len(list_docs_cluster[i])
        print  "\tPrecision: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)" % precision
        recall = float(num_docs_cluster[i]) / 6065
        print  "\tRecall: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)" % recall
        print "\tF-Measure (harmonic avg): %.2f" % (float((2*precision*recall)) / (precision + recall))

    return documents



def clustering_by_kmeans(vectorizer, X, true_k):
    print "Clustering in " + str(true_k) + " groups by K-means..."
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=500, n_init=1)
    km.fit_predict(X)

    print "Measuring..."

    print("Homogeneity: %0.3f" % metrics.homogeneity_score(documents, km.labels_))
    print("Completeness: %0.3f" % metrics.completeness_score(documents, km.labels_))
    print("V-measure: %0.3f" % metrics.v_measure_score(documents, km.labels_))  #V-measure is an entropy-based measure which explicitly measures how successfully the criteria of homogeneity and completeness have been satisfied.
    print("Adjusted Rand-Index: %.3f"   % metrics.adjusted_rand_score(documents, km.labels_))
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))
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
    measuring_kmeans(true_k,clusters)

#
#  DBSCAN
#

def measuring_dbscan(K,labels, documents):
    print "Measuring DBSCAN"
    num_docs_cluster  = [ 0 for y in range(K)]       # Number of only traffic docs
    list_docs_cluster = [[] for y in range(K)]       # Docs on each Cluster
    indice = 0

    for line in open("./data_in/id_tweet_pos_mixer.txt", "r"):
        id = line.split(' ')[0]
        #print labels[indice]
        if (labels[indice]>-1):
            list_docs_cluster[labels[indice]].append("DOC_" + id)
            if int(id) > 8745:
                num_docs_cluster[labels[indice]] = num_docs_cluster[labels[indice]] + 1
        indice += 1

    #print list_docs_cluster
    #print num_docs_cluster
    soma_all = 0
    print "Testing..."
    for i in xrange(0, K):
        if num_docs_cluster[i] == 0:
            num_docs_cluster[i] = 0.00001
        print "Cluster " + str(i) + ":"
        print  "\tNum itens: %d\t(%.2f%%)"  %(len(list_docs_cluster[i]),100*float(len(list_docs_cluster[i]))/len(labels))
        soma_all +=len(list_docs_cluster[i])
        print  "\tTraffic itens/Cluster doc itens: %.2f%%\t# Porcentagem de tweets de transito em relacao a quantidade global" % (100*float(num_docs_cluster[i]) /len(labels))
        n_list = len(list_docs_cluster[i])
        if n_list == 0:
            n_list = 0.000001
        precision = float(num_docs_cluster[i]) / n_list
        print  "\tPrecision: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)" % precision
        recall = float(num_docs_cluster[i]) / 6065
        print  "\tRecall: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)" % recall
        print "\tF-Measure (harmonic avg): %.2f" % (float((2*precision*recall)) / (precision + recall))
    print "Total = %d " % soma_all



def clustering_by_dbscan(vectorizer, X, documents):
    print "Clustering vectors by DBSCAN"
    db = DBSCAN(eps=0.99, min_samples=10).fit(X)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    print('Estimated number of clusters: %d' % n_clusters)
    print labels
    measuring_dbscan( n_clusters,labels, documents)




if __name__ == "__main__":
    print "#\tClustering by TF-IDF"
    documents = mount_text()

    print "Converting to vectors..."
    #vectorize the text i.e. convert the strings to numeric features
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=(1,3))
    X = vectorizer.fit_transform(documents)



    true_k = 40

    Method = 1
    if Method:
        clustering_by_kmeans(vectorizer, X, true_k)
    else:
        clustering_by_dbscan(vectorizer, X,documents)



