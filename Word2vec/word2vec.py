#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from gensim.models import word2vec
from sklearn.cluster import KMeans
import time


import sys
reload(sys)
sys.setdefaultencoding("utf-8")


def read_sentences():
    documents = [[]]
    for line in open("./data_in/id_tweet_pos_mixer.txt", "r"):
        tokens = line.split(' ')
        sentence = []
        for i in xrange(1,len(tokens)-1):
            sentence.append(tokens[i])
        documents.append(sentence)

    return documents[1:]







if __name__ == "__main__":
    sentences = read_sentences()

    # Import the built-in logging module and configure it so that Word2Vec
    # creates nice output messages
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # Set values for various parameters
    num_features = 300    # Word vector dimensionality
    min_word_count = 40   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words

    # Initialize and train the model (this will take some time)
    print "Training model..."
    model = word2vec.Word2Vec(sentences, workers=num_workers, size=num_features, min_count = min_word_count, window = context, sample = downsampling)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.

    model.init_sims(replace=True)

    # It can be helpful to create a meaningful model name and
    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "tweets"
    model.save(model_name)

    print model.most_similar("transit")

    start = time.time() # Start time
    word_vectors = model.syn0
    num_clusters = 8

    # Initalize a k-means object and use it to extract centroids
    kmeans_clustering = KMeans( n_clusters = num_clusters )
    idx = kmeans_clustering.fit_predict( word_vectors )

    # Get the end time and print how long the process took
    end = time.time()
    elapsed = end - start
    print "Time taken for K Means clustering: ", elapsed, "seconds."

    # Create a Word / Index dictionary, mapping each vocabulary word to
    # a cluster number
    word_centroid_map = dict(zip( model.index2word, idx ))

    # For the first 10 clusters
    words = [ [] for y in range(num_clusters)]
    for cluster in xrange(0, num_clusters):
        #
        # Print the cluster number
        print "\nCluster %d" % cluster
        #
        # Find all of the words for that cluster number, and print them out

        for i in xrange(0, len(word_centroid_map.values())):
            if (word_centroid_map.values()[i] == cluster):
                words[cluster].append(word_centroid_map.keys()[i])
        #print words[cluster]

    num_docs_cluster  = [ 0 for y in range(num_clusters)]       # Number of only traffic docs
    all_docs_cluster = [ 0 for y in range(num_clusters)]       # Docs on each Cluster

    f = open("./data_out/Clusters.txt",'w')
    for line in open("./data_in/id_tweet_pos_mixer.txt", "r"):
        tokens = line.split(' ')
        vec_cluster = [0 for y in range(num_clusters)]
        for i in xrange(1, len(tokens) - 1):
            for y in xrange(0,len(words)):
                if tokens[i] in words[y]:
                    vec_cluster[y]+=1
                    break
        msg= "DOC_"+tokens[0] + " is on Cluster " + str(vec_cluster.index(max(vec_cluster)))
        if int(tokens[0]) > 8745:
            num_docs_cluster[vec_cluster.index(max(vec_cluster))]+=1
        all_docs_cluster[vec_cluster.index(max(vec_cluster))]+=1
        f.write(msg+"\n")
    f.close()

    print "Calculando..."
    for i in xrange(0, num_clusters):
        if num_docs_cluster[i] == 0:
            num_docs_cluster[i] = 0.00001
        print "Cluster " + str(i) + ":"
        print "\tNum itens: %d\t(%.2f%%)"  %  (all_docs_cluster[i], 100*float(all_docs_cluster[i])/len(sentences))
        print "\tTraffic itens/Cluster doc itens: %.2f%%\t# Porcentagem de tweets de transito em relacao a quantidade global" % (100 * float(num_docs_cluster[i]) /len(sentences))
        precision = float(num_docs_cluster[i]) / (all_docs_cluster[i])
        print  "\tPrecision: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)" % precision
        recall = float(num_docs_cluster[i]) / 6065
        print  "\tRecall: %.3f\t# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)" % recall
        print "\tF-Measure (harmonic avg): %.2f" % (float((2*precision*recall)) / (precision + recall))

        



