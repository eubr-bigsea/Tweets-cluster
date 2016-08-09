# About

This project aims to conduct some experiments with clustering tweet. There are three approaches, the first using Doc2vec, the second is using tf-idf and the last is using Word2vec.

# Dataset

- On the file *id_tweets_without_clean.txt*  are 14645 tweets without any clean:
    * id_0 to id_8745 are about several subjects (8746 tweets or 59,7%)
	* id_8756 to id_17490 are about traffic ( 5900 tweets or 40,3%)
- On the file *id_tweets_pos_mine.txt*  are 13884 tweets without any clean:
    * id_0 to id_8745 are about several subjects (7871 tweets or 56,69%)
	* id_8756 to id_17490 are about traffic ( 6013 tweets or 43,31%)
- On the file *id_tweets_pos_mixer.txt*  are 14220 tweets without any clean:
    * id_0 to id_8745 are about several subjects (8155 tweets or 57,35%)
	* id_8756 to id_17490 are about traffic ( 6065 tweets or 42,65%)

# Instructions to Run the first approach (Doc2vec)

- First execute: `$ sh 1_script_dataCleanup.sh`
- Second, execute: `$ sh 2_script_run.sh`
- At the end, it will create a file on "data_out" folder with the clustering performed 

# Instructions to Run the second approach (TF-IDF)

- First execute: `$python tf_idf.py`


# Instructions to Run the third approach (Word2vec)

- First execute: `$python word2vec.py`



# Examples 

## Doc2vec

```sh
Testing... (14220 tweets(6065 traffic) and 8 clusters) 
Cluster 0:
	Num itens: 3509	(24.67%)
	Traffic itens/Cluster doc itens: 36.02%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.36	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.21	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.26
Cluster 1:
	Num itens: 2169	(15.25%)
	Traffic itens/Cluster doc itens: 46.06%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.46	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.16	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.24
Cluster 2:
	Num itens: 2401	(16.88%)
	Traffic itens/Cluster doc itens: 43.27%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.43	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.17	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.25
Cluster 3:
	Num itens: 1078	(7.58%)
	Traffic itens/Cluster doc itens: 44.99%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.45	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.08	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.14
Cluster 4:
	Num itens: 629	(4.42%)
	Traffic itens/Cluster doc itens: 45.95%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.46	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.05	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.09
Cluster 5:
	Num itens: 625	(4.39%)
	Traffic itens/Cluster doc itens: 43.04%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.43	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.04	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.08
Cluster 6:
	Num itens: 2565	(18.03%)
	Traffic itens/Cluster doc itens: 42.50%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.42	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.18	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.25
Cluster 7:
	Num itens: 1248	(8.77%)
	Traffic itens/Cluster doc itens: 50.48%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.50	# Relação:(Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.10	# Relação: (Tweets relevantes que foram e recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.17
```

## TF-IDF
```sh
Clustering by TF-IDF
Converting to vectors...
Clustering in 8 groups...
Measuring...
Homogeneity: 0.144
Completeness: 1.000
V-measure: 0.251
Adjusted Rand-Index: 0.000
Silhouette Coefficient: 0.020
Total de 14224 documents
Testing...
Cluster 0:
	Num itens: 507	(3.56%)
	Traffic itens/Cluster doc itens: 0.00%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.00
Cluster 1:
	Num itens: 2309	(16.23%)
	Traffic itens/Cluster doc itens: 16.08%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.990	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.377	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.55
Cluster 2:
	Num itens: 8451	(59.41%)
	Traffic itens/Cluster doc itens: 21.87%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.368	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.513	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.43
Cluster 3:
	Num itens: 276	(1.94%)
	Traffic itens/Cluster doc itens: 0.00%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.00
Cluster 4:
	Num itens: 1071	(7.53%)
	Traffic itens/Cluster doc itens: 1.16%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.154	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.027	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.05
Cluster 5:
	Num itens: 519	(3.65%)
	Traffic itens/Cluster doc itens: 0.06%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.017	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.001	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.00
Cluster 6:
	Num itens: 842	(5.92%)
	Traffic itens/Cluster doc itens: 1.72%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.290	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.040	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.07
Cluster 7:
	Num itens: 249	(1.75%)
	Traffic itens/Cluster doc itens: 1.75%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 1.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.041	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.08
```
## Word2vec

```sh

Cluster 0:
	Num itens: 4396	(30.91%)
	Traffic itens/Cluster doc itens: 4.34%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.140	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.102	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.12
Cluster 1:
	Num itens: 3148	(22.13%)
	Traffic itens/Cluster doc itens: 19.51%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.882	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.458	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.60
Cluster 2:
	Num itens: 889	(6.25%)
	Traffic itens/Cluster doc itens: 5.53%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.885	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.130	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.23
Cluster 3:
	Num itens: 3400	(23.90%)
	Traffic itens/Cluster doc itens: 1.53%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.064	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.036	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.05
Cluster 4:
	Num itens: 400	(2.81%)
	Traffic itens/Cluster doc itens: 2.73%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.970	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.064	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.12
Cluster 5:
	Num itens: 422	(2.97%)
	Traffic itens/Cluster doc itens: 0.02%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.007	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.000	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.00
Cluster 6:
	Num itens: 887	(6.24%)
	Traffic itens/Cluster doc itens: 5.76%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.924	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.135	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.24
Cluster 7:
	Num itens: 682	(4.79%)
	Traffic itens/Cluster doc itens: 3.22%	# Porcentagem de tweets de transito em relacao a quantidade global
	Precision: 0.672	# Relação: (Tweets relevantes que foram recuperados)/(tweets recuperados)
	Recall: 0.076	# Relação: (Tweets relevantes que foram recuperados)/(tweets relevantes)
	F-Measure (harmonic avg): 0.14

```


