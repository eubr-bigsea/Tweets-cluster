#!/usr/bin/env python
# -*- coding: utf-8 -*-

#--------------------------------------------------------------#
#                                                              #
#  Author: Lucas Miguel S. Ponce  (lucasmsp@gmail.com)         #
#                                                              #
#--------------------------------------------------------------#

''' IMPORT PACKAGES '''
import argparse
import re
from unicodedata import normalize
import nltk
import sys
import json

from filters import *

# THIS CODE ENSURE THE CORRECT ENCODING (OPTIONAL)
reload(sys)
sys.setdefaultencoding("utf-8")


def remove_perfil(line):
	return re.sub(r'(@(\w|\d)*)', '', line, flags=re.MULTILINE)	



def remove_smiles(line):
	return re.sub(r'(kk+)|((h|a){2,})| ((k|a){2,})', 'rs', line, flags=re.MULTILINE)

def get_id_text(line):
	#print line
	msg = line['text'].replace("\n"," ")
	msg = str(line['_id']) + " " + msg
	return msg

def remove_url(line):
	return re.sub(r'http\S+', ' ', line)


def clean_data(in_data,out_data,type):
	#nltk.download()
	stemmer = nltk.stem.RSLPStemmer()
	stopwords = nltk.corpus.stopwords.words('portuguese')
	f = open(in_data,'r')
	new_f = open(out_data,'w')
	for tweet in f:
		if type:
			line = json.loads(tweet)
			line = get_id_text(line)
		else:
			line = tweet
		tokens = line.split(' ',1)
		line = ''.join(token + " " for token in tokens[1:])
		line = filter_url(line)
		line = filter_punct(line)
		line = filter_accents(line)
		line = filter_charRepetition(line)
		line = remove_smiles(line)																	#Remove: kkk haha


		#line = normalize('NFKD', line.decode('utf-8')).encode('ASCII', 'ignore').lower()  			# Remove: acents
		#line = remove_perfil(line)																	#Remove: @perfil
		#line = remove_url(line)																		#Remove: urls
		#line = remove_smiles(line)																	#Remove: kkk haha
		#line = re.sub('\W+',' ', line ) 															#Remove: symbols

		line = ''.join(token + " " for token in line.split() if (token not in stopwords))			#Stopword Portuguese
		line = ''.join(stemmer.stem(token) + " " for token in line.split() if (len(token)>1))							#Stemming Portuguese
		#print line
		if(len(line.split()) > 1):
			new_f.write(tokens[0]+" " +line+"\n")

	f.close()
	new_f.close()

def clean_text(line,type):
	stemmer = nltk.stem.RSLPStemmer()
	stopwords = nltk.corpus.stopwords.words('portuguese')
	if type:
		line_t = json.loads(line)
		line = get_id_text(line_t)
	
	line = normalize('NFKD', line.decode('utf-8')).encode('ASCII', 'ignore').lower()  			# Remove: acents
	line = remove_perfil(line)  																# Remove: @perfil
	line = remove_url(line)  																	# Remove: urls
	line = remove_smiles(line)  																# Remove: kkk haha
	line = re.sub('\W+', ' ', line)  															# Remove: symbols

	line = ''.join(token + " " for token in line.split() if (token not in stopwords))  # Stopword Portuguese
	line = ''.join(stemmer.stem(token) + " " for token in line.split())  # Stemming Portuguese
		# print line
	if (len(line.split()) > 2):
		return (line + "\n")
	else:
		return ''

if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description='Tweet cleaner')
	parser.add_argument('-i', '--input', required=True, help='path to the input file')
	parser.add_argument('-t', '--type', type=int,  default=0, required=False, help='1 for twitters json, 0 for text file (format: id<space>text)')
	parser.add_argument('-o', '--output', required=True, help='path to the output file')
	arg = vars(parser.parse_args())

	in_data = arg['input']
	out_data = arg['output']
	type = arg['type']

	clean_data(in_data,out_data,type)
