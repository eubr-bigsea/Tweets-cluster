#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
import string
import unicodedata
import sys
# THIS CODE ENSURE THE CORRECT ENCODING (OPTIONAL)
reload(sys)
sys.setdefaultencoding("utf-8")


def filter_punct (ent):
	''' Remove sinais de pontuaçao STRING --> STRING '''
	punct = re.compile('[%s]' % string.punctuation.replace('#', '').replace('@', ''))  #Tags precisam ser mantidas, bem como @ 
	ent = punct.sub('', ent)
	
	return ent

def filter_charRepetition (ent):
	''' Remover caracteres repetidos em excesso, com o "o" em gooooooooooooooooool STRING --> STRING '''

	#print ent
	expRepeticao1 = re.compile('^([rs])\\1')
	expRepeticao2 = re.compile('([rs])\\1$')
	expRepeticao3 = re.compile('([^rs])\\1{1,}')
	expRepeticao4 = re.compile('([\\S\\s])\\1{2,}')
	
	ent = expRepeticao4.sub('\\1\\1', ent)
	ent = expRepeticao3.sub('\\1', ent)
	ent = expRepeticao2.sub('\\1', ent)
	ent = expRepeticao1.sub('\\1', ent)

	return ent
	
def filter_url (ent):
	''' Remove URL STRING --> STRING '''
	urlRef = re.compile("((https?|ftp):[\/]{2}[\w\d:#@%/;\$()~_?\+-=\\\.&]*)")
	
	ent = urlRef.sub('', ent)
	
	return ent

def gen_NGrams(N,text, ignore_stops = True, create_subgrams=True, ngram_sep='', stop_words = []):
	''' Retorna um SET contendo as N-gramas do texto STRING --> SET '''
	NList = [] # start with an empty list
	if N > 1:
		partes = text.split() + (N *[''])
	else:
		partes = text.split()
	# append the slices [i:i+N] to NList
	for i in range(len(partes) - (N - 1) ):
		NList.append(partes[i:i+N])

	result = set()
	for item in NList:
		if create_subgrams:
			list_iterations = xrange(1, N + 1)
		else:
			list_iterations = [N]
		for i in list_iterations:
			stops_found = [x for x in item[0:i] if x in stop_words or x == ""]
			#Ignora N-gramas so com stop words
			dado = ngram_sep.join(item[0:i])
			if ngram_sep.join(stops_found) != dado or ignore_stops == False:
				if dado != ngram_sep:
					result.add(dado)
	return result
	
# Filtra Acentos String --> String
def filter_accents(s):
	return unicodedata.normalize('NFKD', s.decode('utf-8')).encode('ASCII', 'ignore').lower()
	#return ''.join((c for c in unicodedata.normalize('NFD', s.decode('utf-8')) if unicodedata.category(c) != 'Mn'))



def filter_numbers(gramsSet):
	''' Filtra numeros sozinhos Set --> Set '''
	return [item for item in gramsSet if not item.isdigit()]
	
def filter_small_words(gramsSet, min_size):
	''' Filtra termos menores que min_size  set--> Set '''
	return [item for item in gramsSet if len(item)>= min_size]

def __non_empty_container_or_object__(obj):
	if isinstance(obj, dict) or isinstance(obj, tuple) or isinstance(obj, list):
		return len(obj) > 0
	else:
		return True

	
