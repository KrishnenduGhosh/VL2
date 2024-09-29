import warnings
warnings.filterwarnings("ignore")
import os
import time
import subprocess
import xml.dom.minidom
import re
import tagme
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.translate import IBMModel1, AlignedSent, Alignment
import spacy
from rank_bm25 import BM25Okapi
import word2vec
import pandas as pd
import json
from stackapi import StackAPI
from xml.dom import minidom
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import scoring
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

def load_wv():
	wrdvec_path = './lib/wrdvecs-text8.bin'
	model = word2vec.load(wrdvec_path)
	wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
	filename = "./lib/finalized_model.sav"
	pickle.dump(model, open(filename, 'wb'))
	del model
	sentence_analyzer = nltk.data.load('tokenizers/punkt/english.pickle')
	return wrdvecs,sentence_analyzer

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row.split("\t")[1])
	return dataset

def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = metrics.accuracy_score(actual, predicted)
		scores.append(accuracy)
	return scores

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return 1.0 / (1.0 + exp(-activation))

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

def process1(text): # Processes xml data to txt format
	return re.sub('<[^<]+>', "", text).replace('\n','').replace('#','')
	
def process2(text): # Processes xml data to txt format
	return text.replace('><',',').replace('<','').replace('>','')

def get_wv(line,wrdvecs,sentence_analyzer): # 
	sentenced_text = sentence_analyzer.tokenize(line)
	vecr = CountVectorizer(vocabulary=wrdvecs.index)
	sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
	wv = [0] * 200
	for sv in sentence_vectors:
		for i in range(len(sv)):
			wv[i] += sv[i]
	for i in range(len(wv)):
			wv[i] = round(float(wv[i]/len(sentence_vectors)),2)
	return wv

def get_f1(line1,line2): # Feature 1,2,3: Word n-gram (n = 1,2,3) overlap
	token1 = word_tokenize(line1)
	token2 = word_tokenize(line2)
	qgram1 = len(list(ngrams(token1,1)))
	qgram2 = len(list(ngrams(token1,2)))
	qgram3 = len(list(ngrams(token1,3)))
	onegram = len([value for value in list(ngrams(token1,1)) if value in list(ngrams(token2,1))])
	twogram = len([value for value in list(ngrams(token1,2)) if value in list(ngrams(token2,2))])
	threegram = len([value for value in list(ngrams(token1,3)) if value in list(ngrams(token2,3))])
	gram1 = 0.0
	gram2 = 0.0
	gram3 = 0.0
	if qgram1 != 0.0:
		gram1 = round(float(onegram/qgram1),4)
	if qgram2 != 0.0:
		gram2 = round(float(twogram/qgram2),4)
	if qgram3 != 0.0:
		gram3 = round(float(threegram/qgram3),4)
	f = "1:"+str(gram1)+",2:"+str(gram2)+",3:"+str(gram3)
	return f

def get_f2(line1,line2,corpus): # Feature 4: BM25 score
	bm25_score = 0.0
	tokenized_corpus = [doc.split(" ") for doc in corpus]
	bm25 = BM25Okapi(tokenized_corpus)
	tokenized_query = line1.split(" ")
	doc_scores = bm25.get_scores(tokenized_query)
	for i in range(len(corpus)):
		if line2 == corpus[i]:
			bm25_score=doc_scores[i]
	f = "4:"+str(round(bm25_score,4))
	return f

def get_f4(line1,line2): # Feature 8: Noun overlap
	nouns1 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line1))):
		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
			nouns1.append(word)
	nouns2 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line2))):
		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
			nouns2.append(word)
	no = [value for value in nouns1 if value in nouns2]
	f = "8:0.0"
	if len(nouns1) != 0:
		f = "8:"+str(round(float(len(no)/len(nouns1)),4))
	return f

def get_f5(line1,line2): # Feature 9: Verb overlap
	verbs1 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line1))):
		if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
			verbs1.append(word)
	verbs2 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line2))):
		if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
			verbs2.append(word)
	vo = [value for value in verbs1 if value in verbs2]
	f = "9:0.0"
	if len(verbs1) != 0:
		f = "9:"+str(round(float(len(vo)/len(verbs1)),4))
	return f

def get_f6(line1,line2): # Feature 10: Dependency pair overlap
	nlp = spacy.load("en_core_web_sm")
	dep1 = []
	for token in nlp(line1):
		dep1.append(token.text+"<"+token.dep_+"<"+token.head.text)
	dep2 = []
	for token in nlp(line2):
		dep2.append(token.text+"<"+token.dep_+"<"+token.head.text)
	dpo = [value for value in dep1 if value in dep2]
	f = "10:0.0"
	if len(dep1) != 0:
		f = "10:"+str(round(float(len(dpo)/len(dep1)),4))
	return f

def get_f7(line1,line2): # Feature 11: Named entity overlap
	ne1=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str(line1))))
	ne2=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str(line2))))
	neo = [value for value in ne1 if value in ne2]
	f = "11:0.0"
	if len(ne1) != 0:
		f = "11:"+str(round(float(len(neo)/len(ne1)),4))
	return f

def get_f31(line1,line2,wrdvecs,sentence_analyzer): # Feature 10
	wv1 = get_wv(line1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(line2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = "5:"+str(round(cosim,4))
	return f

def get_f32(line1,line2,wrdvecs,sentence_analyzer): # Feature 11
	st = PorterStemmer()
	sline1 = ''
	sline2 = ''
	for w in line1.split(" "):
		sline1 = sline1 + ' ' + st.stem(w)
	for w in line2.split(" "):
		sline2 = sline2 + ' ' + st.stem(w)
	sline1 = sline1.replace('  ',' ')
	sline2 = sline2.replace('  ',' ')
	wv1 = get_wv(sline1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(sline2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = "6:"+str(round(cosim,4))
	return f

def get_f33(line1,line2,wrdvecs,sentence_analyzer): # Feature 12
	sline1 = ''
	sline2 = ''
	for w in line1.split(" "):
		if w not in stopwords.words('english'):
			sline1 = sline1 + ' ' + w
	for w in line2.split(" "):
		if w not in stopwords.words('english'):
			sline2 = sline2 + ' ' + w
	sline1 = sline1.replace('  ',' ')
	sline2 = sline2.replace('  ',' ')
	wv1 = get_wv(sline1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(sline2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = "7:"+str(round(cosim,4))
	return f

def get_feature(line1,line2,corpus,vecr,sentence_analyzer): # generate features for <query,QA pair>
	f1 = get_f1(line1,line2)
	f2 = get_f2(line1,line2,corpus)
	f31 = get_f31(line1,line2,vecr,sentence_analyzer)
	f32 = get_f32(line1,line2,vecr,sentence_analyzer)
	f33 = get_f33(line1,line2,vecr,sentence_analyzer)
	f4 = get_f4(line1,line2)
	f5 = get_f5(line1,line2)
	f6 = get_f6(line1,line2)
	f7 = get_f7(line1,line2)
	feature = str(f1)+","+str(f2)+","+str(f31)+","+str(f32)+","+str(f33)+","+str(f4)+","+str(f5)+","+str(f6)+","+str(f7)
	return feature

def build_index(t): # create corpus for BM25
	corpus = []
	with open('4_Topic/'+t+'.json') as json_file:
		data = json.load(json_file)
		for d in data:
			corpus.append(d['text'])
	return corpus

def dnn(file): # DNN performance
	dataset = load_csv(file)
	for i in range(len(dataset[0])-1):
		str_column_to_float(dataset, i)
	str_column_to_int(dataset, len(dataset[0])-1)
	minmax = dataset_minmax(dataset)
	normalize_dataset(dataset, minmax)

	n_folds = 5
	l_rate = 0.3
	n_epoch = 5
	n_hidden = 5
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

	l_rate = 0.3
	n_epoch = 1
	n_hidden = 5
	predicted = back_propagation(dataset, dataset, l_rate, n_epoch, n_hidden)
	print('len(predicted): %s' % len(scores))
	return predicted

def merge(file2): # merge qQ.txt and GS_Retrieval.txt as qQ.csv
	rel = {}
	feat = {}
	file1 = 'GS/GS_Retrieval.txt' # qid > Qid relevance_score
	f1=open(file1,'r')
	for l1 in f1:
		line1 = l1.strip().split(" ")
		rel[line1[0]+" > "+line1[2]]=line1[3]
	f1.close()
	file2 # qid > Qid feats
	f2=open(file2,'r')
	for l2 in f2:
		line2 = l2.strip().split(" ")
		feat[line2[0]+" > "+line2[2]]=line2[3]
	f2.close()
	filew = file2.replace(".txt",".csv")
	fw=open(filew,'r')
	for r,f in zip(rel.keys(),feats.keys()):
		fw.write(feats[f]+','+rel[r])
	fw.close()

def create_qQ(t): #
	corpus = build_index(t)
	wrdvecs,sentence_analyzer = load_wv()

	qQlist = []
	f=open('GS/GS_Retrieval.txt','r')
	for l in f:
		lpart = l.strip().split(" ")
		qQlist.append(lpart[2])
	f.close()

	if not os.path.isfile('8_Retrieval/qQ.csv'):
		with open('7_Course/'+t+'.json') as json_file:
			data = json.load(json_file)
			for d in data:
				idd = d['id']
				for con in d['course']:
					ctr=0
					lineq=con
					fQ=open('QA/QA_f.txt','r')
					for lQ in fQ:
						Qpart=lQ.strip().split("\t")
						key = qqpart[0]+'_'+qqpart[1]+'_'+Qpart[0]
						if key in qQlist:
							lineQ=Qpart[1]
							feats=get_feature2(lineq,lineQ,corpus,wrdvecs,sentence_analyzer)
							fw=open('8_Retrieval/qQ.csv','a')
							fw.write(feats+',1\n')
							fw.close()
							fw=open('8_Retrieval/qQ.txt','a')
							fw.write(qqpart[0]+'_'+qqpart[1]+' > '+key+' 1\n')
							fw.close()
						if key not in qQlist and ctr < 10:
							lineQ=Qpart[1]
							feats=get_feature2(lineq,lineQ,corpus,wrdvecs,sentence_analyzer)
							fw=open('8_Retrieval/qQ.csv','a')
							fw.write(feats+',0\n')
							fw.close()
							fw=open('8_Retrieval/qQ.txt','a')
							fw.write(qqpart[0]+'_'+qqpart[1]+' > '+key+' 0\n')
							fw.close()
						ctr+=1
					fQ.close()
	print("qQ.txt and qQ.csv stored ....")

def retrieve(): # initial retrieval
	topics = ['AI','CA', 'DSA','GT','ML','NLP']
	for t in topics:
		print(t)
		create_qQ(t)

	#merge('GS_Retrieval.txt')
	dnn('8_Retrieval/qQ.csv')
	#print("DNN executed ....")
	# write > write top 10 QA pairs > RT.txt
	rlist = []
	f=open('8_Retrieval/qQ.txt','r')
	i = 0
	for line in f:
		lpart = line.strip().split(" ")
		if lpart[2] not in rlist and str(predicted[i]) == '1':
			fw=open('8_Retrieval/RT1.txt','a')
			fw.write(lpart[0]+' > '+lpart[2]+' '+str(predicted[i])+" 1 RT\n")
			fw.close()
			rlist.append(lpart[2])
		if lpart[2] not in rlist and str(predicted[i]) == '0':
			fw=open('8_Retrieval/RT1.txt','a')
			fw.write(lpart[0]+' > '+lpart[2]+' '+str(predicted[i])+" 0 RT\n")
			fw.close()
			rlist.append(lpart[2])
		i+=1
	print(i)
	f.close()

	line_dict = {}
	label_dict = {}
	fr=open('8_Retrieval/RT1.txt','r')
	for line in fr:
		lpart = line.strip().split(" ")
		temp_line = []
		temp_label = []
		if lpart[0] in line_dict.keys():
			temp_line = line_dict[lpart[0]]
			temp_line.append(line.strip())
			line_dict[lpart[0]] = temp_line
		else:
			temp_line = []
			temp_line.append(line.strip())
			line_dict[lpart[0]] = temp_line
		if lpart[0] in label_dict.keys():
			temp_label = label_dict[lpart[0]]
			temp_label.append(lpart[3])
			label_dict[lpart[0]] = temp_label
		else:
			temp_label = []
			temp_label.append(lpart[0])
			label_dict[lpart[0]] = temp_label
	fr.close()

	for kk in line_dict.keys():
		temp_line = line_dict[kk]
		temp_label = label_dict[kk]
		clist = [temp_line for _,temp_line in sorted(zip(temp_label,temp_line))]
		clist.reverse()
		for cline in clist:
			fw=open('8_Retrieval/RT.txt','a')
			fw.write(cline+"\n")
			fw.close()

def sup_s(Qi,Q_init,Qis): # support scoring for a Q in Q_init
	QQ = []
	QQs = []
	Qfs = float(Qis)
	fr2 = open('QA/BM25_Q.txt','r')
	for line in fr2:
		lpart = line.strip().split("\t")
		if lpart[0] == Qi:
			Q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
			QQ = Q.split(", ")
			Qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
			QQs = Qs.split(", ")
			break
	fr2.close()

	temp_s = 0.0
	for q in Q_init:
		for i in range(0,len(QQ)):
			if q == QQ[i]:
				temp_s += QQs[i]
	Qfs *= temp_s
	return Qfs

def cum_s(Qi,Q_init,Qis): # support scoring for a Q in Q_init
	QQ = []
	QQs = []
	Qfs = Qis
	fr2 = open('QA/BM25_Q.txt','r')
	for line in fr2:
		lpart = line.strip().split("\t")
		if lpart[0] == Q and Q in Q_init:
			Q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
			QQ = Q.split(", ")
			Qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
			QQs = Qs.split(", ")
			continue
	fr2.close()

	cQQs = []
	for i in range(0,len(QQ)):
		temps = 0.0
		fr2 = open('QA/BM25_Q.txt','r')
		for line in fr2:
			lpart = line.strip().split("\t")
			if lpart[0] == QQ[i] and QQ[i] in Q_init:
				cQs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
				cum_Qs = cQs.split(", ")
				for s in cum_Qs:
					temps += s
				break
		cQQs[i] = temps
		fr2.close()

	temp_s = 0.0
	for i in range(0,len(QQ)):
		if q == QQ[i]:
			temp_s += cQQs[i]
	Qfs *= temp_s
	return Qfs

def cscore(Q_init,Q_init_s): # cumulative scoring for all Q_init
	Q_final = []
	Q_final_s = []
	for i in range(0,len(Q_init)):
		Qi = Q_init[i]
		Qis = Q_init_s[i]
		Qfs = sup_s(Qi,Q_init,Qis)
		#Qfs = cum_s(Qi,Q_init,Qis)
		Q_final.append(Qi.replace("\'",""))
		Q_final_s.append(Qis.replace("\'",""))
	return  Q_final, Q_final_s

def rerank(): # reranking
	fr1 = open('QA/BM25_q.txt','r')
	for line1 in fr1:
		flpart = line1.strip().split(" ")
		linepart = flpart[20:]
		line = " ".join(linepart)
		lpart = line.strip().split("\t")
		q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
		Q_init = q.split(", ")
		qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
		Q_init_s = qs.split(", ")
		Q_final, Q_final_s = cscore(Q_init,Q_init_s)
		#fw=open('QA/RR_q.txt','a')
		#fw.write(str(lpart[0]).replace(" ","_")+"\t"+str(Q_final)+"\t"+str(Q_final_s)+"\n")
		#fw.close()
		for i in range(0,len(Q_final)):
			fw=open('8_Retrieval/RR.txt','a')
			fw.write(str(lpart[0]).replace(" ","_")+" > "+lpart[0].replace(" ","_")+"_"+str(Q_final[i]).replace("\'","")+" 1 "+str(Q_final_s[i])+" RR\n")
			fw.close()

def evaluate(): # evaluate retrieval, reranking models
	cmd1 = 'trec_eval/trec_eval -M10 10_Retrieval/GS_Retrieval.txt 10_Retrieval/RT.txt'
	cmd2 = 'trec_eval/trec_eval -M10 10_Retrieval/GS_Retrieval.txt 10_Retrieval/RR.txt'
	os.system(cmd1)
	os.system(cmd2)

def write_result():
	dir_in1='./6_Retrieved/'
	dir_out='./8_Result/trec_eval/test'
	topics=['AI','DSA','GT','NLP','ML','CA']
	rtdata = []
	for t in topics:
		with open(dir_in1 + t +'.json') as json_file:
			data = json.load(json_file)
			for d in data:
				#qid = d['id'] + ":"
				for i in range(0,len(d['off'])):
					qid = d['id'] + ":" + d['off'][i].replace(" ","_") #
					docs = d['ret'][i]
					scores = d['score'][i]
					for j in range(0,len(docs)):
						docno = docs[j] #
						sim = scores[j] #
						rank = (j+1) #
						rtdata.append(qid + " > " + qid + ":" + docno + " " + str(rank) + " " + str(sim) + " RT")
	rtdata = sorted(set(rtdata))
	for rt in rtdata:
		file1 = open(dir_out + "/RT.txt","a")
		file1.write(rt + "\n")
		file1.close()

def base():
	schema = Schema(id=TEXT(stored=True), text=TEXT(stored=True))
	ix = create_in("index", schema)
	writer = ix.writer()
	dir_in='QA/QA.txt'
	dir_out_path1='QA/QA_final.txt'
	dir_out_path2='QA/BM25_q.txt'
	topics=['AI','DSA','GT','NLP','ML','CA']
	print("Creating index.......")
	fr = open(dir_in,'r')
	for line in fr:
		lpart = line.strip().split("\t")
		if int(lpart[0]) < 25615218:
			writer.add_document(id=str(lpart[0]), text=str(lpart[1]))
	writer.commit()
	fr.close()
	
	rdict = {}
	print("Retrieving segments.......")
	for t in topics:
		print(t)
		with open('7_Course/' + t +'.json') as json_file:
			data = json.load(json_file)
			for d in data:
				for con in d['course']:
					with ix.searcher() as s:
						q1 = QueryParser("text", ix.schema).parse(con)
						results = s.search(q1, limit = 20)
						#print("len(results): ",len(results))
						rlist = []
						for r in results:
							#print("len(results): ",len(results))
							#print(r['id'] + ' ' + r['text'])
							rlist.append(r['id'])
							fw = open(dir_out_path1,'a')
							fw.write(r['id'] + ' ' + r['text'] + "\n")
							fw.close()
					fw = open(dir_out_path2,'a')
					fw.write(d['id']+'_'+con.replace(' ','_')+'\t'+str(rlist)+'\n')
					fw.close()

def cleanhtml(raw_html):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', raw_html)
	return cleantext

def read():
	f = open('QA/Posts.txt','r')
	for line in f:
		myxml = minidom.parseString(line)
		rows = myxml.getElementsByTagName("row")
		PostTypeId = rows[0].attributes['PostTypeId'].value
		if PostTypeId == '1':
			qid = rows[0].attributes['Id'].value
			question = rows[0].attributes['Title'].value + " " + rows[0].attributes['Body'].value
			question = cleanhtml(question).strip().replace("\n","").replace("\r","").replace("\t","").replace("  "," ")
			fw = open('QA/QA.txt','a')
			fw.write(qid + "\t" + cleanhtml(question).replace("\t","") + "\n")
			fw.close()
	f.close()

def listToString(s):
	str1 = ""
	for ele in s:
		str1 += ele
	return str1

def get_sim(line1,line2):
	token1 = word_tokenize(line1)
	token2 = word_tokenize(line2)
	qgram1 = len(list(ngrams(token1,1)))
	qgram2 = len(list(ngrams(token1,2)))
	onegram = len([value for value in list(ngrams(token1,1)) if value in list(ngrams(token2,1))])
	gram1 = 0.0
	if qgram1 != 0.0:
		gram1 = round(float(onegram/qgram1),4)
	return gram1

def whoosh_Q(l_lim,u_lim):
	Qdlist = []
	fr = open('QA/BM25_Q.txt','r')
	for line in fr:
		lpart = line.split("\t")
		Qdlist.append(lpart[0])
	fr.close()

	fq=open('QA/QA_f.txt','r')
	i=0
	for lq in fq:
		qpart=lq.strip().split(" ")
		if i >= l_lim and i < u_lim and qpart[0] not in Qdlist:
			Qlist = []
			Slist = []
			print(str(i)+'\t'+qpart[0])
			fQ=open('QA/QA_f.txt','r')
			for lQ in fQ:
				Qpart=lQ.strip().split(" ")
				if lq != lQ:
					line1 = ' '.join(qpart[1:len(qpart)])
					line2 = ' '.join(Qpart[1:len(Qpart)])
					Slist.append(get_sim(line1,line2))
					Qlist.append(Qpart[0])
			fQ.close()
			sQlist = [Qlist for _,Qlist in sorted(zip(Slist,Qlist),reverse=True)]
			Slist.sort(reverse=True)
			sSlist = Slist[0:10]
			sQlist = sQlist[0:10]
			fw = open('QA/BM25_Q.txt','a')
			fw.write(qpart[0]+'\t'+str(sQlist)+'\t'+str(sSlist)+'\n')
			fw.close()
		i+=1
	fq.close()

def merge_GS(): # merge QA/<subject>.txt as BM25_Retrieval.txt and GS_Retrieval.txt
	clist = []
	f1=open('QA/BM25_q.txt','r')
	for line1 in f1:
		lpart = line.strip().split("\t")
		QQ = lpart[1].replace("[","").replace("]","").replace("\'\'","")
		Qlist = QQ.split(", ")
		if len(Qlist) > 1:
			ctr = 0
			for Q in Qlist:
				if lpart[0]+'_'+Q.replace("\'","") not in clist:
					if ctr < 5:
						if randrange(1,10) > -1:
							fw1 = open('GS/GS_Retrieval.txt','a')
							fw1.write(lpart[0]+' > '+lpart[0]+'_'+Q.replace("\'","")+' 1\n')
							fw1.close()
							clist.append(lpart[0]+'_'+Q.replace("\'",""))
							ctr+=1
					elif ctr < 10:
						if randrange(1,10) > -1:
							fw1 = open('GS/GS_Retrieval.txt','a')
							fw1.write(lpart[0]+' > '+lpart[0]+'_'+Q.replace("\'","")+' 1\n')
							fw1.close()
							clist.append(lpart[0]+'_'+Q.replace("\'",""))
							ctr+=1
					else:
						if randrange(1,10) > -1:
							fw1 = open('GS/GS_Retrieval.txt','a')
							fw1.write(lpart[0]+' > '+lpart[0]+'_'+Q.replace("\'","")+' 1\n')
							fw1.close()
							clist.append(lpart[0]+'_'+Q.replace("\'",""))
							ctr+=1
	f1.close()

def whoosh_q():
	fr1 = open('QA/BM25_qQ.txt','r')
	for row in fr1:
		flpart = row.strip().split(" ")
		linepart = flpart[20:]
		line = " ".join(linepart)
		lpart = row.split("\t")
		llpart = lpart[0].split("_")
		concept = llpart[-1]
		Qlist = []
		Slist = []
		#print("query: ",concept)
		fr2 = open('QA/QA_f.txt','r')
		for QQ in fr2:
			Qpart = QQ.strip().split(" ")
			question = " ".join(Qpart[1:])
			#print("question: ",question)
			Slist.append(get_sim(concept,question))
			Qlist.append(Qpart[0])
		fr2.close()
		sQlist = [Qlist for _,Qlist in sorted(zip(Slist,Qlist),reverse=True)]
		Slist.sort(reverse=True)
		sSlist = Slist[0:10]
		sQlist = sQlist[0:10]
		fw = open('QA/BM25_q.txt','a')
		fw.write(concept+'\t'+str(sQlist)+'\t'+str(sSlist)+'\n')
		fw.close()
	fr1.close()

def main(): #main for menu
	start_time = time.time()

	#read() # post to QA
	#base() # relevant QAs for qs
	whoosh_q()
	#whoosh_Q(down,up) # relevant QAs for Qs

	#merge_GS()
	#retrieve()
	#rerank()
	#evaluate()
	print("Exuction time: ", (time.time() - start_time))

if __name__ == '__main__':
	#main()
	main()
