import warnings
warnings.filterwarnings("ignore")
import time
import os
import random
import json
import re
from yattag import Doc, indent

def process(mystring):
	#re.sub('^[a-zA-Z0-9_-]+$',"",mystring)
	return re.sub('[^a-zA-Z0-9 ]','',mystring)

def generate_augment(topic): # combining the features with gold-standard

	aug = {}
	infile1='8_Retrieval/RR.txt'
	f1=open(infile1,'r')
	for line in f1:
		lpart=line.strip().split(" ")
		if lpart[3] == '1' and lpart[0].startswith(topic):
			llpart=lpart[2].split("_")
			id = "_".join(llpart[0:-1])
			augm = llpart[-1]
			if id in aug.keys():
				alist = aug[id]
				alist.append(augm)
				aug[id] = alist
			else:
				alist = []
				alist.append(augm)
				aug[id] = alist
	f1.close()
	#print(len(aug.keys()))

	augm = {} # 38478 > If $L$ is a matrix that represents real physical quantity, why is $L^2$ non-negative real physical quantity? In my textbook, it says that when $L$ is a matrix that represents real($\mathbb{R}$) physical quantity, $L^2$ represents non-negative real physical quantity. What would be the proof of this? #quantum-mechanics observables
	infile1='QA/QA_f.txt'
	f1=open(infile1,'r')
	for line in f1:
		lpart=line.strip().split("\t")
		augm[lpart[0]] = lpart[1]
	f1.close()

	text_list = {}
	infile='4_Topic/'+topic+'.json'
	with open(infile) as json_file:
		data = json.load(json_file)
		for d in data:
			id = d['id']
			ptext = process(d['text'])
			text_list[id] = ptext

	dump_file['examples']=[]
	infile='7_Course/'+topic+'.json'
	with open(infile) as json_file:
		data = json.load(json_file)
		for d in data:
			id = d['id']
			text = text_list[id]
			c_list = d['course']
			section = d['id']+"_"+m.replace(" ","_")
			if section in augid.keys() and augid[section] in augm.keys():
			aug_list = augm[aug[section]]
			dump_file['examples'].append({"id":id_no,"text":data,"course":c_list,"aug":aug_list})
	filename=os.path.join("8_Retrieval/"+t+".json")
	with open(filename, 'w') as outfile:  
		json.dump(dump_file['examples'], outfile, indent=4)

def main(): #main for menu
	start_time = time.time()
	topics = ['AI','CA', 'DSA','GT','ML','NLP']
	for s in topics:
		print(s)
		generate_augment(s)
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	main()
