#import the library used to query a website
import json

topics=['CA']
topics1 = ['CA', 'ML', 'NLP', 'AI', 'GT', 'DSA']
topics2 = ['PSP', 'ENA', 'DC', 'SE', 'RA', 'PPL', 'MA2', 'NMP', 'MA3', 'DMS', 'FAL', 'CG', 'CN', 'DAA', 'CD', 'AGT', 'OP', 'BIO', 'COM', 'SI', 'PEC', 'RTS', 'OS', 'DM', 'CO', 'PC', 'AMT', 'FLAT', 'IT', 'CAL', 'PTA', 'CNS', 'AEM', 'AMA', 'PDS', 'MAL', 'SP', 'SS', 'NMC', 'LPE', 'LCS', 'TOC', 'MI', 'SM', 'FA', 'LA', 'NMO', 'DDA', 'PA', 'MLO', 'COP', 'COV', 'CGR', 'PS', 'PR', 'CAR', 'DD', 'BAG', 'NOP', 'MA1', 'SAD', 'FOO']

print("Statistics for Dataset Data 1 Annotated so far")
ctr1 = 0 # Number of segments
ctr2 = 0 # Number of concepts
ctr3 = 0 # Number of words
ctr4 = 0 # Number of courses
for topicname in topics:
	dir_name='./4_Topic/'+topicname+'.json'
	with open(dir_name, 'rt') as f:	
		data = json.load(f)
		for d in data:
#			print("d['topics']: ",d['topics']," size: ",len(d['topics']))
			ctr2 += len(d['topics'])
#			print("d['text']: ",d['text']," size: ",len(d['text']))
			ctr3 += len(d['text'])
		ctr1 += len(data)
print("Number of segments: ", ctr1)
print("Number of concepts: ", ctr2)
print("Number of words: ", ctr3)
print("Number of courses: ", len(topics))

print("Statistics for Dataset Data 1")
ctr1 = 0 # Number of segments
ctr2 = 0 # Number of concepts
ctr3 = 0 # Number of words
ctr4 = 0 # Number of courses
for topicname in topics1:
	dir_name='./4_Topic/'+topicname+'.json'
	with open(dir_name, 'rt') as f:	
		data = json.load(f)
		for d in data:
#			print("d['topics']: ",d['topics']," size: ",len(d['topics']))
			ctr2 += len(d['topics'])
#			print("d['text']: ",d['text']," size: ",len(d['text']))
			ctr3 += len(d['text'])
		ctr1 += len(data)
print("Number of segments: ", ctr1)
print("Number of concepts: ", ctr2)
print("Number of words: ", ctr3)
print("Number of courses: ", len(topics1))

print("Statistics for Dataset Data 2")
ctr1 = 0 # Number of segments
ctr2 = 0 # Number of concepts
ctr3 = 0 # Number of words
ctr4 = 0 # Number of courses
for topicname in topics2:
	dir_name='./4_Topic/'+topicname+'.json'
	with open(dir_name, 'rt') as f:	
		data = json.load(f)
		for d in data:
#			print("d['topics']: ",d['topics']," size: ",len(d['topics']))
			ctr2 += len(d['topics'])
#			print("d['text']: ",d['text']," size: ",len(d['text']))
			ctr3 += len(d['text'])
		ctr1 += len(data)
print("Number of segments: ", ctr1)
print("Number of concepts: ", ctr2)
print("Number of words: ", ctr3)
print("Number of courses: ", len(topics2))

print("Statistics for Annotation 1: Course-relevant")
ctr1 = 0
ctr2 = 0
ctr3 = 0
ctr4 = 0
ctr5 = 0
ctr6 = 0
ctr7 = 0
ctr8 = 0
ctr9 = 0
with open('GS/Course/GS1.txt', 'rt') as f1:
	line1 = f1.readlines()
f1.close()
with open('GS/Course/GS2.txt', 'rt') as f2:
	line2 = f2.readlines() 
f2.close()

for i in range(len(line1)):
	lpart1 = line1[i].strip().split(" ")
	lpart2 = line2[i].strip().split(" ")
#	print(lpart1[1] + "\t" + lpart2[1])
	if lpart1[1] == '1':
		if lpart2[1] == '1':
			ctr1+=1
		elif lpart2[1] == '2':
			ctr2+=1
		else:
			ctr3+=1
	elif lpart1[1] == '2':
		if lpart2[1] == '1':
			ctr4+=1
		elif lpart2[1] == '2':
			ctr5+=1
		else:
			ctr6+=1
	else:
		if lpart2[1] == '1':
			ctr7+=1
		elif lpart2[1] == '2':
			ctr8+=1
		else:
			ctr9+=1

print(str(ctr1) + "\t" + str(ctr2) + "\t" + str(ctr3))
print(str(ctr4) + "\t" + str(ctr5) + "\t" + str(ctr6))
print(str(ctr7) + "\t" + str(ctr8) + "\t" + str(ctr9))

print("Statistics for Annotation 2: Retrieval")
ctr1 = 0
ctr2 = 0
ctr3 = 0
ctr4 = 0
with open('GS/Retrieval/GS1.txt', 'rt') as f1:
	line1 = f1.readlines()
f1.close()
with open('GS/Retrieval/GS2.txt', 'rt') as f2:
	line2 = f2.readlines() 
f2.close()

for i in range(len(line1)):
	lpart1 = line1[i].strip().split(" ")
	lpart2 = line2[i].strip().split(" ")
#	print(lpart1[1] + "\t" + lpart2[1])
	if lpart1[3] == '0':
		if lpart2[3] == '0':
			ctr1+=1
		else:
			ctr2+=1
	else:
		if lpart2[3] == '0':
			ctr3+=1
		else:
			ctr4+=1

print(str(ctr1) + "\t" + str(ctr2))
print(str(ctr3) + "\t" + str(ctr4))