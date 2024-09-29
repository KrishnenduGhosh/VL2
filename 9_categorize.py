import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import arff, numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler
from skmultilearn.dataset import load_from_arff
from skmultilearn.dataset import load_dataset
from sklearn.datasets import load_svmlight_file
from libsvm.svmutil import *
from sklearn.model_selection import train_test_split
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from skmultilearn.ensemble import RakelD
from skmultilearn.adapt import MLkNN
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics

def get_id(aug):
  augpart = aug.split("#")
  return augpart[0]

def get_feature(line1,line2,corpus): # generate features for augmentation
	f1 = get_f1(line1,line2)
	f2 = get_f2(line1,line2)
	f3 = get_f3(line1,line2)
	f4 = get_f4(line1,line2)
	f5 = get_f5(line1,line2)
	f6 = get_f6(line1,line2)
	f7 = get_f7(line1,line2)
	f8 = get_f8(line1,line2)
	feature = str(f1)+","+str(f2)+","+str(f3)+","+str(f4)+","+str(f5)+","+str(f6)+","+str(f7)+","+str(f8)
	return feature

def extract(t): # Extract features
	if not os.path.isfile('9_Categorization/features.txt'):
		with open('8_Retrieval/' + t +'.json') as json_file:
			data = json.load(json_file)
			for d in data:
				for a in d['aug']:
					idd = get_id(a)
					feats=get_feature(a)
					fw=open('9_Categorization/features.txt','a')
					fw.write(idd+'\t'+feats+'\n')
					fw.close()
	print("Features extracted for course: " + t)

def run(t): # Load data & run classifiers

  #X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train') # not running for only classify3()
  #X_test, y_test, _, _ = load_dataset('emotions', 'test')
  #classify(X_train, y_train,X_test, y_test,1)
	
	X, y = load_svmlight_file('9_Categorization/'+t+'.txt') # running for only classify1()
	X_train = X[:200]
	y_train = y[:200]
	X_test = X[200:]
	y_test = y[200:]
	print("type(X_train): ",type(X_train))
	print("type(y_train): ",type(y_train))
	print("type(X_test): ",type(X_test))
	print("type(y_test): ",type(y_test))
	classify(X_train, y_train,X_test, y_test,2)

def evaluate_multiclass(y_test, pred):
	print("Accuracy: ",metrics.accuracy_score(y_test, pred)) # accuracy
	print("Hamming loss: ",metrics.hamming_loss(y_test, pred)) # hamming loss
	print("Precision score: ",metrics.precision_score(y_test, pred)) # precision score
	print("Recall score: ",metrics.recall_score(y_test, pred)) # recall score
	print("F1 score: ",metrics.f1_score(y_test, pred)) # f1 score
	print("Average_precision_score: ",metrics.average_precision_score(y_test, pred)) # average_precision_score
	print("hinge_loss: ",metrics.hinge_loss(y_test, pred)) # hinge_loss
	print("jaccard_score: ",metrics.jaccard_score(y_test, pred)) # jaccard_score
	print("top_k_accuracy_score: ",metrics.top_k_accuracy_score(y_test, pred)) # top_k_accuracy_score,k=2
	print("zero_one_loss: ",metrics.zero_one_loss(y_test, pred)) # zero_one_loss

def evaluate_multilabel(y_test, pred):
	print("Accuracy: ",metrics.accuracy_score(y_test, pred)) # accuracy
	print("Hamming loss: ",metrics.hamming_loss(y_test, pred)) # hamming loss
	print("Precision score: ",metrics.precision_score(y_test, pred,average='samples')) # precision score
	print("Recall score: ",metrics.recall_score(y_test, pred,average='samples')) # recall score
	print("F1 score: ",metrics.f1_score(y_test, pred,average='samples')) # f1 score
	print("Micro F1 score: ",metrics.f1_score(y_test, pred,average='micro')) # f1 score
	print("macro F1 score: ",metrics.f1_score(y_test, pred,average='macro')) # f1 score
	print("jaccard_score: ",metrics.jaccard_score(y_test, pred,average='samples')) # jaccard_score
	print("zero_one_loss: ",metrics.zero_one_loss(y_test, pred)) # zero_one_loss

def eval(y_test,pred,val):
	if val == 1:
		evaluate_multilabel(y_test, pred)
	else:
		evaluate_multiclass(y_test, pred)

def classify(X_train, y_train,X_test, y_test,val):
	eval(y_test,classify1(X_train, y_train,X_test, y_test),val)
	eval(y_test,classify2(X_train, y_train,X_test, y_test),val)
	#eval(y_test,classify3(X_train, y_train,X_test, y_test),val)
	eval(y_test,classify4(X_train, y_train,X_test, y_test),val)

def classify1(X_train, y_train,X_test, y_test): # Classifier 1: BR
	print("\nClassifier: Binary Relevance\n")

	clf1 = BinaryRelevance(classifier=SVC())
	clf1.fit(X_train, y_train)
	pred1 = clf1.predict(X_test).todense()
	return pred1

def classify2(X_train, y_train,X_test, y_test): # Classifier 2: RAkEL
	print("\nClassifier: RAkEL\n")

	clf2 = RakelD(base_classifier=GaussianNB(),base_classifier_require_dense=[True, True],labelset_size=4)
	clf2.fit(X_train, y_train)
	pred2 = clf2.predict(X_test).todense()
	return pred2

def classify3(X_train, y_train,X_test, y_test): # Classifier 3: Rank SVM
	print("\nClassifier: Rank SVM\n")

	clf3 = SVC()
	clf3.fit(X_train, y_train)
	pred3 = clf3.predict(X_test).todense()
	return pred3

def classify4(X_train, y_train,X_test, y_test): # Classifier 4: MLKNN
	print("\nClassifier: MLkNN\n")

  #parameters = {'k': range(1,11), 's': [0.3,0.5,0.7,0.9]}
  #clf4 = GridSearchCV(MLkNN(), parameters, scoring='f1_macro')
  #print (clf4.best_params_, clf4.best_score_)
	
	clf4 = MLkNN(k=3)
	clf4.fit(X_train, y_train)
	pred4 = clf4.predict(X_test).todense()
	return pred4

def main(): #main for menu
	start_time = time.time()
	topics=['AI','DSA','GT','NLP','ML','CA']
	for t in topics:
		print(t)
		extract(t)
		run(t)
	print("Exuction time: ", (time.time() - start_time))

if __name__ == '__main__':
	main()
