#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################

from sklearn import svm
import time
clf = svm.SVC(kernel="rbf", C = 10000)

#features_train = features_train[:len(features_train)/100] 
#labels_train = labels_train[:len(labels_train)/100]

t0 = time.time()
clf.fit(features_train, labels_train)
print "\ntraining time : ",(time.time()-t0)," s"

t0 = time.time()
pred = clf.predict(features_test)
print "\nprediction time : ",(time.time()-t0)," s"

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(labels_test, pred)
print "\naccuracy is ", accuracy

no_chris_emails = 0

for i in pred:
	if(i==1):
		no_chris_emails += 1

print "\nno of emails by chris : ", no_chris_emails

#########################################################


