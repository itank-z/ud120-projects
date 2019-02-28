#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary','deferral_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',\
 'shared_receipt_with_poi', 'expenses', 'other', 'from_this_person_to_poi', 'deferred_income',\
 'long_term_incentive', 'from_poi_to_this_person'] 
 # You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


######## Scaling the features using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(features)
scaler.transform(features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.19, random_state=42)


########## Creating new features using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(features_train)

print "\nExplained variance ratio : ", pca.explained_variance_ratio_
print "\n"

new_features_train = pca.transform(features_train)
new_features_test = pca.transform(features_test)


######### Training the classifier
"""
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, min_samples_split=2)
"""
clf.fit(new_features_train, labels_train)
pred = clf.predict(new_features_test)
"""
print "True  Pred"
n = len(labels_test)
for i in range(n):
	print labels_test[i], "     ", pred[i]
"""
######### Evaluation using precision and recall
from sklearn.metrics import precision_score, recall_score
print "Precision : ", precision_score(labels_test, pred)
print "Recall : ", recall_score(labels_test, pred)

import matplotlib.pyplot as plt


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)