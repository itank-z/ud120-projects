#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "rb"))

print enron_data["SKILLING JEFFREY K"]
print "\n\n"


minm = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]
maxm = enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

for person in enron_data.keys():
	if enron_data[person]["exercised_stock_options"] != 'NaN':
		if minm > enron_data[person]["exercised_stock_options"]:
			minm = enron_data[person]["exercised_stock_options"]
		if maxm < enron_data[person]["exercised_stock_options"]:
			maxm = enron_data[person]["exercised_stock_options"]

print "min", minm
print "max", maxm