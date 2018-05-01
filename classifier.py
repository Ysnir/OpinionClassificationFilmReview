#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 19:40:08 2018

Opinion classifier for movie review
Must be launched with a directory named data containing files named :
dataset.csv and label.csv to fit the chosen model :

@author: Yann Desmarais
"""
#Imports
import csv
from nltk.tokenize import RegexpTokenizer
import numpy           
import pandas as p
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

corpus_path = input()
print("Path for the data to classify : ", corpus_path)

#Load corpus from a csv file into a python list
corpus = []
with open(corpus_path) as csvfile:
    spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
    for row in spamreader:
        corpus.append(row)
        
#tokenization of the words removing non-alphanumeric character
tokenizer = RegexpTokenizer(r'\w+')

tokenized_corpus = []

for row in corpus:
    words = tokenizer.tokenize(str(row).lower())
    tokenized_corpus.append(words)
    
from nltk.corpus import stopwords
stopwords_corpus = []

stop_words = set(stopwords.words("english"))

#Creating a corpus without the stop words (from nltk list)
for doc in tokenized_corpus:
    words = []
    for word in doc:
        if word not in stop_words:
            words.append(word)
    stopwords_corpus.append(words)
    
#Recreating the stopword-less corpus
documents = []
for doc in stopwords_corpus:
    word = " ".join(doc)
    documents.append(word)
   
#Exporting the brut-text with stopwords filtering into a csv file
new_path = "data/processed.csv"
with open(new_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter='\n', quoting=csv.QUOTE_NONE, quotechar='')
        for doc in documents:
            writer.writerow([doc])

############################ Vectorization ############################

#Importing datasets and label into a Dataframe
label = p.read_table('data/labels.csv', header=None, names=['label'])
stopword = p.read_table('data/stopwords.csv', header=None, names=['review'])
to_predict = p.read_table(new_path, header=None, names=['new_review'])

##to align label and text
label.reset_index(drop=True, inplace=True)
stopword.reset_index(drop=True, inplace=True)
to_predict.reset_index(drop=True, inplace=True)

#concatenate label and text together aligned in a Dataframe & extract features en class separatly
training_corpus = p.concat([stopword.reset_index(drop=True), label.reset_index(drop=True)], axis=1)
FEATURES = training_corpus.review
CLASS = training_corpus.label
NEW_FEATURES = to_predict.new_review

#Vectorizing the text data with bigram included
vect = TfidfVectorizer(ngram_range=(1,2))

#create a matrix with terms in each documents
train_feature_m = vect.fit_transform(FEATURES)
corpus_feature_m = vect.transform(NEW_FEATURES)

############################ Model Fitting ############################

#instanciating and fitting the model on the training data
clf = SVC(kernel='linear', C=1)

clf.fit(train_feature_m, CLASS)

############################ Prediction ############################

#Predicting the class of each documents with the trained classifier
pred_class = clf.predict(corpus_feature_m)

#Saving the result in data/class_pred.csv
numpy.savetxt("data/class_pred.csv", pred_class.astype(int),fmt='%i', delimiter="")