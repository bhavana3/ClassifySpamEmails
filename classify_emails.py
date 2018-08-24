# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 21:00:17 2018

@author: bhavana
"""

from flask import Flask, render_template, request, redirect, url_for, jsonify
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *

import os
import pandas as pd



app = Flask(__name__)

global Classifier
global Vectorizer

mails  = pd.read_csv('spam.csv', encoding = 'latin-1')
columns = ['v1','v2']
messages = mails[columns]


train_feature, test_feature, train_class, test_class = \
    train_test_split(messages['v2'], messages['v1'], \
    train_size=0.75, test_size=0.25)

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_feature)
Classifier.fit(vectorize_text, train_class)

test_feature = test_feature.tolist()
test_class = test_class.tolist()


@app.route('/', methods=['GET'])
def index():
    message = request.args.get('message', '')
    error = ''
    predict_proba = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
          predict_proba = Classifier.predict_proba(vectorize_message).tolist()
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    if( predict == 'ham'): predict = 'Not Spam'
    
    return jsonify(message=message, predict_proba=predict_proba,
              predict=predict, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run()


 # score
#vectorize_text = Vectorizer.transform(test_feature)
#score = Classifier.score(vectorize_text, test_class)
#val = '. Has score: ' + str(score)
#print(val)
#        
#csv_arr = []
#for i in range(0,len(test_feature)):
#    answer = test_class[i]
#    text = test_feature[i]
#    vectorize_text = Vectorizer.transform([text])
#    predict = Classifier.predict(vectorize_text)[0]
#    if predict == answer:
#        result = 'right'
#    else:
#        result = 'wrong'
#    csv_arr.append([len(csv_arr), text, answer, predict, result])


