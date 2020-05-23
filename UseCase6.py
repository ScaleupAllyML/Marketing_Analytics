# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 17:54:30 2019

@author: ankit
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import scipy
import numpy as np
import re
import nltk
import datetime
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import random
import spacy
import gensim 
from datetime import datetime,date,time
from gensim.models import Word2Vec 
from nltk.tokenize import sent_tokenize, word_tokenize 
import matplotlib.pyplot as plt
pd.set_option("display.max_columns",50)
pd.set_option("display.max_rows",300)
from nltk import FreqDist
import seaborn as sns
import plotly.graph_objects as go #pip install plotly==4.1.0
import plotly
from plotly.offline import init_notebook_mode;import plotly.graph_objs as go;plotly.offline.init_notebook_mode(connected=True)# offline ploting of box plots
import pyod
from datetime import date, time
os.chdir('F:\\LocalDriveD\\Analytics\\Freelancing\\Scaleup\\Project 2 Sales and Marketing Analytics\\Use case 6')

data = pd.read_csv('Email_Data.csv', delimiter=',', decimal=',')
data.head()
data.info()
data.describe(include='all')

# Variable Understanding

data.Sent_Date.describe()
del data['Sent_Date']
      
data.Age_Group.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Age_Group", data=data)

data.CTA_position.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="CTA_position", data=data)

data.Size_of_best_image_in_mail.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Size_of_best_image_in_mail", data=data)

data.Total_Emoticon_in_subject_line.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Total_Emoticon_in_subject_line", data=data)

data.Type_of_Content.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Type_of_Content", data=data)

data['Click'] = np.where(data['Click']== 0,'No','Yes')
data.Click.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Click", data=data)

data['Total_Images'].hist()


# Data Cleaning
data.isnull().sum()/data.shape[0]

data_copy = data.copy()
data_cat = data[[ 'Age_Group',  'CTA_position',
       'Size_of_best_image_in_mail', 'Type_of_Content']]
data_num = data[['Email_ID','Total_Emoticon_in_subject_line',
                 'Total_Images','Click','Message']]

data_Cat_dummy = pd.get_dummies(data_cat)
data = pd.concat([data_num,data_Cat_dummy],axis=1)


## As per variable understanding we will check outlier points
sns.boxplot(x=data_num['Total_Images'])
data.hist('Total_Images')
median = float(data['Total_Images'].median())
data["Total_Images"] = np.where(data["Total_Images"] > median, median, data['Total_Images'])


####################################### Text Mining
# Strip white space
data['Message'] = [re.sub(r"s+"," ", w, flags = re.I) for w in data['Message']]
# Cleaning the texts
corpus = []
for i in range(0, data.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', data['Message'][i])
    review = review.lower()
    review = review.split()
    review = ' '.join(review)
    corpus.append(review)

# # function to remove stopwords
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
def remove_stopwords(rev):
    rev_new = " ".join([i for i in rev if i not in stop_words])
    return rev_new
data['Message'] = [remove_stopwords(r.split()) for r in data['Message']]
### total words/character in sentence
data['totalChar'] = data['Message'].str.len()
data['totalwords'] = data['Message'].str.split().str.len()

### Document-Term Matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(strip_accents='unicode',max_features = 1500, analyzer='word', ngram_range=(1,3), norm='l2')
X1 = vectorizer.fit_transform(corpus).toarray()
X1 = pd.DataFrame(X1)
X1.columns = vectorizer.get_feature_names()


X = data[['Total_Emoticon_in_subject_line', 'Total_Images',
       'Age_Group_25-30', 'Age_Group_30-35', 'Age_Group_35-40',
       'Age_Group_40-50', 'Age_Group_less than 25', 'Age_Group_more than 50',
       'CTA_position_Bottom on mail', 'CTA_position_Center',
       'CTA_position_Top on mail', 'Size_of_best_image_in_mail_0',
       'Size_of_best_image_in_mail_1080X600',
       'Size_of_best_image_in_mail_1080X760',
       'Size_of_best_image_in_mail_1220X600',
       'Size_of_best_image_in_mail_360X720',
       'Size_of_best_image_in_mail_480X480',
       'Size_of_best_image_in_mail_600X600',
       'Size_of_best_image_in_mail_700X600',
       'Size_of_best_image_in_mail_700X800',
       'Size_of_best_image_in_mail_720X720',
       'Size_of_best_image_in_mail_960X600', 'Type_of_Content_NonPromotion',
       'Type_of_Content_Promotional', 'totalChar', 'totalwords']]
X = pd.concat([X,X1],axis=1)
y = data['Click']

######## Model Development ##############
from sklearn.preprocessing import StandardScaler
Scale = StandardScaler()
ScaledObject = Scale.fit(X)
X_scaled = ScaledObject.transform(X)
X_scaled = pd.DataFrame(X_scaled,columns = X.columns)
######### Train/Test Split
a = random.sample(range(0, X.shape[0]), int(X.shape[0]*0.75))
X_train = X.iloc[a,] 
X_test = X.drop(X.index[a])
y_train = y.iloc[a,] 
y_test = y.drop(y.index[a])

from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
# Let's first try a simple model to make a benchmark for our model performance
from sklearn.tree import DecisionTreeClassifier
scoring = ['precision_macro','recall_macro','accuracy']
clf = DecisionTreeClassifier(max_depth = 50,random_state=0)
scores = cross_validate(estimator = clf,X=X_train, y=y_train,cv=3,scoring=scoring)
scores

################## Precision,Accuracy,Recall through cross validation###
from xgboost import XGBClassifier
Performance = pd.DataFrame(columns = ['test_accuracy','test_precision_macro','test_recall_macro',
                                 'Max_depth','n_estimators','learning_rate'])
max_depth = [10,30,50]
for k in range(1,2):
    for j in range(1,2):
        for i in max_depth:
            clf = XGBClassifier(learning_rate =k/10, max_depth = i,n_estimators = j*100,random_state=0)
            scores = cross_validate(estimator = clf,X=X_train, y=y_train,cv=2,scoring=scoring)
            a = pd.DataFrame(scores)
            a = a[['test_accuracy','test_precision_macro','test_recall_macro']]
            a['Max_depth']=i
            a['n_estimators']=j*100
            a['learning_rate']=k/10
            Performance = pd.concat([a,Performance],axis=0)
Performance[['Max_depth','n_estimators','learning_rate','test_accuracy']][Performance['test_accuracy'] ==Performance['test_accuracy'].max()] #50
print(Performance)

####### Validation of model results from best obtained model ( frm c.v.) on test data
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score,roc_curve,auc
from sklearn.preprocessing import label_binarize
from scipy import interp
from itertools import cycle
import matplotlib.pyplot as plt
xgb = XGBClassifier(max_depth = 10,n_estimators =100,learning_rate = 0.1,random_state=0)
xgb.fit(X_train,y_train)
y_pred = xgb.predict(X_test)
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))

### Prediction on new data
Output = X_test.copy()
Output = Output.reset_index() 
del Output['level_0']
prediction_prob = pd.DataFrame(xgb.predict_proba(X_test))
prediction_prob.columns = ['No','Yes']
prediction = pd.Series(xgb.predict(X_test))
Output = pd.concat([Output,prediction_prob],axis=1)
Output['Prediction'] = prediction
Output.to_csv('final_output.csv')


########## ROC Curve on best model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
n = [0 for _ in range(len(y_test))]
prob = xgb.predict_proba(X_test)
# Positive Outcome Probability
pred = prob[:, 1]
# plot the roc curve for the model
y_test = np.where(y_test== 'No',0,1)
ns_fpr, ns_tpr, _ = roc_curve(y_test, n)
lr_fpr, lr_tpr, _ = roc_curve(y_test, pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

