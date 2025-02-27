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
os.chdir('F:\\LocalDriveD\\Use case 4')

data = pd.read_csv('DataCommunicationTime.csv', delimiter=',', decimal=',')
data.head()
data.info()
data.describe(include='all')
data.columns
# Variable Understanding

data.Communication_Type.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Communication_Type", data=data)

data.Communication_Day.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Communication_Day", data=data)

data.City.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Communication_Type", data=data)

data.Qualification.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Qualification", data=data)

data.Gender.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Gender", data=data)

data.Marital_Status.value_counts()
data.Marital_Status = data.Marital_Status.str.lower()      
data['Marital_Status'] = np.where(data['Marital_Status']=='u','unmarried',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='m','married',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='w','widow',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='d','divorcee',data['Marital_Status'])
sns.set(style="darkgrid")
ax = sns.countplot(x="Marital_Status", data=data)

data.Occupation_Code.value_counts()
sns.set(style="darkgrid")
ax = sns.countplot(x="Occupation_Code", data=data)

data.TwitterPresence.value_counts()

data.LinkedinPresence.value_counts()

data.PreviousAttemptOfCommunication.describe()

data.SuccessfullCommunication.describe()

data.Communication_Success.value_counts()

data.MonthPeriod.value_counts()

data.Industry.value_counts()

data.Type_Accomodation.value_counts()

data['DateOfBirth'].head()
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'])
data['Today']  = pd.to_datetime(date.today())
data['age'] = data['Today']-data['DateOfBirth']
data['age'] = np.round(data['age'].dt.days/365)
data['age'] = data['age'].astype(int)
data['age'].head()
del data['DateOfBirth']
del data['Today']
data['age'].hist()

# Data Cleaning
data.isnull().sum()/data.shape[0]
limitper = len(data)*0.9
data = data.dropna(thresh = limitper,axis=1)

data_copy = data.copy()
data_cat = data[['Communication_Day', 'MonthPeriod', 'Communication_Type',
       'City', 'Industry', 'Qualification', 'Gender', 'Type_Accomodation',
       'Marital_Status', 'Occupation_Code', 'FacebookPresence',
       'TwitterPresence', 'LinkedinPresence', 'PreviousAttemptOfCommunication']]
data_num = data[['Customer_ID','SuccessfullCommunication' , 'age']]


data_cat= data_cat.fillna(data_cat.mode().iloc[0])
data_num= data_num.fillna(data_num.mean().iloc[0])
data_Cat_dummy = pd.get_dummies(data_cat)
data = pd.concat([data_num,data_Cat_dummy],axis=1)
data['Communication_Success'] = data_copy['Communication_Success']
data.isnull().sum()/data.shape[0]

## Since no variable has a significant large values we can omit outlier test

data['Communication_Success'].value_counts()
X = data[['SuccessfullCommunication', 'age', 'FacebookPresence',
       'TwitterPresence', 'LinkedinPresence', 'PreviousAttemptOfCommunication',
       'Communication_Day_weekday', 'Communication_Day_weekend',
       'MonthPeriod_10To20Days', 'MonthPeriod_First10Days',
       'MonthPeriod_Last10Days', 'Communication_Type_promotional',
       'Communication_Type_service', 'City_Tier1', 'City_Tier2', 'City_Tier3',
       'Industry_Banking/Finance', 'Industry_IT', 'Industry_Others',
       'Industry_Retail', 'Industry_Telecom', 'Industry_manufacturing',
       'Qualification_Diploma', 'Qualification_Graduate',
       'Qualification_Others', 'Qualification_PHD',
       'Qualification_Post Graduate', 'Gender_female', 'Gender_male',
       'Gender_other', 'Type_Accomodation_rented',
       'Type_Accomodation_self_home', 'Marital_Status_divorcee',
       'Marital_Status_married', 'Marital_Status_other',
       'Marital_Status_unmarried', 'Marital_Status_widow',
       'Occupation_Code_farmer', 'Occupation_Code_house wife',
       'Occupation_Code_other', 'Occupation_Code_pensioner',
       'Occupation_Code_salaried',
       'Occupation_Code_self employeed -nonprofessional',
       'Occupation_Code_self employeed -professional',
       'Occupation_Code_student', 'Occupation_Code_trader']]
y = data[ 'Communication_Success']

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
max_depth = [10,50]
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
prediction_prob = pd.DataFrame(xgb.predict_proba(X_test))
prediction_prob.columns = ['12PM_6PM','6AM_12PM','6PM_12PM']
prediction = pd.Series(xgb.predict(X_test))
Output = pd.concat([Output,prediction_prob],axis=1)
Output['Prediction'] = prediction
Output.to_csv('final_output.csv')


########## ROC Curve on best model
# Compute ROC curve and ROC area for each class
y_score = xgb.predict_proba(X_test)
from sklearn.preprocessing import LabelBinarizer
y_test_binary = label_binarize(y_test, classes=[0, 1, 2])
n_classes = 3
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_binary[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_test_binary.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Compute macro-average ROC curve and ROC area
# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
lw=2
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()
