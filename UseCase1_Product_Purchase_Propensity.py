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
os.chdir('F:\\LocalDriveD\\Use case 1')

data = pd.read_csv('bank_data.csv', delimiter=',', decimal=',')
data.head()
data.info()
data.describe(include='all')

# Variable Understanding

data.Sent_Date.describe()
del data['Sent_Date']

data.Qualification.value_counts()
data.Qualification = data.Qualification.str.lower()      

data.Gender.value_counts()
data.Gender = data.Gender.str.lower()      
data['Gender'] = np.where(data['Gender']=='m','male',data['Gender'])
data['Gender'] = np.where(data['Gender']=='f','female',data['Gender'])

data.Marital_Status.value_counts()
data.Marital_Status = data.Marital_Status.str.lower()      
data['Marital_Status'] = np.where(data['Marital_Status']=='u','unmarried',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='m','married',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='w','widow',data['Marital_Status'])
data['Marital_Status'] = np.where(data['Marital_Status']=='d','divorcee',data['Marital_Status'])

data.Occupation_Code.value_counts()
data.Occupation_Code = data.Occupation_Code.str.lower()      

data['DateOfBirth'].head()
data['DateOfBirth'] = pd.to_datetime(data['DateOfBirth'])
data['Today']  = pd.to_datetime(date.today())
data['age'] = data['Today']-data['DateOfBirth']
data['age'] = np.round(data['age'].dt.days/365)
data['age'] = data['age'].astype(int)
data['age'].head()
del data['DateOfBirth']
del data['Today']

# Data Cleaning
data.isnull().sum()/data.shape[0]
limitper = len(data)*0.9
data = data.dropna(thresh = limitper,axis=1)

data_copy = data.copy()
data_cat = data[[ 'City',  'Qualification',
       'Gender', 'Marital_Status', 'Occupation_Code']]
data_num = data[['Customer_ID', 'age','Total_Credit_Cards',
       'Total_CC_monthly_limit', 'Total_Personal_Loans_till_time',
       'Total_HL_Loans', 'Total_Business_Loans', 'Total_Agriculture_Loans',
       'Total_Education_Loan', 'Total_Auto_Loan',
       'Total_balance_in_fixed_deposits', 'Total_Monthly_EMI',
       'Monthly_Mutual_fund_SIPs', 'Cibil_Score', 'Unsercued_loan_last6M',
       'Sercued_loan_last6M', 'Purchased']]
data_cat= data_cat.fillna(data_cat.mode().iloc[0])
data_num= data_num.fillna(data_num.mean().iloc[0])
data_Cat_dummy = pd.get_dummies(data_cat)
data = pd.concat([data_num,data_Cat_dummy],axis=1)
data.isnull().sum()/data.shape[0]
data.mean()
#data.to_csv('Bank_Data.csv')

## As per variable understanding we will check outlier points
sns.boxplot(x=data_num['Total_CC_monthly_limit'])
data.hist('Total_CC_monthly_limit')
median = float(data['Total_CC_monthly_limit'].median())
data["Total_CC_monthly_limit"] = np.where(data["Total_CC_monthly_limit"] > median, median, data['Total_CC_monthly_limit'])

sns.boxplot(x=data_num['age'])
data.hist('age')
median = float(data['age'].median())
data["age"] = np.where(data["age"] > median, median, data['age'])

sns.boxplot(x=data_num['Total_balance_in_fixed_deposits'])
data.hist('Total_CC_monthly_limit')
median = float(data['Total_balance_in_fixed_deposits'].median())
data["Total_balance_in_fixed_deposits"] = np.where(data["Total_balance_in_fixed_deposits"] > median, median, data['Total_balance_in_fixed_deposits'])

sns.boxplot(x=data_num['Total_Monthly_EMI'])
data.hist('Total_CC_monthly_limit')
median = float(data['Total_Monthly_EMI'].median())
data["Total_Monthly_EMI"] = np.where(data["Total_Monthly_EMI"] > median, median, data['Total_Monthly_EMI'])

sns.boxplot(x=data_num['Monthly_Mutual_fund_SIPs'])
data.hist('Total_CC_monthly_limit')
median = float(data['Monthly_Mutual_fund_SIPs'].median())
data["Monthly_Mutual_fund_SIPs"] = np.where(data["Monthly_Mutual_fund_SIPs"] > median, median, data['Monthly_Mutual_fund_SIPs'])

data['Purchased'].value_counts()
X = data[[ 'age', 'Total_Credit_Cards', 'Total_CC_monthly_limit',
       'Total_Personal_Loans_till_time', 'Total_HL_Loans',
       'Total_Business_Loans', 'Total_Agriculture_Loans',
       'Total_Education_Loan', 'Total_Auto_Loan',
       'Total_balance_in_fixed_deposits', 'Total_Monthly_EMI',
       'Monthly_Mutual_fund_SIPs', 'Cibil_Score', 'Unsercued_loan_last6M',
       'Sercued_loan_last6M',  'City_Tier1', 'City_Tier2',
       'City_Tier3', 'Qualification_diploma', 'Qualification_graduate',
       'Qualification_others', 'Qualification_phd',
       'Qualification_post graduate', 'Gender_female', 'Gender_male',
       'Gender_other', 'Marital_Status_divorcee', 'Marital_Status_married',
       'Marital_Status_other', 'Marital_Status_unmarried',
       'Marital_Status_widow', 'Occupation_Code_farmer',
       'Occupation_Code_house wife', 'Occupation_Code_other',
       'Occupation_Code_pensioner', 'Occupation_Code_salaried',
       'Occupation_Code_self employeed -nonprofessional',
       'Occupation_Code_self employeed -professional',
       'Occupation_Code_student', 'Occupation_Code_trader']]
y = data['Purchased']

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

########## ROC Curve on best model
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
n = [0 for _ in range(len(y_test))]
prob = xgb.predict_proba(X_test)
# Positive Outcome Probability
pred = prob[:, 1]
# plot the roc curve for the model
ns_fpr, ns_tpr, _ = roc_curve(y_test, n)
lr_fpr, lr_tpr, _ = roc_curve(y_test, pred)
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()

