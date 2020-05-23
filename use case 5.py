
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
import lifetimes

########### Data Collection #############################
os.chdir('F:\\LocalDriveD\\Analytics\\Freelancing\\Scaleup\\Project 2 Sales and Marketing Analytics\\Use case 5')
maindata = pd.read_csv('Casino_Club_Cabana.csv', delimiter=',', decimal=',')

################## Data Understanding ########################
maindata.head()

maindata.isnull().sum()

maindata['SalesAmount'].hist()

maindata.info()
maindata['Game_Date'] = pd.to_datetime(maindata['Game_Date'])

maindata.describe(include='all')

############### Data Preparation ########################
data = maindata[['Game_Date', 'CustomerID','SalesAmount']]
data.columns = ['date','id','sale']

# First we need to derive frequency , Age and Recency to train Beta-Geo model  
# pakage lifetimes provide inbuilt function to prepare data
from lifetimes.datasets import load_transaction_data
from lifetimes.utils import summary_data_from_transaction_data
summary = summary_data_from_transaction_data(data, 'id', 'date','sale',
                                             observation_period_end=data.date.max())
print(summary.head())

'''
- recency denotes the age/period which is equal to the duration between a 
customer's first purchase and their 
 latest purchase. (so if customer has made 1 purchase that means recency is 0)'
- frequency represents the number of *repeat* purchases the customer has made. 
    This means that it's one less than the total number of purchases.
- T represents the age/period of the customer in whatever time units chosen (days above).
    This is equal to the duration between a customer's first purchase and the end of 
    the period under study.
'''

############################## Transaction Data Prepation ##############
from lifetimes.datasets import load_cdnow_summary_data_with_monetary_value
summary_with_money_value = summary.copy()
summary_with_money_value.head()
## Filtering out customers who have only 1 purchase
returning_customers_summary = summary_with_money_value[summary_with_money_value['frequency']>0]


############################### Average Profit Calulation ##########
#At this point we can train our Gamma-Gamma submodel and predict the conditional, expected average lifetime value of our customers.
from lifetimes import GammaGammaFitter
ggf = GammaGammaFitter(penalizer_coef = 0)
ggf.fit(returning_customers_summary['frequency'],returning_customers_summary['monetary_value'])
print(ggf)

#We can now estimate the average transaction value:
AVG_Profit = ggf.conditional_expected_average_profit(
returning_customers_summary['frequency'],
returning_customers_summary['monetary_value']
)
AVG_Profit = pd.Series(AVG_Profit)

############################### Customer Life Time Value Calculationn ##########
# refit the BG model to the summary_with_money_value dataset, #the model to use to predict the number of future transactions
from lifetimes import BetaGeoFitter
bgf = BetaGeoFitter(penalizer_coef=0.0)
bgf.fit(returning_customers_summary['frequency'], 
        returning_customers_summary['recency'], returning_customers_summary['T'])

CLV_1Year = ggf.customer_lifetime_value(
bgf, 
returning_customers_summary['frequency'],
returning_customers_summary['recency'],
returning_customers_summary['T'],
returning_customers_summary['monetary_value'],
time=12,freq = 'D')
CLV_1Year = pd.Series(CLV_1Year)

################# Churn Probability ###############################
# probability of being churn: model is going to predict customer churn, i.e probability of customer being dead or probability that a customer will leave
alive = bgf.conditional_probability_alive( returning_customers_summary['frequency'], 
                                  returning_customers_summary['recency'],
                                  returning_customers_summary['T'])




################ Final Output ###############################
returning_customers_summary2 = returning_customers_summary.copy()
returning_customers_summary2['Churn_Probability'] = 1- alive 
returning_customers_summary2['AVG_SALE']= AVG_Profit
returning_customers_summary2['CLV_1Year']= CLV_1Year
returning_customers_summary2.to_csv('output.csv')
