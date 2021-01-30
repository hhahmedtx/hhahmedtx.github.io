#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# In[2]:


import os


# In[3]:


os.getcwd()


# In[4]:


os.listdir()


# In[5]:


os.chdir('C:\\Users\\Hasan\\Downloads\\bank')


# In[6]:


os.getcwd()


# In[7]:


os.listdir()


# Section I

# PART I Loading the dataset into the notebook

# In[8]:


bank = pd.read_csv('bank-full.csv',delimiter=';')


# In[9]:


bank


# Part II Exploring the dataset and making note of the attribute

# In[10]:


bank.describe()


# In[11]:


bank.info


# Part III Significance of the y column and the value counts of the y column

# In[12]:


bank['y'].value_counts()


# PART IV: The y column as above indicates the yes and no regarding the predictions made previously regarding of whether the people have subscribed to a term deposit or not. 

# In[13]:


5289/39922 


# The ratio of two classes is imbalanced as only 13% of the people have subscribed to a term deposit as in comparison to 87% of what haven't subscribed to a term deposit. 

# Section II

# PART I getting all the datatypes of all the columns of our dataset

# In[14]:


bank.dtypes 


# Part II : checking for errors in the dataset

# In[15]:


bank.isnull().sum() # no missing value


# Part III: there have not been any deviation in the dataset compared to the description provided by UCI

# Section III Exploring data with group by

# In[16]:


bank.groupby('y').mean() # group by mean of y


# In[17]:


bank.groupby('job').mean() # group by mean of job


# In[18]:


bank.groupby('marital').mean()


# In[19]:


bank.groupby('education').mean()


# Exploratory data analysis

# In[20]:


bank.groupby(['job','y'])['y'].count()
y = bank.groupby(['job','y'])['y'].count()
x = y.index
x = [' '.join([a,b]) for a,b in x] # do this same code for marital and education.
plt.bar(x,y)


# In[21]:


bank.groupby(['marital','y'])['y'].count()
y = bank.groupby(['marital','y'])['y'].count()
x = y.index
x = [' '.join([a,b]) for a,b in x]
plt.bar(x,y)


# In[22]:


bank.groupby(['education','y'])['y'].count()
y = bank.groupby(['education','y'])['y'].count()
x = y.index
x = [' '.join([a,b]) for a,b in x]
plt.bar(x,y)


# In[23]:


plt.hist(bank['age'])


# Section IV : creating dummy variables along with label encoders

# In[24]:


dtypes_categorical = bank.dtypes


# In[25]:


dtypes_categorical


# In[26]:


obj_col = dtypes_categorical.loc[dtypes_categorical == object].index
obj_col


# In[27]:


bank.head()


# 
#    2 - job : type of job (categorical: "admin.","unknown","unemployed","management","housemaid","entrepreneur","student",
#                                        "blue-collar","self-employed","retired","technician","services") - dummy variables
#    3 - marital : marital status (categorical: "married","divorced","single"; note: "divorced" means divorced or widowed)-dummy variables
#    4 - education (categorical: "unknown","secondary","primary","tertiary") - label encoder
#    5 - default: has credit in default? (binary: "yes","no")- dummy variables
# 
#    7 - housing: has housing loan? (binary: "yes","no") - dummy variables
#    8 - loan: has personal loan? (binary: "yes","no") - dummy variables
#    
#    9 - contact: contact communication type (categorical: "unknown","telephone","cellular") - dummy variables 
#   11 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec") - label encoder
#    
#   16 - poutcome: outcome of the previous marketing campaign (categorical: "unknown","other","failure","success") - dummy variables
# 
#   Output variable (desired target):
#   17 - y - has the client subscribed a term deposit? (binary: "yes","no") - labelencoder.
# 

# In[28]:


# label encoder can be done manually. 


# y, education and month are not dummies

# In[29]:


job_dummies = pd.get_dummies('job')


# In[30]:


dummy_col = [x for x in obj_col if x not in ('y','education','month')]
dummy_col


# In[31]:


for col in dummy_col:
    dummies = pd.get_dummies(bank[col]) # for every column we are generating a dummy
    dummies.columns = ['_'.join((col,x)) for x in dummies.columns]
    bank = bank.drop(col, axis = 1) # we are dropping the original column related to the string
    bank = pd.concat([bank,dummies], axis =1) # add new columns or dummy columns which are numeric representations. 

bank.info()


# In[32]:


le_col = ['y','education','month']


# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


le = {} # empty dictionary
for col in ('y','education'):
    outcome = LabelEncoder() # from the bank head we can see that for y the first three are no's, therefore from the array, we can see that no is equal to zeroes 
#out_one = outcome.fit_transform(bank['y'])
    out_one = outcome.fit_transform(bank[col])
    le[col]=outcome
    bank[col]=out_one


# In[35]:


#outcome.inverse_transform([0,1]) # 0 is no, 1 is yes. noise
#outcome.classes_


# In[36]:


#outcome.inverse_transform(range(12))# went according to the alphabetical order. so anything with a will be 0 and since sep is the last it is 11.


# In[37]:


month_dictionary = {'apr':4, 'aug':8, 'dec':12, 'feb':2,'jan':1, 'jul':7, 'jun':6, 'mar':3, 'may':5,
       'nov':11, 'oct':10,'sep':9}

for name,num in month_dictionary.items():
    bank.loc[bank['month'] == name,'month'] = num
bank['month'] = bank['month'].astype(int)


# In[38]:


bank['month'] # 5,5,5,5, shows the month of may, check bank.head()


# In[39]:


bank.head()


# In[40]:


bank.info()


# Section V : Preliminary training

# In[41]:


from sklearn.model_selection import train_test_split


# In[42]:


x = bank.drop('y', axis=1)
y = bank['y']
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25) # 75 -25 train- test, test size is 25% split


# In[43]:


from sklearn.linear_model import LogisticRegression


# In[44]:


model = LogisticRegression()
model = model.fit(x_train,y_train)


# In[45]:


model.score(x_train, y_train) # accuracy score. 


# In[46]:


model.score(x_test,y_test) # similar performance. 


# Section VI: improving the performance of the model

# Part I: when we trained the model in the previous section it was stated that the model score or the accuracy for both the test and the train were very close to one another, the reason for this is simple. There were already a lot of no's in the y variable to begin with thus it wasn't going to change much even if the model was trained with a different percentage of the dataset. 

# Implementing SMOTE

# In[47]:


pip install imbalanced-learn 


# In[48]:


from imblearn.over_sampling import SMOTE


# In[49]:


balance = SMOTE()
x_bal,y_bal = balance.fit_resample(x,y)


# In[50]:


x_bal.info()


# In[51]:


print("The value count of the classes")
y_bal.value_counts()


# after making note of the y label it is seen that the yes and no's are equal or balanced after SMOTE was applied.

# In[52]:


model2 = LogisticRegression()
model2 = model2.fit(x_bal,y_bal)


# In[53]:


model2.score(x_bal,y_bal)


# In[54]:


model2.score(x_test,y_test)


# In[55]:


from sklearn.feature_selection import RFE
estimator = LogisticRegression()
selector = RFE(estimator, n_features_to_select=10) # selects everything but the y column
selector_one = selector.fit_transform(x,y)


# In[56]:


selector.support_ # false are the features that are not selected. vice versa


# In[57]:


selected_x_train = selector.transform(x_train) # selected x_train


# In[58]:


model3 = LogisticRegression()
model3 = model3.fit(selected_x_train,y_train)


# In[59]:


model3.score(selected_x_train, y_train)


# Section VIII training the model with a new set of data.

# In[60]:


x = bank.drop('y', axis=1)
y = bank['y']
Selected_x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25) # 75 -25 train- test, test size is 25% split

balance = SMOTE()
selected_x_train, y_train = balance.fit_resample(x,y)


# In[61]:


model4 = LogisticRegression()
model4 = model4.fit(selected_x_train, y_train)


# In[62]:


model4.score(selected_x_train, y_train)


# In[63]:


model4.score(x_test, y_test)


# In[64]:


y_pred_val = model4.predict(x_test) # over here what are doing is using the x test to predict a dependent variable named as the y_pred_val. 


# In[65]:


display(x_test)
display(y_pred_val)


# Section IX

# Here after retraining the model it is seen that the accuracies are very similar to the train and test models even after the yes and no's have been balanced

# In[69]:


from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
 
confusion_matrix(y_test, y_pred_val) 
# this gives us our true positive, false positive, true negative, false negative. 


# TP = 8000<, FP = 1600<, FN = 370<, TN = 900<. After using the confusion matrix it can be stated that more than 8000 of the values are TP, 900 are TN. these are the true values and a combination of both these are more than the false positives and false negatives combined. 

# In[71]:


y_test


# In[72]:


y_pred_val


# In[73]:


print(classification_report(y_test, y_pred_val))

Section IX
# from the above classification report, a number of aspects come into sight. 
# 
# First the precision for a termed deposit to be negative or no or(0), the ratio of tp/tp + fp = 0.96. This is a very high number meaning there aren't many false positives here and most of them are true positives. In terms of a termed deposit to be predicted or classified positive or 1, the precision from the classification report turns out to be 0.36. This means there can be a lot of false positives from this model. Thus it is not often correct even if the model classified the termed deposit to be a yes or 1. 
# 
# Secondly the recall for 0 and 1 are respectively 0.83 and 0.71. This means that for 0 or a termed deposit to be classified as no, 83% of the time model predicts it correctly. In terms of a yes, the percentage or ratio is lower, thus for a termed deposit to be classified as yes or 1 71% of the time the model has correctly predicted it.
# 
# f1 score is weighted average between precision and recall. 
# 
# Accuracy from the classification report is 0.82 or 82%. This shows that considering our dataset and after balancing it, we can get an accuracy rate of 82%. 

# Section X
# 
# After balancing the models, it can be seen that according to our dataset, the majority of the classes predicted for a term deposit was no. Even when the model was imbalanced, it showed that majority of the term deposits can be classified as no. However, that model was imbalanced and it was easily skewed to the 0 or no. But once the model was trained, balanced and retrained, even though the answers didn't change, the model became more reliable. Thus when a different dataset with different values and different attributes is tested with the help of this model, it will show more accurate results in contrast if the model wasn't balanced out and retrained. 
# 
# The only recommendation I would be making is test out different datasets with different attributes. In that way the model can be improved. Working, training and balancing out only dataset is not enough to get the best results out of the model. It wouldn't give out the best predictions 
