#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score



# In[30]:


# load the data from csv file to pandas Dataframe
titanic_data = pd.read_csv("train.csv")


# In[31]:


#printing the first 5 row of the dataFrame
titanic_data.head()


# In[32]:


# number of rows and columns
titanic_data.shape


# In[33]:


# getting information about the data
titanic_data.info()


# In[41]:


# check the number of missing values in each column
titanic_data.isnull().sum()


# In[154]:





# In[151]:


# replacing the missing age value by the mean value
titanic_data["Age"].fillna(titanic_data["Age"].mean(),inplace=True)


# In[123]:


# finding the mode value of "Embarked" column
print(titanic_data["Embarked"].mode())


# In[44]:


print(titanic_data['Embarked'].mode()[0])


# In[46]:


# replacing the missing values in "Embarked" column with mode values
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0],inplace=True)


# In[47]:


titanic_data.isnull().sum()


# In[48]:


#getting some statistical measures about the data
titanic_data.describe()


# In[49]:


# finding the people survived and not survived
titanic_data['Survived'].value_counts()


# In[54]:


#Data visualization
sns.set()


# In[60]:


# making a count plot for "Survived" column
ax = sns.countplot(x = 'Survived', data = titanic_data)


# In[61]:


# making a count plot for "Sex" column
ax = sns.countplot(x = 'Sex', data = titanic_data)


# In[62]:


titanic_data['Sex'].value_counts()


# In[64]:


# number of survivors gender wise
ax=sns.countplot(x = 'Sex', hue='Survived', data=titanic_data)


# In[65]:


# making a count plot for "Pclass" column
ax = sns.countplot(x = 'Pclass', data = titanic_data)


# In[66]:


ax = sns.countplot(x ='Pclass', hue='Survived', data=titanic_data)


# In[67]:


# Encoding the categorial columns
titanic_data['Sex'].value_counts()


# In[68]:


titanic_data['Embarked'].value_counts()


# In[98]:


#coverting categorial columns
titanic_data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1 ,'Q':2 }}, inplace=True)


# In[99]:


titanic_data.replace()


# In[101]:


# separating features & target
#X = titanic_data.drop(columns = ['PassengerId','Name','Ticket','Survived'],axis=1)
Y= titanic_data['Survived']


# In[87]:


print(X)


# In[102]:


print(Y)


# In[103]:


#splitting the data into training data and test data
X_train, X_test,Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=2)


# In[104]:


print(X.shape, X_train.shape, X_test.shape)


# In[105]:


#model training data
#logistic Regression
model = LogisticRegression()


# In[106]:


model.fit(X_train, Y_train)


# In[107]:


# accuracy on training date
X_train_prediction = model.predict(X_train)


# In[108]:


print(X_train_prediction)


# In[111]:


training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
print('Accuracy score of training data : ', training_data_accuracy)


# In[117]:


X_test_prediction = model.predict(X_test)


# In[118]:


print(X_test_prediction)


# In[120]:


test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print('Accuracy score of test data : ', test_data_accuracy)


# In[ ]:




