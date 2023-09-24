#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


titanic_df=pd.read_csv('tested.csv')


# In[4]:


titanic_df


# In[6]:


titanic_df.head()


# In[7]:


titanic_df.shape


# In[8]:


titanic_df.describe()


# In[9]:


titanic_df.isnull().sum()


# In[10]:


titanic_df['Age'].fillna(titanic_df['Age'].mean(),inplace=True)


# In[11]:


titanic_df.isnull().sum()


# In[12]:


titanic_df=titanic_df.drop(columns='Cabin',axis=1)


# In[13]:


titanic_df.isnull().sum()


# In[15]:


titanic_df['Fare'].fillna(titanic_df['Fare'].mean(),inplace=True)


# In[16]:


titanic_df.isnull().sum()


# In[17]:


titanic_df['Survived'].value_counts()


# In[18]:


sns.countplot(x='Survived',data=titanic_df)


# In[20]:


sns.countplot(x='Sex',data=titanic_df)


# In[21]:


sns.countplot(x='Pclass',hue='Survived',data=titanic_df)


# In[23]:


titanic_df.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':3}},inplace=True)


# In[24]:


titanic_df.head()


# In[25]:


X=titanic_df.drop(columns=['PassengerId','Survived','Ticket','Name'],axis=1)


# In[26]:


Y=titanic_df['Survived']


# In[27]:


print(X)


# In[28]:


print(Y)


# In[29]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=2)


# In[30]:


print(X.shape,X_train.shape,X_test.shape)


# In[33]:


model = LogisticRegression(max_iter=1000)


# In[34]:


model.fit(X_train,Y_train)


# In[36]:


X_train_prediction=model.predict(X_train)


# In[37]:


X_train_prediction


# In[38]:


training_data_accuracy=accuracy_score(Y_train,X_train_prediction)
print("Accuracy score of training data:",training_data_accuracy)


# In[39]:


X_test_prediction=model.predict(X_test)
X_test_prediction


# In[40]:


test_data_accuracy=accuracy_score(Y_test,X_test_prediction)
print("Accuracy score of test data:",test_data_accuracy)


# In[ ]:




