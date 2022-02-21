#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
import numpy as np


# In[23]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(heart.drop(columns=['HeartDisease'], axis=1))


# In[25]:


trans_heart = scaler.transform(heart.drop(columns=['HeartDisease'], axis=1))
trans_heart = pd.DataFrame(trans_heart, columns=heart.columns[:-1])
trans_heart.head()


# In[26]:


X = trans_heart.copy()
y = heart[['HeartDisease']]
y.head()


# In[27]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)


# In[28]:


from sklearn.linear_model import LogisticRegression
lrig = LogisticRegression()
lrig.fit(X_train, y_train.values.ravel())


# In[29]:


predictions = lrig.predict(X_test)


# In[30]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Y_test : ",y_test)
print("Predictions : ", predictions)


# In[31]:


print(classification_report(y_test, predictions))
print('Accuracy : ', accuracy_score(y_test, predictions))


# In[32]:


print(confusion_matrix(y_test, predictions))

