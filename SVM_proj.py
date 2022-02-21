#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np


# In[33]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[34]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(heart.drop(columns=['HeartDisease'], axis=1))


# In[35]:


trans_heart = scaler.transform(heart.drop(columns=['HeartDisease'], axis=1))
trans_heart = pd.DataFrame(trans_heart, columns=heart.columns[:-1])
trans_heart.head()


# In[36]:


X = trans_heart.copy()
y = heart[['HeartDisease']]
y.head()


# In[37]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)


# In[38]:


from sklearn import svm
model = svm.SVC()
model.fit(X_train, y_train.values.ravel())


# In[39]:


predictions = model.predict(X_test)


# In[40]:


print('Predctions : ', predictions)
print('y_test : ', y_test)


# In[41]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print(classification_report(y_test, predictions))
print("Accuracy : ", accuracy_score(y_test, predictions))


# In[42]:


print(confusion_matrix(y_test, predictions))


# In[ ]:




