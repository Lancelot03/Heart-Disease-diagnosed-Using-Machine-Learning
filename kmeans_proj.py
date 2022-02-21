#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[37]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(heart.drop(columns=['HeartDisease'], axis=1))


# In[38]:


trans_heart = scaler.transform(heart.drop(columns=['HeartDisease'], axis=1))
trans_heart


# In[39]:


trans_heart = pd.DataFrame(trans_heart, columns=heart.columns[:-1])
trans_heart.head()


# In[40]:


X = trans_heart.copy()
y = heart[['HeartDisease']]
y.head()


# In[41]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=101)


# In[42]:


from sklearn.cluster import KMeans
model = KMeans(n_clusters=2, random_state=101)
model.fit(X_train)


# In[43]:


predictions = model.predict(X_test)
lables = model.labels_


# In[44]:


print('Lables : ', lables)


# In[45]:


print('y_test : ', y_test)
print('Predictions : ', predictions)


# In[46]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(classification_report(y_test, predictions))
print('Accuracy : ', accuracy_score(y_test, predictions))


# In[47]:


print(confusion_matrix(y_test, predictions))


# In[ ]:




