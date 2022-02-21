#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


heart = pd.read_csv('heart_modified.csv')
heart = heart.drop(columns=['Unnamed: 0'], axis=1)    #because during making the heart_modified csv the index also was copied
heart.head()


# In[45]:


from sklearn.model_selection import train_test_split
X = heart.drop(columns=['HeartDisease'], axis=1)
y = heart[['HeartDisease']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[46]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
rfor = RandomForestClassifier(max_depth=2, random_state=101)
rfor.fit(X_train, y_train.values.ravel())


# In[47]:


predictions = rfor.predict(X_test)


# In[48]:


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
print(classification_report(y_test, predictions))
print('Accuracy : ', accuracy_score(y_test, predictions))


# In[49]:


print(confusion_matrix(y_test, predictions))


# #### Finding the most effective max_depth 

# In[50]:


from sklearn.metrics import f1_score


# In[51]:


f1_scores = []
max_depths = []
for i in range(1,41):
    rfor = RandomForestClassifier(max_depth=i, random_state=101)
    rfor.fit(X_train, y_train.values.ravel())
    predictions_1 = rfor.predict(X_test)
    f1_scores.append(f1_score(y_test, predictions_1))
    max_depths.append(i)


# In[52]:


sns.lineplot(x=max_depths, y=f1_scores)
plt.title('max_depth vs f1_scores')
plt.xlabel('max_depth')
plt.ylabel('f1_scores')


# #### Here you can see after max_depth = 7 we get no significant change in the value of f1_score
# #### so we can choose max_depth = 7

# In[53]:


rfor = RandomForestClassifier(max_depth=7, random_state=101)
rfor.fit(X_train, y_train.values.ravel())


# In[54]:


predictions = rfor.predict(X_test)


# In[55]:


print(classification_report(y_test, predictions))
print('Accuracy : ', accuracy_score(y_test, predictions))


# In[56]:


print(confusion_matrix(y_test, predictions))


# In[ ]:




