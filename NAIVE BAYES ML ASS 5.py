#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder


# In[3]:


dataset = [
    ['sunny', 'hot', 'high', 'weak', 'no'],
    ['sunny', 'hot', 'high', 'strong', 'no'],
    ['overcast', 'hot', 'high', 'weak', 'yes'],
    ['rainy', 'mild', 'high', 'weak', 'yes'],
    ['rainy', 'cool', 'normal', 'weak', 'yes'],
    ['rainy', 'cool', 'normal', 'strong', 'no'],
    ['overcast', 'cool', 'normal', 'strong', 'yes'],
    ['sunny', 'mild', 'high', 'weak', 'no'],
    ['sunny', 'cool', 'normal', 'weak', 'yes'],
    ['rainy', 'mild', 'normal', 'weak', 'yes'],
    ['sunny', 'mild', 'normal', 'strong', 'yes'],
    ['overcast', 'mild', 'high', 'strong', 'yes'],
    ['overcast', 'hot', 'normal', 'weak', 'yes'],
    ['rainy', 'mild', 'high', 'strong', 'no']]


# In[4]:


columns = ['Outlook', 'Temperature', 'Humidity', 'Wind', 'Play']
df = pd.DataFrame(dataset, columns=columns)


# In[5]:


encoder = OrdinalEncoder()
encoded_features = encoder.fit_transform(df.iloc[:, :-1])


# In[6]:


classifier = CategoricalNB()
classifier.fit(encoded_features, df['Play'])


# In[7]:


new_instances = [['rainy', 'cool', 'normal', 'strong'],
                 ['sunny', 'hot', 'high', 'weak']]
encoded_new_instances = encoder.transform(new_instances)
predictions = classifier.predict(encoded_new_instances)


# In[8]:


for instance, prediction in zip(new_instances, predictions):
    print("Weather conditions:", instance)
    print("Prediction:", prediction)
    print()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




