#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import math
data= {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
        }
dataread=pd.DataFrame(data)
def entropy(attribute):
    values = dataread[attribute].unique()
    entropy = 0
    for value in values:
        p = len(dataread[dataread[attribute] == value]) / len(dataread)
        entropy += -p * math.log2(p)
    return entropy

age_entropy = entropy('age')
income_entropy = entropy('income')
student_entropy = entropy('student')
credit_rating_entropy = entropy('credit_rating')
buys_computer_entropy = entropy('buys_computer')
print('Entropy  Age:', age_entropy)
print('Entropy  Income:', income_entropy)
print('Entropy  Student:', student_entropy)
print('Entropy Credit Rating:', credit_rating_entropy)
print('Entropy  Buys Computer :', buys_computer_entropy)

def info_gn(attribute):
    values = dataread[attribute].unique()
    info_gn = buys_computer_entropy
    for value in values:
        subset = dataread[dataread[attribute] == value]
        p = len(subset) / len(dataread)
        info_gn -= p * entropy('buys_computer')
    return info_gn


age_info_gn = info_gn('age')
income_info_gn = info_gn('income')
student_info_gn = info_gn('student')
credit_rating_info_gn = info_gn('credit_rating')


print('Info gain for Age:', age_info_gn)
print('Info Gain for Income:', income_info_gn)
print('Info Gain for Student:', student_info_gn)
print('Info Gain for Credit Rating:', credit_rating_info_gn)

 

 

root_node = max(age_info_gn, income_info_gn, student_info_gn, credit_rating_info_gn)
if root_node == age_info_gn:
    print('The first feature for constructing the decision tree is Age.')
elif root_node == income_info_gn:
    print('The first feature for constructing the decision tree is Income.')
elif root_node == student_info_gn:
    print('The first feature for constructing the decision tree is Student.')
else:
    print('The first feature for constructing the decision tree is Credit Rating.')


# In[7]:


import pandas as pd
from sklearn.tree import DecisionTreeClassifier
data = {
    'age': ['<=30', '<=30', '31…40', '>40', '>40', '>40', '31…40', '<=30', '<=30', '>40', '<=30', '31…40', '31…40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}
df = pd.DataFrame(data)
df_encoded = df.apply(lambda col: pd.factorize(col)[0])
Tr_X = df_encoded.drop(columns=['buys_computer'])
Tr_y = df_encoded['buys_computer']
model = DecisionTreeClassifier()
model.fit(Tr_X, Tr_y)
training_accuracy = model.score(Tr_X, Tr_y)
print(f"Training Set Accuracy: {training_accuracy}")
tree_depth = model.get_depth()
print(f"Tree Depth: {tree_depth}")

