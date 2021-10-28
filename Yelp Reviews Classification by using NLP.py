#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# - In this project, Natural Language Processing (NLP) strategies will be used to analyze Yelp reviews data
# - Number of 'stars' indicate the business rating given by a customer, ranging from 1 to 5
# - 'Cool', 'Useful' and 'Funny' indicate the number of cool votes given by other Yelp Users. 
# 
# Photo Credit: https://commons.wikimedia.org/wiki/File:Yelp_Logo.svg
# 

# ![image.png](attachment:image.png)

# # STEP #0: LIBRARIES IMPORT
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP #1: IMPORT DATASET

# In[2]:


yelp_df = pd.read_csv("yelp.csv")


# In[3]:


yelp_df.head(10)


# In[4]:


yelp_df.tail()


# In[5]:


yelp_df.describe()


# In[6]:


yelp_df.info()


# # STEP #2: VISUALIZE DATASET

# In[7]:


# Let's get the length of the messages
yelp_df['length'] = yelp_df['text'].apply(len)
yelp_df.head()


# In[8]:


yelp_df['length'].plot(bins=100, kind='hist') 


# In[9]:


yelp_df.length.describe()


# In[10]:


# Let's see the longest message 43952
yelp_df[yelp_df['length'] == 4997]['text'].iloc[0]


# In[11]:


# Let's see the shortest message 
yelp_df[yelp_df['length'] == 1]['text'].iloc[0]


# In[12]:


# Let's see the message with mean length 
yelp_df[yelp_df['length'] == 710]['text'].iloc[0]


# In[13]:


sns.countplot(y = 'stars', data=yelp_df)


# In[16]:


g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=3)


# In[20]:


g = sns.FacetGrid(data=yelp_df, col='stars', col_wrap=5)
g.map(plt.hist, 'length', bins = 20, color = 'r')


# In[21]:


# Let's divide the reviews into 1 and 5 stars


# In[22]:


yelp_df_1 = yelp_df[yelp_df['stars']==1]


# In[23]:


yelp_df_5 = yelp_df[yelp_df['stars']==5]


# In[24]:


yelp_df_1


# In[25]:


yelp_df_5


# In[26]:


yelp_df_1_5 = pd.concat([yelp_df_1 , yelp_df_5])


# In[27]:


yelp_df_1_5


# In[28]:


yelp_df_1_5.info()


# In[29]:


print( '1-Stars percentage =', (len(yelp_df_1) / len(yelp_df_1_5) )*100,"%")


# In[30]:


print( '5-Stars percentage =', (len(yelp_df_5) / len(yelp_df_1_5) )*100,"%")


# In[31]:


sns.countplot(yelp_df_1_5['stars'], label = "Count") 


# # STEP #3: CREATE TESTING AND TRAINING DATASET/DATA CLEANING

# # STEP 3.1 EXERCISE: REMOVE PUNCTUATION

# In[32]:


import string
string.punctuation


# In[33]:


Test = 'Hello Mr. Future, I am so happy to be learning AI now!!'


# In[34]:


Test_punc_removed = [char for char in Test if char not in string.punctuation]
Test_punc_removed


# In[35]:


# Join the characters again to form the string.
Test_punc_removed_join = ''.join(Test_punc_removed)
Test_punc_removed_join


# # STEP 3.2 EXERCISE: REMOVE STOPWORDS

# In[36]:


# You have to download stopwords Package to execute this command
from nltk.corpus import stopwords
stopwords.words('english')


# In[37]:


Test_punc_removed_join


# In[38]:


Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]


# In[39]:


Test_punc_removed_join_clean # Only important (no so common) words are left


# In[ ]:


mini_challenge = 'Here is a mini challenge, that will teach you how to remove stopwords and punctuations!'


# In[ ]:


challege = [ char     for char in mini_challenge  if char not in string.punctuation    ]
challenge = ''.join(challege)
challenge = [  word for word in challenge.split() if word.lower() not in stopwords.words('english')  ] 


# # STEP 3.3 EXERCISE: COUNT VECTORIZER EXAMPLE 

# In[40]:


from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first document.','This document is the second document.','And this is the third one.','Is this the first document?']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)


# In[41]:


print(vectorizer.get_feature_names())


# In[42]:


print(X.toarray())  


# In[ ]:


mini_challenge = ['Hello World','Hello Hello World','Hello World world world']

vectorizer_challenge = CountVectorizer()
X_challenge = vectorizer_challenge.fit_transform(mini_challenge)
print(X_challenge.toarray())


# # LET'S APPLY THE PREVIOUS THREE PROCESSES TO OUR YELP REVIEWS EXAMPLE

# In[43]:


# Let's define a pipeline to clean up all the messages 
# The pipeline performs the following: (1) remove punctuation, (2) remove stopwords

def message_cleaning(message):
    Test_punc_removed = [char for char in message if char not in string.punctuation]
    Test_punc_removed_join = ''.join(Test_punc_removed)
    Test_punc_removed_join_clean = [word for word in Test_punc_removed_join.split() if word.lower() not in stopwords.words('english')]
    return Test_punc_removed_join_clean


# In[44]:


# Let's test the newly added function
yelp_df_clean = yelp_df_1_5['text'].apply(message_cleaning)


# In[45]:


print(yelp_df_clean[0]) # show the cleaned up version


# In[46]:


print(yelp_df_1_5['text'][0]) # show the original version


# # LET'S APPLY COUNT VECTORIZER TO OUR YELP REVIEWS EXAMPLE

# In[47]:


from sklearn.feature_extraction.text import CountVectorizer
# Define the cleaning pipeline we defined earlier
vectorizer = CountVectorizer(analyzer = message_cleaning)
yelp_countvectorizer = vectorizer.fit_transform(yelp_df_1_5['text'])


# In[48]:


print(vectorizer.get_feature_names())


# In[49]:


print(yelp_countvectorizer.toarray())  


# In[50]:


yelp_countvectorizer.shape


# # STEP#4: TRAINING THE MODEL WITH ALL DATASET

# In[51]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
label = yelp_df_1_5['stars'].values


# In[52]:


label


# In[53]:


NB_classifier.fit(yelp_countvectorizer, label)


# In[54]:


testing_sample = ['amazing food! highly recommmended']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# In[55]:


testing_sample = ['shit food, made me sick']
testing_sample_countvectorizer = vectorizer.transform(testing_sample)
test_predict = NB_classifier.predict(testing_sample_countvectorizer)
test_predict


# # STEP#4: DIVIDE THE DATA INTO TRAINING AND TESTING PRIOR TO TRAINING

# In[56]:


X = yelp_countvectorizer
y = label


# In[57]:


X.shape


# In[58]:


y.shape


# In[59]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[60]:


from sklearn.naive_bayes import MultinomialNB

NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)


# In[ ]:


# from sklearn.naive_bayes import GaussianNB 
# NB_classifier = GaussianNB()
# NB_classifier.fit(X_train, y_train)


# # STEP#5: EVALUATING THE MODEL 

# In[61]:


from sklearn.metrics import classification_report, confusion_matrix


# In[62]:


y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)


# In[63]:


# Predicting the Test set results
y_predict_test = NB_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot=True)


# In[64]:


print(classification_report(y_test, y_predict_test))


# # STEP #6: LET'S ADD ADDITIONAL FEATURE TF-IDF

# - Tf–idf stands for "Term Frequency–Inverse Document Frequency" is a numerical statistic used to reflect how important a word is to a document in a collection or corpus of documents. 
# - TFIDF is used as a weighting factor during text search processes and text mining.
# - The intuition behing the TFIDF is as follows: if a word appears several times in a given document, this word might be meaningful (more important) than other words that appeared fewer times in the same document. However, if a given word appeared several times in a given document but also appeared many times in other documents, there is a probability that this word might be common frequent word such as 'I' 'am'..etc. (not really important or meaningful!).
# 
# 
# - TF: Term Frequency is used to measure the frequency of term occurrence in a document: 
#     - TF(word) = Number of times the 'word' appears in a document / Total number of terms in the document
# - IDF: Inverse Document Frequency is used to measure how important a term is: 
#     - IDF(word) = log_e(Total number of documents / Number of documents with the term 'word' in it).
# 
# - Example: Let's assume we have a document that contains 1000 words and the term “John” appeared 20 times, the Term-Frequency for the word 'John' can be calculated as follows:
#     - TF|john = 20/1000 = 0.02
# 
# - Let's calculate the IDF (inverse document frequency) of the word 'john' assuming that it appears 50,000 times in a 1,000,000 million documents (corpus). 
#     - IDF|john = log (1,000,000/50,000) = 1.3
# 
# - Therefore the overall weight of the word 'john' is as follows 
#     - TF-IDF|john = 0.02 * 1.3 = 0.026

# In[65]:


yelp_countvectorizer


# In[74]:


from sklearn.feature_extraction.text import TfidfTransformer

yelp_tfidf = TfidfTransformer().fit_transform(yelp_countvectorizer)
print(yelp_tfidf.shape)


# In[75]:


yelp_tfidf


# In[76]:


print(yelp_tfidf[:,:])
# Sparse matrix with all the values of IF-IDF


# In[80]:


X = yelp_tfidf
y = label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix
y_predict_train = NB_classifier.predict(X_train)
y_predict_train
cm = confusion_matrix(y_train, y_predict_train)
sns.heatmap(cm, annot=True)

