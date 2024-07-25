#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('amazon_alexa.csv')
df


# In[4]:


df.head()


# In[5]:


df['rating'].isna().sum()


# In[6]:


df.info()


# In[7]:


print(np.unique(df.rating))


# In[8]:


import matplotlib.pyplot as plt   #pie chart to visulaize the data
y=np.unique(df.rating)
mylabels=[1, 2, 3, 4, 5]
plt.pie(y, labels=y, autopct='%d', explode=(0.1, 0.1, 0.1, 0.1, 0.1))
plt.legend(title='rating', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


# In[9]:


def marking(rating):
    if(int(rating)<2):
        return 0
    else:
        return 1


# In[10]:


df['rating']=df['rating'].apply(marking)


# In[11]:


df.head()


# In[12]:


df1=df.drop(['date','variation', 'feedback'], axis=1)


# In[13]:


df1.rating.value_counts()


# # lemmatisation and token generation

# In[14]:


import re
import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')


# In[23]:


df2=df1.rename(columns={"verified_reviews": "review_description"})
df2


# In[24]:


df2['review_description']=df2['review_description'].fillna('')


# In[26]:


df2['review_description'] = df2['review_description'].apply(lambda x: x.lower()) #changing the data to lowercase
df2['review_description'] = df2['review_description'].apply(lambda x: re.sub('[,\.!?:()"]', '', x)) #removing special characters
df2['review_description'] = df2['review_description'].apply(lambda x: re.sub('[^a-zA-Z"]', ' ', x))
df2['review_description'] = df2['review_description'].apply(lambda x: x.strip()) #removing space


# In[27]:


df2.head()


# In[28]:


sw=stopwords.words("english")
df2['review_description'] = df2['review_description'].apply(lambda x: " ".join(x for x in x.split() if x not in sw))


# In[29]:


import wordcloud
import matplotlib.pyplot as plt
common_words=""
for i in df2.review_description:
    i=str(i)
    tokens=i.split()
    common_words += " ".join(tokens) + " "
wordcloud = wordcloud.WordCloud(font_path=None).generate(common_words)
plt.imshow(wordcloud)
plt.show()


# In[30]:


from textblob import Word
#nltk.download('wordnet') #large database which return lexical form
df2['review_description']=df2['review_description'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))


# In[31]:


#nltk.download('punkt')
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
df2['review_description'] = df2['review_description'].apply(lambda x: " ".join([porter_stemmer.stem(word) for word in x.split()]))


# # Splitting data

# In[32]:


from sklearn.model_selection import train_test_split
train_x, test_x, y_train,y_test=train_test_split(df2['review_description'],
                                                 df2["rating"],
                                                 test_size=0.20,
                                                 random_state=42)
print("X_train shape",train_x.shape)
print("y_train shape",test_x.shape)
print("X_test shape",y_train.shape)
print("y_test shape",y_test.shape)


# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cv_train=cv.fit_transform(train_x)
cv_test=cv.transform(test_x)
print('BOW_cv_train:',cv_train.shape)
print('BOW_cv_test:',cv_test.shape)


# In[34]:


tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tv_train=tv.fit_transform(train_x)
tv_test=tv.transform(test_x)
print('Tfidf_train:',tv_train.shape)
print('Tfidf_test:',tv_test.shape)


# # logistic Regression

# In[35]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
lr_bow=lr.fit(cv_train,y_train)
print(lr_bow)

lr_tfidf=lr.fit(tv_train,y_train)
print(lr_tfidf)


# In[36]:


lr_bow_predict=lr.predict(cv_test)
print(lr_bow_predict)

lr_tfidf_predict=lr.predict(tv_test)
print(lr_tfidf_predict)


# In[37]:


from sklearn.metrics import accuracy_score
lr_bow_score=accuracy_score(y_test,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)

lr_tfidf_score=accuracy_score(y_test, lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)


# In[38]:


from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb_bow=mnb.fit(cv_train,y_train)
print(mnb_bow)

mnb_tfidf=mnb.fit(tv_train,y_train)
print(mnb_tfidf)


# In[39]:


mnb_bow_predict=mnb.predict(cv_test)
print(mnb_bow_predict)

mnb_tfidf_predict=mnb.predict(tv_test)
print(mnb_tfidf_predict)


# # Accuracy Score

# In[40]:


mnb_bow_score=accuracy_score(y_test,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)

mnb_tfidf_score=accuracy_score(y_test,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)


# In[ ]:




