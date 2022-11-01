#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re
import spacy
import tarfile
import unicodedata

from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import SGDClassifier, LogisticRegression
import sklearn.model_selection as ms
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

from xgboost import XGBClassifier

pd.set_option('max_colwidth', 500)


# # Load data

# In[2]:


with tarfile.open("rt-polaritydata.tar.gz", "r") as f:
    # check what's in the file
    print(f.getmembers())
    # and extract the data
    
    import os
    
    def is_within_directory(directory, target):
        
        abs_directory = os.path.abspath(directory)
        abs_target = os.path.abspath(target)
    
        prefix = os.path.commonprefix([abs_directory, abs_target])
        
        return prefix == abs_directory
    
    def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
    
        for member in tar.getmembers():
            member_path = os.path.join(path, member.name)
            if not is_within_directory(path, member_path):
                raise Exception("Attempted Path Traversal in Tar File")
    
        tar.extractall(path, members, numeric_owner=numeric_owner) 
        
    
    safe_extract(f)


# In[3]:


# load the separate positive and negative reviews datasets, 
# concatenate, and create a train and a test set
# for further analysis/model creation

with open('rt-polaritydata/rt-polarity.pos') as f:
    lines = f.readlines()
    pos = pd.DataFrame({'review': lines, 'target': [1]*len(lines)})
    
with open('rt-polaritydata/rt-polarity.neg') as f:
    lines = f.readlines()
    neg = pd.DataFrame({'review': lines, 'target': [0]*len(lines)})
    
data = pd.concat([pos, neg])

train_val, test = ms.train_test_split(
    data, test_size = 0.2, shuffle=True, random_state=42
)


# In[4]:


train_val.sample(5)


# # Preprocessing

# In[5]:


nlp = spacy.load('en_core_web_lg')

class Cleaner(object):
    """
    callable class for cleaning text data
    -- encodes text as ascii (ignoring special characters)
    -- extracts words (latin letters, while also preserving contractions (symbol "'"))
    """
    WORDS = re.compile(r"([a-z]+'?[a-z]*)")
    
    def __call__(self, text):
        text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
        text = Cleaner.WORDS.findall(text)
        return ' '.join(text)
    

class Tokenizer(object):
    """
    callable class for tokenizing and lemmatizing text data
    -- lemmatizes text input
    -- converts negative sequence to 'NOT_myword' ('do not work' --> 'NOT_work')
    """
    NEGATIONS = re.compile(r"((do not|be not|not)\s(\w+))")
    def __call__(self, text):
        tokens = nlp(text)
        # lemmatize, except remove 'determiners'
        lemmas = ' '.join([
            token.lemma_ for token in tokens
            if token.pos_!='DET'
            and not "'" in token.lemma_
            and len(token.lemma_) > 1
        ])
        newtext, _ = Tokenizer.NEGATIONS.subn(r'NOT_\3', lemmas)
        return newtext.split()


class TotalW2VTransformer(TransformerMixin):
    """
    Transformer class that takes a pd.Series object containing strings
    and transforms each row/string to a total (sum) 300d word2vec representation
    """
    def fit(self, X, y=None, **fit_params):
        return self
    
    def _get_total_vector(self, text: str):
        """
        returns a summed word2vec representation of text
        """
        tokens = nlp(text)
        total_vector = np.zeros([300])
        for token in tokens:
            total_vector += token.vector
        return total_vector

    def transform(self, X, y=None, **fit_params):
        cleaner = Cleaner()
        return np.array([self._get_total_vector(cleaner(row)) for row in X.values])


# # Model 1 - support vector machines

# In[19]:


# cleaned and tokenized data goes to the tf-idf vectorizer
tfidf = TfidfVectorizer(
    preprocessor=Cleaner(), tokenizer=Tokenizer(), analyzer='word',
    ngram_range=(1,3), max_features=6000, sublinear_tf=True, norm='l2'
)

# additionally, I create a 300d word2vec representation of each review
create_total_vector = Pipeline([
    ('total_vect', TotalW2VTransformer()),
    ('scaler', StandardScaler())
])

# linear SVM classifier
svm = LinearSVC(penalty='l2', loss='squared_hinge', dual=False, tol=1e-4, C=0.1, max_iter=3000)

# final pipeline, where tfidf vectorized data is concatenated
# with the word2vec transformation and goes to the svm classifier
pipe_svm = Pipeline([
    ('preproc', FeatureUnion([
        ('tfidf', tfidf),
        ('total_vect_scaled', create_total_vector)
    ])),
    ('clf', svm)
])


# In[21]:


# evaluate using 5-fold cross-validation
cv_score_svm = ms.cross_val_score(
    pipe_svm, train_val.review, train_val.target, scoring='accuracy', cv=5
)


# In[24]:


# cross validation accuracy
print('cv accuracies:', cv_score_svm)
print('mean accuracy:', cv_score_svm.mean())
print('std accuracy:', cv_score_svm.std())


# In[25]:


# retrain on train_validation sets
pipe_svm.fit(train_val.review, train_val.target)

# predict on test set
y_pred_test = pipe_svm.predict(test.review)
print('test set accuracy:', metrics.accuracy_score(test.target, y_pred_test))


# In[28]:


metrics.confusion_matrix(test.target, y_pred_test)


# # Model 2 - xgboost

# In[52]:


# cleaned and tokenized data goes to the tf-idf vectorizer
countvect = CountVectorizer(
    preprocessor=Cleaner(), tokenizer=Tokenizer(), analyzer='word',
    ngram_range=(1,2), max_features=1000
)

# xgboost classifier
xgb = XGBClassifier(
    max_depth=3, learning_rate=0.1, n_estimators=1000, verbosity=1,
    objective='binary:logistic', reg_lambda=1
)

# final pipeline, where tfidf vectorized data is concatenated
# with the word2vec transformation and goes to the xgboost classifier
pipe_xgb = Pipeline([
    ('preproc', FeatureUnion([
        ('vectorizer', countvect),
        ('total_vect', TotalW2VTransformer())
    ])),
    ('clf', xgb)
])


# In[54]:


# evaluate using 5-fold cross-validation
cv_score_xgb = ms.cross_val_score(
    pipe_xgb, train_val.review, train_val.target, scoring='accuracy', cv=5
)


# In[55]:


# cross validation accuracy
print('cv accuracies:', cv_score_xgb)
print('mean accuracy:', cv_score_xgb.mean())
print('std accuracy:', cv_score_xgb.std())


# In[56]:


# retrain on train_validation sets
pipe_xgb.fit(train_val.review, train_val.target)

# predict on test set
y_pred_test = pipe_xgb.predict(test.review)
print('test set accuracy:', metrics.accuracy_score(test.target, y_pred_test))


# In[57]:


metrics.confusion_matrix(test.target, y_pred_test)

