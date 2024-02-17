import pandas as pd 
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

#pd.set_option('display.max_colwidth', None)

text = tarfile.open('assignment3/20021010_easy_ham.tar.bz2', 'r:bz2')
text.extractall('assignment3')

text = tarfile.open('assignment3/20021010_hard_ham.tar.bz2', 'r:bz2')
text.extractall('assignment3')

text = tarfile.open('assignment3/20021010_spam.tar.bz2', 'r:bz2')
text.extractall('assignment3')

#file = open('assignment3/easy_ham/0005.8c3b9e9c0f3f183ddaf7592a11b99957', encoding='UTF-8')
df_easy_ham = pd.DataFrame(columns=['Text', 'Spam?'])
df_hard_ham = pd.DataFrame(columns=['Text', 'Spam?'])
i = 0
j = 0
for filename in os.listdir('assignment3/easy_ham/'):
    file = open('assignment3/easy_ham/'+ filename).read()
    df_easy_ham.loc[i, ['Text', 'Spam?']] = [file, 0] 
    i += 1

for filename in os.listdir('assignment3/hard_ham/'):
    file = open('assignment3/hard_ham/'+filename).read()
    df_hard_ham.loc[j, ['Text', 'Spam?']] = [file, 0] 
    j += 1

for filename in os.listdir('assignment3/spam/'):
    file = open('assignment3/spam/'+filename, encoding='iso-8859-1').read()
    df_easy_ham.loc[i, ['Text', 'Spam?']] = [file, 1] 
    df_hard_ham.loc[j, ['Text', 'Spam?']] = [file, 1] 
    i += 1
    j += 1



df_easy_ham_train, df_easy_ham_test = train_test_split(df_easy_ham)

cv = CountVectorizer()
X_train = cv.fit_transform(df_easy_ham_train['Text'])
X_test = cv.transform(df_easy_ham_test['Text'])

le = LabelEncoder()
y_train = le.fit_transform(df_easy_ham_train['Spam?'])
y_test = le.transform(df_easy_ham_test['Spam?'])

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)

print('Bernoulli Ham:')
print( f'Accuracy: {accuracy_score(y_test, y_pred_bnb)}' )
print( f'Precision: {precision_score(y_test, y_pred_bnb)}' )
print( f'Recall: {recall_score(y_test, y_pred_bnb)}' )
print( f'Confusion matrix: {confusion_matrix(y_test, y_pred_bnb)}' ) 

print('\nMultinomial Ham:')
print( f'Accuracy: {accuracy_score(y_test, y_pred_mnb)}' )
print( f'Precision: {precision_score(y_test, y_pred_mnb)}' )
print( f'Recall: {recall_score(y_test, y_pred_mnb)}' )
print( f'Confusion matrix: {confusion_matrix(y_test, y_pred_mnb)}'  )

# Hard ham
df_hard_ham_train, df_hard_ham_test = train_test_split(df_hard_ham)

cv = CountVectorizer()
X_train = cv.fit_transform(df_hard_ham_train['Text'])
X_test = cv.transform(df_hard_ham_test['Text'])

y_train = df_hard_ham_train['Spam?'].astype(np.int64)
y_test = df_hard_ham_test['Spam?'].astype(np.int64)

bnb = BernoulliNB()
bnb.fit(X_train, y_train)
y_pred_bnb = bnb.predict(X_test)

mnb = MultinomialNB()
mnb.fit(X_train,y_train)
y_pred_mnb = mnb.predict(X_test)

print('\nBernoulli Hard Ham:')
print( f'Accuracy: {accuracy_score(y_test, y_pred_bnb)}' )
print( f'Precision: {precision_score(y_test, y_pred_bnb)}' )
print( f'Recall: {recall_score(y_test, y_pred_bnb)}' )
print( f'Confusion matrix: {confusion_matrix(y_test, y_pred_bnb)}' ) 

print('\nMultinomial Hard Ham:')
print( f'Accuracy: {accuracy_score(y_test, y_pred_mnb)}' )
print( f'Precision: {precision_score(y_test, y_pred_mnb)}' )
print( f'Recall: {recall_score(y_test, y_pred_mnb)}' )
print( f'Confusion matrix: {confusion_matrix(y_test, y_pred_mnb)}'  )

