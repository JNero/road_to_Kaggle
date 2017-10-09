# -*- coding: utf-8 -*-
# @Time    : 17-10-9 上午11:39
# @Author  : QIAO

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

if __name__ == '__main__':
    news=fetch_20newsgroups(subset='all')
    X_train, X_test, y_train, y_test = train_test_split(
        news.data, news.target, test_size=0.25, random_state=33)
    vec=CountVectorizer()
    X_train=vec.fit_transform(X_train)
    X_test=vec.transform(X_test)
    mnb=MultinomialNB()
    mnb.fit(X_train,y_train)
    y_predict=mnb.predict(X_test)
    print("The accuracy of Naive Bayes Classifier is",mnb.score(X_test,y_test))
    print(classification_report(y_test,y_predict,target_names=news.target_names))
