# -*- coding: utf-8 -*-
# @Time    : 17-10-9 下午2:16
# @Author  : QIAO

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

if __name__ == '__main__':
    titanic = pd.read_csv(
        'http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
    X = titanic[['pclass', 'age', 'sex']]
    y = titanic['survived']
    X['age'].fillna(X['age'].mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=33)
    vec = DictVectorizer(sparse=False)
    X_train = vec.fit_transform(X_train.to_dict(orient='record'))
    X_test = vec.transform(X_test.to_dict(orient='record'))
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    dtc_y_pred = dtc.predict(X_test)
    rfc = RandomForestClassifier()
    gbc = GradientBoostingClassifier()
    gbc.fit(X_train, y_train)
    gbc_y_pred = gbc.predict(X_test)
