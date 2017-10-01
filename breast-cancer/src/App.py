# -*- coding: utf-8 -*-
# @Time    : 17-9-30 下午1:55
# @Author  : QIAO

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    column_names = [
        'samle code number',
        'clump Thickness',
        'Uniformity of Cell Size',
        'uniformity  of Cell shape',
        'Marginal Adhesion',
        'Single Epitheila Cell Size',
        'Bare Nuclei',
        'Bland chromatin',
        'Normal Nucleoli',
        'Mitoses',
        'Class']

    data = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',
        names=column_names)
    data = data.replace(to_replace='?', value=np.nan)
    data = data.dropna(how='any')
    print(data.shape)
    X_train, X_test, y_train, y_test = train_test_split(
        data[column_names[1:10]], data[column_names[10]], test_size=0.25, random_state=33)
    print(y_train.value_counts())
    ss = StandardScaler()
    X_train = ss.fit_transform(X_train)
    X_test = ss.transform(X_test)
    lr = LogisticRegression()
    sgdc = SGDClassifier()
